#!/usr/bin/env python3
"""Multi-rank ZeRO-3 test of the fused lm_head+CE path in Actor.forward.

Runs the SAME model twice on the same rank:
  1) reference: logits path (OPENRLHF_FUSED_CE unset)
  2) fused:     GatheredParameters(lm_head.weight, fwd_module=self) path

and compares loss + per-parameter grads (rank-local ZeRO-3 partitions, so a
per-rank comparison is exact). Verifies the deepspeed external-parameter
mechanics (gather for forward, no external grad to scatter since the lm_head
base weight is frozen under LoRA) don't deadlock or assert.

Usage: deepspeed --include localhost:2,4 test_fused_z3.py --seqlen 1024

Do not combine CUDA_VISIBLE_DEVICES with --num_gpus: DeepSpeed 0.19 ignores
CUDA_VISIBLE_DEVICES in that combination and may remap onto physical GPU 0.
"""
import argparse
import os

import torch

from openrlhf.models import Actor
from openrlhf.utils import get_strategy

# Standard Qwen2.5 LoRA projections — EXCLUDES lm_head on purpose: the fused
# path reads the frozen base lm_head.weight directly, so training LoRA on
# lm_head would silently diverge from the reference path.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqlen", type=int, default=1024)
    parser.add_argument(
        "--torch-optimizer",
        action="store_true",
        help="Use client-created torch AdamW so the smoke does not require the DeepSpeed FusedAdam extension.",
    )
    parser.add_argument(
        "--disable-gradient-checkpointing",
        action="store_true",
        help="Disable HF activation checkpointing for the short ZeRO-3 correctness smoke.",
    )
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--ds.zero_stage", type=int, default=3)
    parser.add_argument("--ds.param_dtype", type=str, default="bf16")
    parser.add_argument("--ds.adam_offload", action="store_true", default=False)
    parser.add_argument("--ds.zpg", type=int, default=1)
    parser.add_argument("--ds.use_universal_ckpt", action="store_true", default=False)
    parser.add_argument("--ds.grad_accum_dtype", type=str, default=None)
    parser.add_argument("--ds.overlap_comm", action="store_true", default=False)
    parser.add_argument("--ds.deepcompile", action="store_true", default=False)
    parser.add_argument("--ds.tensor_parallel_size", type=int, default=1)
    parser.add_argument("--ds.ring_attn_size", type=int, default=1)
    parser.add_argument("--ds.ring_attn_head_stride", type=int, default=1)
    parser.add_argument("--ds.lora.rank", type=int, default=16)
    parser.add_argument("--ds.lora.alpha", type=int, default=32)
    parser.add_argument("--ds.lora.target_modules", type=str, nargs="*", default=LORA_TARGETS)
    parser.add_argument("--ds.lora.dropout", type=float, default=0)
    parser.add_argument("--ds.load_in_4bit", action="store_true", default=False)
    parser.add_argument("--ds.packing_samples", action="store_true", default=False)
    parser.add_argument("--ds.use_liger_kernel", action="store_true", default=False)
    parser.add_argument("--ds.attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--ds.experts_implementation", type=str, default=None)
    parser.add_argument("--model.gradient_checkpointing_enable", action="store_true", default=True)
    parser.add_argument("--model.gradient_checkpointing_reentrant", action="store_true", default=False)
    args = parser.parse_args()
    os.environ["TORCH_DISTRIBUTED_INIT"] = "default"

    from openrlhf.utils.config import hierarchize

    args = hierarchize(args)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.train = argparse.Namespace(seed=42, batch_size=world_size, micro_batch_size=1, max_epochs=3)
    args.data = argparse.Namespace(max_len=args.seqlen, multiturn=False, apply_chat_template=False,
                                   max_samples=1000000, dataset_split="train", input_template="User: {}\nAssistant: ",
                                   input_key="input", output_key=None)
    args.ckpt = argparse.Namespace(output_dir="/tmp/probe", save_steps=-1, save_hf=False, disable_ds=False,
                                   path="./ckpt/checkpoints_sft", max_num=3, max_mem=int(1e8), load_enable=False)
    args.logger = argparse.Namespace(logging_steps=5)
    args.eval = argparse.Namespace(steps=-1, dataset=None, split="train")
    args.model = argparse.Namespace(model_name_or_path="/media/imc/data/yzy/agent/project2/phase1/models/Qwen2.5-Coder-7B-Instruct",
                                    pretrain_mode_enable=False, gradient_checkpointing_enable=True,
                                    gradient_checkpointing_reentrant=args.model.gradient_checkpointing_reentrant,
                                    aux_loss_coef=0)
    strategy = get_strategy(args)
    strategy.setup_distributed()

    model = Actor(
        "/media/imc/data/yzy/agent/project2/phase1/models/Qwen2.5-Coder-7B-Instruct",
        attn_implementation="flash_attention_2",
        param_dtype="bf16",
        lora_rank=16,
        lora_alpha=32,
        target_modules=LORA_TARGETS,
        lora_dropout=0,
        ds_config=strategy.get_ds_train_config(is_actor=True),
        packing_samples=False,
        use_liger_kernel=False,
    )
    if os.environ.get("Z3_PRINT_MODEL"):
        strategy.print(model)
    actor = model
    if not args.disable_gradient_checkpointing:
        actor.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.model.gradient_checkpointing_reentrant}
        )
    cfg = dict(optim="adam", muon=None, adam=dict(lr=5e-5, betas=(0.9, 0.95), eps=1e-8, weight_decay=0),
               lr_scheduler="cosine_with_min_lr", lr_warmup_ratio=0.03, min_lr_ratio=0.1, max_norm=1.0,
               scheduler_steps=100)
    if args.torch_optimizer:
        # The runtime-only CUDA container intentionally has no compiler toolchain,
        # so DeepSpeed's config-created AdamW would try to JIT-build FusedAdam and
        # fail before ZeRO-3 starts. A client optimizer exercises the same ZeRO-3
        # partition/gather/backward path without making compiler availability part
        # of this correctness smoke.
        import deepspeed

        raw_model, model_parameters = strategy._get_model_parameters(model, optim="adam", weight_decay=0)
        optim = torch.optim.AdamW(
            model_parameters,
            lr=cfg["adam"]["lr"],
            betas=cfg["adam"]["betas"],
            eps=cfg["adam"]["eps"],
            weight_decay=cfg["adam"]["weight_decay"],
        )
        ds_config = strategy.get_ds_train_config(optim_dict=None, max_norm=cfg["max_norm"])
        ds_config["zero_allow_untested_optimizer"] = True
        engine, optim, _, scheduler = deepspeed.initialize(
            model=raw_model,
            optimizer=optim,
            model_parameters=model_parameters,
            config=ds_config,
            args={"local_rank": int(os.environ.get("LOCAL_RANK", "-1"))},
            dist_init_required=True,
        )
        model.model = engine
    else:
        model, optim, scheduler = strategy.prepare((model, cfg))
    if torch.distributed.get_rank() == 0:
        lay = actor.model.module.model.model.layers[0]
        print(f"[Z3] actor.training={actor.training} layer0.gc={lay.gradient_checkpointing}", flush=True)
    if os.environ.get("Z3_DEBUG_CKPT"):
        import functools

        import torch.utils.checkpoint as tk

        def dbg(fn):
            return functools.partial(tk.checkpoint, use_reentrant=False, debug=True)

        for lay in actor.model.module.model.model.layers:
            lay._gradient_checkpointing_func = dbg(lay._gradient_checkpointing_func)
            for m in lay.modules():
                if hasattr(m, "_gradient_checkpointing_func"):
                    m._gradient_checkpointing_func = dbg(m._gradient_checkpointing_func)
    actor.train()

    from openrlhf.models import SFTLoss

    loss_fn = SFTLoss(token_level_loss=True)
    seqlen = args.seqlen
    torch.manual_seed(0)
    ids = torch.randint(100, 50000, (1, seqlen), dtype=torch.long).cuda()
    attn = torch.ones_like(ids)
    mask = torch.ones_like(ids).bool()[:, :-1]
    rank = torch.distributed.get_rank()

    engine = model.model  # prepare returns the Actor; .model is the DeepSpeedEngine

    def run_once(fused: bool, zero_first: bool):
        if zero_first:
            engine.zero_grad()
        if fused:
            os.environ["OPENRLHF_FUSED_CE"] = "1"
        else:
            os.environ.pop("OPENRLHF_FUSED_CE", None)
        # ZeRO-3 consumes/partitions gradients during backward, so reading
        # ``p.grad`` afterwards commonly returns None. Capture autograd's
        # per-parameter gradients before DeepSpeed clears the public fields.
        grads = {}
        handles = []
        for name, param in engine.module.named_parameters():
            if not param.requires_grad:
                continue

            def capture(grad, param_name=name):
                grads[param_name] = grad.detach().clone()
                return grad

            handles.append(param.register_hook(capture))
        lp, out = model(ids, attention_mask=attn, return_output=True, return_logprobs=True)
        loss = loss_fn(lp, mask)
        # engine backward = what OpenRLHF's strategy.backward does (plain
        # loss.backward() never populates ZeRO-3 partitioned grads)
        try:
            engine.backward(loss)
            torch.cuda.synchronize()
        finally:
            for handle in handles:
                handle.remove()
        os.environ.pop("OPENRLHF_FUSED_CE", None)
        return loss.detach(), grads

    try:
        loss_ref, grads_ref = run_once(fused=False, zero_first=False)
        print(f"[Z3] rank{rank} ref  loss={loss_ref.item():.6f} n_grads={len(grads_ref)}", flush=True)
        calls_before = getattr(actor, "_fused_ce_calls", 0)
        loss_fus, grads_fus = run_once(fused=True, zero_first=True)
        calls_after = getattr(actor, "_fused_ce_calls", 0)
        print(f"[Z3] rank{rank} fused loss={loss_fus.item():.6f} n_grads={len(grads_fus)}", flush=True)
        assert calls_after == calls_before + 1, (
            f"fused branch was not observed: calls_before={calls_before}, calls_after={calls_after}"
        )
        torch.distributed.barrier()

        loss_abs = (loss_ref - loss_fus).abs().item()
        if rank == 0:
            print(f"[Z3] loss |ref-fused| = {loss_abs:.3e}")
        assert loss_abs < 5e-3, f"loss mismatch: {loss_abs:.3e}"

        # compare rank-local grad partitions
        keys = sorted(grads_ref.keys() & grads_fus.keys())
        missing = (set(grads_ref) ^ set(grads_fus))
        if rank == 0:
            print(f"[Z3] params with grads: ref={len(grads_ref)} fused={len(grads_fus)} "
                  f"common={len(keys)} asymmetric={sorted(missing)[:5]}")
        assert grads_ref, "no trainable-parameter gradients were captured"
        assert not missing, f"asymmetric grads: {sorted(missing)}"
        worst = 0.0
        worst_name = None
        worst_abs = 0.0
        worst_scale = 0.0
        max_abs = 0.0
        total_scale = 0.0
        diff_sq = 0.0
        ref_sq = 0.0
        fused_sq = 0.0
        dot = 0.0
        for n in keys:
            g = grads_ref[n].float()
            f = grads_fus[n].float()
            scale = g.abs().max().item()
            total_scale = max(total_scale, scale)
            d = (g - f).abs().max().item()
            max_abs = max(max_abs, d)
            diff_sq += (g - f).square().sum().item()
            ref_sq += g.square().sum().item()
            fused_sq += f.square().sum().item()
            dot += (g * f).sum().item()
            if scale > 1e-8 and d / scale > worst:
                worst = d / scale
                worst_name = n
                worst_abs = d
                worst_scale = scale
        global_rel_l2 = (diff_sq / max(ref_sq, 1e-30)) ** 0.5
        cosine = dot / max((ref_sq * fused_sq) ** 0.5, 1e-30)
        if rank == 0:
            print(
                f"[Z3] grad global-rel-l2={global_rel_l2:.3e} cosine={cosine:.8f} "
                f"max-abs={max_abs:.3e}; max-rel={worst:.3e} @ {worst_name} "
                f"(that abs={worst_abs:.3e}, param max|g|={worst_scale:.3e}, global max|g|={total_scale:.3e})"
            )
        # BF16 fused CE is judged by aggregate gradient fidelity plus an
        # absolute guard. A per-parameter max-relative metric is unstable for
        # small gradients and is retained above as a diagnostic only.
        assert global_rel_l2 < 2e-2, f"gradient global relative L2 mismatch: {global_rel_l2:.3e}"
        assert cosine > 0.999, f"gradient cosine mismatch: {cosine:.8f}"
        assert max_abs < 2e-2, f"gradient max absolute mismatch: {max_abs:.3e}"
        if rank == 0:
            print(f"[Z3] PASS: fused == reference (loss + grads, {world_size}-rank ZeRO-3)")
    except Exception as e:
        rank = torch.distributed.get_rank()
        print(f"[Z3] rank{rank} EXC: {type(e).__name__}: {e}")
        torch.distributed.destroy_process_group()
        os._exit(1)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
