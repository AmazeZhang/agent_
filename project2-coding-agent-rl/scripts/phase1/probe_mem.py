#!/usr/bin/env python3
"""Minimal OpenRLHF SFT memory probe — mirrors train_sft.py setup exactly,
runs ONE micro-batch forward+backward, dumps memory at each stage.

Usage (inside tmux, after checking every selected physical GPU):
  deepspeed --include localhost:2,4,6,7 probe_mem.py --seqlen 16000

Never combine CUDA_VISIBLE_DEVICES with DeepSpeed --num_gpus: DeepSpeed 0.19
may ignore that filter and remap workers onto physical GPU 0.
Env: PROBE_REENTRANT=1 to test use_reentrant=True checkpointing.
"""
import argparse
import os

import torch

from openrlhf.models import Actor
from openrlhf.utils import get_strategy, get_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqlen", type=int, default=16000)
    parser.add_argument("--fused", action="store_true", default=False,
                        help="use fused lm_head+CE branch (OPENRLHF_FUSED_CE=1)")
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
    parser.add_argument("--ds.lora.target_modules", type=str, nargs="*", default="all-linear")
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
        # Explicit Qwen2.5 projections, lm_head EXCLUDED: the fused CE path reads
        # the frozen base lm_head.weight directly — LoRA on lm_head would silently
        # never train under OPENRLHF_FUSED_CE=1 and diverge from the logits path.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        ds_config=strategy.get_ds_train_config(is_actor=True),
        packing_samples=False,
        use_liger_kernel=False,
    )
    strategy.print(model)

    tokenizer = get_tokenizer(
        "/media/imc/data/yzy/agent/project2/phase1/models/Qwen2.5-Coder-7B-Instruct",
        model.model, "right", strategy, use_fast=True,
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": args.model.gradient_checkpointing_reentrant}
    )
    cfg = dict(optim="adam", muon=None, adam=dict(lr=5e-5, betas=(0.9, 0.95), eps=1e-8, weight_decay=0),
               lr_scheduler="cosine_with_min_lr", lr_warmup_ratio=0.03, min_lr_ratio=0.1, max_norm=1.0,
               scheduler_steps=100)
    actor = model
    model, optim, scheduler = strategy.prepare((model, cfg))
    if torch.distributed.get_rank() == 0:
        lay = actor.model.module.model.model.layers[0]
        print(f"[PROBE] actor.training={actor.training} hf.training={actor.model.training} "
              f"layer0.gc={lay.gradient_checkpointing} layer0.func={lay._gradient_checkpointing_func}", flush=True)
        torch.cuda.reset_peak_memory_stats()
    actor.train()

    rank = torch.distributed.get_rank()
    print(f"[PROBE] rank{rank} physical device: {torch.cuda.current_device()}", flush=True)
    if rank == 0:
        cfg = model.model.config
        print(f"[PROBE] _attn_implementation={getattr(cfg, '_attn_implementation', None)}", flush=True)
        print(f"[PROBE] use_flash_attn={getattr(cfg, 'use_flash_attn', None)}", flush=True)
        tot = sum(p.numel() for p in model.model.parameters())
        trn = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        print(f"[PROBE] total_params={tot/1e9:.2f}B trainable={trn/1e6:.1f}M", flush=True)

    seqlen = args.seqlen
    torch.manual_seed(0)
    ids = torch.randint(100, 50000, (1, seqlen), dtype=torch.long).cuda()
    attn = torch.ones_like(ids)
    labels = ids.clone()
    try:
        torch.cuda.memory._record_memory_history(True, trace_alloc_max_entries=200000)
    except Exception as e:
        print(f"[PROBE] mem-history unavailable: {e}", flush=True)

    def dump(tag):
        if torch.distributed.get_rank() == 0:
            a = torch.cuda.memory_allocated() / 1e9
            r = torch.cuda.memory_reserved() / 1e9
            p = torch.cuda.max_memory_allocated() / 1e9
            print(f"[PROBE] {tag}: allocated={a:.2f} GiB reserved={r:.2f} GiB peak={p:.2f} GiB")

    dump("after-load")
    torch.cuda.empty_cache()
    dump("after-load(empty_cache)")
    if args.fused:
        os.environ["OPENRLHF_FUSED_CE"] = "1"
        print(f"[PROBE] rank{torch.distributed.get_rank()} using FUSED lm_head+CE branch", flush=True)
    # mirror SFTTrainer exactly: return_logprobs=True + SFTLoss (no full-vocab CE)
    from openrlhf.models import SFTLoss

    loss_fn = SFTLoss(token_level_loss=True)
    try:
        per_token_log_probs, output = model(
            ids, attention_mask=attn, return_output=True, return_logprobs=True
        )
        dump("after-forward")
        loss = loss_fn(per_token_log_probs, torch.ones_like(labels).bool()[:, :-1])
        loss.backward()
        dump("after-backward")
    except Exception as e:
        rank = torch.distributed.get_rank()
        print(f"[PROBE] rank{rank} EXC: {type(e).__name__}: {e}")
        print(f"[PROBE] rank{rank} state: allocated={torch.cuda.memory_allocated()/1e9:.2f} GiB "
              f"reserved={torch.cuda.memory_reserved()/1e9:.2f} GiB peak={torch.cuda.max_memory_allocated()/1e9:.2f} GiB")
        if rank == 0:
            try:
                snap = torch.cuda.memory_snapshot()
                blocks = [b for seg in snap for b in seg["blocks"]]
                states = sorted({b.get("state", "?") for b in blocks})
                print(f"[PROBE] snapshot: {len(snap)} segs, {len(blocks)} blocks, states={states}", flush=True)
                if blocks:
                    print(f"[PROBE] block keys: {sorted(blocks[0].keys())}", flush=True)
                def bsize(b):
                    return b.get("allocated_size", b.get("size", 0))
                live = [b for b in blocks if b.get("state") != "inactive"]
                live_total = sum(bsize(b) for b in live)
                from collections import Counter
                hist = Counter(round(bsize(b) / 2**20) for b in live)
                print(f"[PROBE] live blocks: {len(live)}, total={live_total/1e9:.2f} GiB, "
                      f"hist(MiB:count)={dict(sorted(hist.items()))[:20] if False else {k: v for k, v in sorted(hist.items()) if v >= 3}}", flush=True)
                top = sorted([b for b in live if bsize(b) > 50 * 1024**2], key=bsize, reverse=True)[:20]
                for b in top:
                    frames = " <- ".join(
                        f"{f['filename']}:{f['line']} {f['name']}" for f in b["frames"][:4]
                    )
                    print(f"[PROBE] top-alloc {bsize(b)/1e9:.2f} GiB [{b.get('state')}]: {frames}", flush=True)
                try:
                    full = torch.cuda.memory._snapshot()
                    events = full.get("device_events", [])
                    print(f"[PROBE] device_events: {len(events)}", flush=True)
                    for ev in events[-12:]:
                        fr = ev.get("frames", [])
                        loc = " <- ".join(f"{f['filename']}:{f['line']} {f['name']}" for f in fr[:3]) or "?"
                        sz = ev.get("size", 0) / 2**20
                        print(f"[PROBE] event {ev.get('action')} {sz:.0f} MiB @ {loc}", flush=True)
                except Exception as se:
                    print(f"[PROBE] events failed: {se}", flush=True)
            except Exception as se:
                print(f"[PROBE] snapshot failed: {se}", flush=True)
        torch.distributed.destroy_process_group()
        os._exit(1)
    if torch.distributed.get_rank() == 0:
        print(f"[PROBE] peak={torch.cuda.max_memory_allocated()/1e9:.2f} GiB")
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
