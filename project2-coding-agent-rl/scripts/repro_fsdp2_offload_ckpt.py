"""Repro: FSDP2 + CPUOffloadPolicy + PEFT LoRA checkpoint save crash.

Mirrors verl's wrap sequence (verl/utils/fsdp_utils.py apply_fsdp2) and tests
which state_dict() path survives after a train step leaves the model offloaded.

Run on one free GPU: CUDA_VISIBLE_DEVICES=1 python scripts/repro_fsdp2_offload_ckpt.py
"""

import os
import sys

os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29555"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"

import torch
import torch.distributed as dist

gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if "0" in gpu.split(",") or not gpu:
    raise SystemExit("REFUSED: use a free GPU (1-7), never 0")
dist.init_process_group("nccl")
torch.cuda.set_device(0)

from peft import LoraConfig, get_peft_model
from torch.distributed._composable.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoModelForCausalLM

MODEL = "/media/imc/data/yzy/agent/project2/hf-cache/hub/models--Qwen--Qwen2.5-Coder-3B-Instruct/snapshots/488639f1ff808d1d3d0ba301aef8c11461451ec5"

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cpu")
model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules="all-linear"))
model.config.use_cache = False

mesh = init_device_mesh("cuda", (1,))
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, cast_forward_inputs=True)
offload = CPUOffloadPolicy(pin_memory=True)

# mirror verl's _select_fsdp2_wrap_targets (match by class name)
wrap_names = set(model._no_split_modules)
modules = [
    m for name, m in model.named_modules()
    if m.__class__.__name__ in wrap_names
    or name.rsplit(".", 1)[-1] in {"embed_tokens", "lm_head"}
]
layers = modules
print(f"wrapping {len(layers)} layers + root")
for layer in layers:
    fully_shard(layer, mesh=mesh, mp_policy=mp, offload_policy=offload, reshard_after_forward=True)
fully_shard(model, mesh=mesh, mp_policy=mp, offload_policy=offload, reshard_after_forward=False)

# one train step
inputs = torch.randint(0, 32000, (1, 64), device="cuda")
out = model(input_ids=inputs, labels=inputs)
out.loss.backward()
torch.cuda.synchronize()
print("after step, first param device:", next(model.parameters()).device.type)

paths = {
    "A plain state_dict on offloaded model": lambda: model.state_dict(),
    "B load_to_gpu (model.to(0)) then state_dict": lambda: (model.to(0), model.state_dict()),
}
for label, fn in paths.items():
    try:
        sd = fn()
        devs = {v.device.type for v in sd.values() if torch.is_tensor(v)}
        print(f"[{label}] OK keys={len(sd)} devices={devs}")
    except Exception as e:
        print(f"[{label}] FAIL {type(e).__name__}: {str(e)[:160]}")

from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

try:
    sd = get_model_state_dict(model, options=StateDictOptions(cpu_offload=True))
    devs = {v.device.type for v in sd.values() if torch.is_tensor(v)}
    print(f"[C get_model_state_dict(cpu_offload=True)] OK keys={len(sd)} devices={devs}")
except Exception as e:
    print(f"[C] FAIL {type(e).__name__}: {str(e)[:160]}")

dist.destroy_process_group()
print("DONE")
