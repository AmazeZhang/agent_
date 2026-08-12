#!/usr/bin/env python3
"""Filter SFT multiturn trajectories by tokenized length (chat-template applied).

Why: OpenRLHF SFTDataset truncates long samples with HF `max_length=` which keeps
the HEAD and cuts the TAIL — for SWE trajectories the tail holds the final fix
patch, so truncation is the wrong loss. Pre-filtering guarantees no sample ever
hits the cap, so truncation never fires (verified: 24K covers 84% of 287).

Usage: filter_sft_len.py <in.jsonl> <out.jsonl> <max_len> [model_dir]
A 256-token safety margin below max_len guards against template-application
tokenizer differences between this script and SFTDataset.
"""
import json
import sys

from transformers import AutoTokenizer

IN, OUT, MAX_LEN = sys.argv[1], sys.argv[2], int(sys.argv[3])
MODEL = sys.argv[4] if len(sys.argv) > 4 else "/media/imc/data/yzy/agent/project2/phase1/models/Qwen2.5-Coder-7B-Instruct"
MARGIN = 256
CAP = MAX_LEN - MARGIN

tok = AutoTokenizer.from_pretrained(MODEL)

kept, dropped = 0, 0
with open(IN) as f, open(OUT, "w") as g:
    for line in f:
        d = json.loads(line)
        msgs = d["input"]
        # replicate SFTDataset.__getitem__ length exactly:
        # prompt = template(msgs[:-1], add_generation_prompt=True); response = template(msgs)[len(prompt):]
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        response = tok.apply_chat_template(msgs, tokenize=False)[len(prompt) :]
        text = (prompt + response).rstrip("\n")
        if not text.endswith(tok.eos_token):
            text += " " + tok.eos_token
        n = len(tok(text, add_special_tokens=False)["input_ids"])
        if n <= CAP:
            g.write(line)
            kept += 1
        else:
            dropped += 1
print(f"kept={kept} dropped={dropped} (cap={CAP}, {100*kept/(kept+dropped):.1f}%)")
