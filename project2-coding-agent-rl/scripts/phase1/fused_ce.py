#!/usr/bin/env python3
"""Fused lm_head + cross-entropy for long-sequence SFT (Liger-style).

Why: the plain path materializes logits (seq, vocab) in forward AND a full
(seq, vocab) dlogits buffer in backward (autograd accumulates the chunked-CE
grads before the lm_head backward consumes them) — 6.96 GiB each at 24K,
OOMing 24 GB cards.

This function fuses the two: forward computes per-token log_probs from the
hidden states directly (chunked matmul against the lm_head weight, no full
logits tensor); backward recomputes each chunk's softmax, reuses that buffer as
dlogits, and immediately contracts it into grad_hidden. grad_weight is only
allocated when lm_head is actually trainable (it is frozen for our LoRA SFT).

Numerically identical to log_softmax(lm_head(h))[label]: bf16 GEMM (fp32
accumulate) + fp32 logsumexp, exactly like the model's lm_head + flash-attn CE.

ZeRO-3: the caller must wrap the call in
``deepspeed.zero.GatheredParameters(weight, fwd_module=<consumer module>)`` —
deepspeed gathers the partitioned weight for the forward. This function clones
that gathered bf16 weight because the context restores the partition before
backward; the clone is used only to propagate gradients into hidden states.

Memory at seq S (Qwen2.5-Coder-7B, vocab 151936, hidden 3584, bf16,
default chunk=512, frozen lm_head): one 1.01 GiB gathered-weight clone plus one
~0.29 GiB fp32 vocabulary chunk and O(S×hidden) hidden/grad_hidden buffers.
"""
import os

import torch


def _chunk_size():
    chunk = int(os.environ.get("OPENRLHF_FUSED_CE_CHUNK_SIZE", "512"))
    if chunk <= 0:
        raise ValueError(f"OPENRLHF_FUSED_CE_CHUNK_SIZE must be positive, got {chunk}")
    return chunk


class FusedLinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, hidden, labels, temperature):
        # weight: (vocab, hidden), hidden: (batch, seq, hidden), labels: (batch, seq)
        batch, seqlen = hidden.shape[0], hidden.shape[1]
        hdim = hidden.shape[-1]
        scaled_hidden = hidden if temperature == 1.0 else hidden / temperature
        flat_h = scaled_hidden.reshape(-1, hdim)
        flat_l = labels.reshape(-1)
        n = flat_h.shape[0]
        lse = torch.empty(n, device=hidden.device, dtype=torch.float32)
        logits_label = torch.empty(n, device=hidden.device, dtype=torch.float32)
        chunk_size = _chunk_size()
        for s in range(0, n, chunk_size):
            e = min(s + chunk_size, n)
            lg = (flat_h[s:e] @ weight.T).float()  # (chunk, vocab) fp32
            lse[s:e] = torch.logsumexp(lg, dim=-1)
            logits_label[s:e] = lg[torch.arange(e - s, device=lg.device), flat_l[s:e]]
        log_probs = (logits_label - lse).reshape(batch, seqlen)
        # clone: the caller gathers the weight via GatheredParameters, and the
        # gather exits (restoring the partition, param.data -> empty(0)) before
        # our backward runs — saving the same object would yield an empty weight.
        # Grad for the original gathered tensor is unaffected.
        ctx.save_for_backward(weight.detach().clone(), hidden, labels)
        ctx.temperature = temperature
        ctx.weight_requires_grad = ctx.needs_input_grad[0]
        ctx.chunk_size = chunk_size
        return log_probs

    @staticmethod
    def backward(ctx, grad_log_probs):
        weight, hidden, labels = ctx.saved_tensors
        temperature = ctx.temperature
        hdim = hidden.shape[-1]
        vocab = weight.shape[0]
        scaled_hidden = hidden if temperature == 1.0 else hidden / temperature
        flat_h = scaled_hidden.reshape(-1, hdim)
        flat_l = labels.reshape(-1)
        flat_g = grad_log_probs.reshape(-1).to(torch.float32)
        # Qwen LoRA excludes lm_head, so this is normally None. Avoiding the
        # unnecessary fp32 (vocab, hidden) buffer saves 2.03 GiB for Qwen2.5-7B.
        grad_weight = (
            torch.zeros(vocab, hdim, device=hidden.device, dtype=torch.float32)
            if ctx.weight_requires_grad
            else None
        )
        grad_hidden = torch.zeros_like(hidden, dtype=torch.float32)
        n = flat_h.shape[0]
        for s in range(0, n, ctx.chunk_size):
            e = min(s + ctx.chunk_size, n)
            lg = (flat_h[s:e] @ weight.T).float()
            # Reuse the logits buffer in-place as dlogits. This keeps the
            # transient vocabulary-sized memory to one chunk, rather than
            # separate logits + softmax + dlogits buffers.
            lg.sub_(torch.logsumexp(lg, dim=-1)[:, None]).exp_()
            # d log_softmax(label) / d logits = one_hot(label) - softmax.
            lg.mul_(-flat_g[s:e, None])
            idx = torch.arange(e - s, device=lg.device)
            lg[idx, flat_l[s:e]] += flat_g[s:e]
            h_chunk = flat_h[s:e]
            if grad_weight is not None:
                grad_weight += lg.T @ h_chunk.float()  # fp32 accumulation
            grad_hidden.reshape(-1, hdim)[s:e] = lg @ weight.float()
        if temperature != 1.0:
            grad_hidden = grad_hidden / temperature
        if grad_weight is not None:
            grad_weight = grad_weight.to(weight.dtype)
        return grad_weight, grad_hidden.to(hidden.dtype), None, None


def fused_log_probs(weight, hidden, labels, temperature=1.0):
    """log_probs = log_softmax(weight @ hidden^T)[labels], fused (no (seq,vocab))."""
    return FusedLinearCE.apply(weight, hidden, labels, float(temperature))
