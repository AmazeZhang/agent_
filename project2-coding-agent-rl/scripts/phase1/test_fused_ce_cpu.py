#!/usr/bin/env python3
"""Numerical test for the chunked fused lm_head + CE autograd path."""

import argparse
import os

import torch
import torch.nn.functional as F

from fused_ce import fused_log_probs


def reference(weight, hidden, labels, temperature):
    logits = (hidden / temperature) @ weight.T
    return F.log_softmax(logits.float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def run_case(weight_requires_grad, temperature, device, dtype):
    torch.manual_seed(7)
    weight = torch.randn(17, 9, device=device, dtype=dtype, requires_grad=weight_requires_grad)
    hidden = torch.randn(2, 7, 9, device=device, dtype=dtype, requires_grad=True)
    labels = torch.randint(0, 17, (2, 7), device=device)
    upstream = torch.randn(2, 7, device=device)
    upstream[0, 1::2] = 0  # exercise the SFT per-token mask gradient shape

    ref = reference(weight, hidden, labels, temperature)
    (ref * upstream).sum().backward()
    ref_hidden_grad = hidden.grad.detach().clone()
    ref_weight_grad = weight.grad.detach().clone() if weight_requires_grad else None

    weight2 = weight.detach().clone().requires_grad_(weight_requires_grad)
    hidden2 = hidden.detach().clone().requires_grad_(True)
    got = fused_log_probs(weight2, hidden2, labels, temperature)
    (got * upstream).sum().backward()

    atol = 2e-2 if dtype == torch.bfloat16 else 2e-5
    rtol = 2e-2 if dtype == torch.bfloat16 else 5e-5
    torch.testing.assert_close(got, ref.detach(), rtol=rtol, atol=atol)
    torch.testing.assert_close(hidden2.grad, ref_hidden_grad, rtol=rtol, atol=atol)
    if weight_requires_grad:
        torch.testing.assert_close(weight2.grad, ref_weight_grad, rtol=rtol, atol=atol)
    else:
        assert weight2.grad is None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    os.environ["OPENRLHF_FUSED_CE_CHUNK_SIZE"] = "3"
    for weight_requires_grad in (False, True):
        for temperature in (1.0, 0.7):
            run_case(weight_requires_grad, temperature, args.device, dtype)
    print(
        f"FUSED_CE_NUMERICAL_PASS device={args.device} dtype={args.dtype}: "
        "outputs + hidden/weight grads + temperature + token mask"
    )


if __name__ == "__main__":
    main()
