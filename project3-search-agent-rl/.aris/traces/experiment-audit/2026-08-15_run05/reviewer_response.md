# Fresh Reviewer Response

Overall verdict: **WARN** (`same-family / provisional`).

Reason code: `real_small_sample_signal_but_not_significant_reused_devset_and_runtime_code_uncommitted`.

The result numbers are real. Raw episode recomputation gives Base 3/32, Step5old 5/32 and train64nqh8
5/32. Base→train64 has two 0→1 flips, zero 1→0 flips and 13/32 raw-action changes. Adapter files exist
and their hashes match the results. Real dataset GT and the 21,015,324-vector Wiki Retriever were used;
there is no self-normalization or training leakage in the manifest.

The evidence supports an engineering-valid evaluation path, real LoRA influence on vLLM outputs, and a
qualified preliminary positive signal on these 32 questions. It does not support statistical significance:
the exact paired McNemar p-value is 0.5; Base Wilson 95% CI is [3.24%, 24.22%] and train64 is
[6.86%, 31.75%]. Search episodes decrease from six to five, so search-policy improvement is unsupported.
The repeatedly inspected heldout-32 should now be treated as dev32, not an independent final confirmation set.

There is also a major reproducibility gap. All five successful vLLM Runs used the current uncommitted
`run_p3_eval_vllm.py`. Commit 22df3fe still contains failing ragged-batch, `LoRARequest` argument and
`LLM.shutdown()` code; bdedc18 did not commit the runtime fixes. Results do not record runtime script SHA or
the dirty diff. Therefore the statement that bdedc18 contains all fixes is false until the script is reviewed
and committed.

Backend choice materially affects output, and vLLM is a reasonable formal engineering backend because its
version and main configuration match training. The evidence only attributes divergence to generation-backend
numeric/implementation differences, not FlashAttention alone. The standalone harness is not the complete veRL
`val_only` worker path.

Five Runs exit 0, expose only physical GPU1 and report no remaining compute process. A process-group cleanup
warning remains and should be fixed. Next: commit the runtime fixes with code hashes; relabel heldout-32 as
dev32; pre-register a new unseen 128–256-question paired vLLM confirmation; and separately establish an
official loose-action-semantics Search-R1 baseline.
