# Project 3 mandatory safety rules

Before running, stopping, resuming, downloading, or deleting anything, read
`docs/EXPERIMENT_SAFETY.md` completely. These rules apply to every human or AI agent working in this
directory.

- Never use physical GPU 0. It is reserved for the Linux desktop. A process-visible `cuda:0` is safe
  only when `CUDA_VISIBLE_DEVICES` explicitly remaps it from an approved nonzero physical GPU.
- Physical GPU 5 is excluded by default because it has been unstable. Do not override this merely to
  gain capacity. Any other GPU must still be checked for ownership and idleness immediately before use.
- Never start a GPU job directly. Run the read-only preflight, then use `scripts/run_managed.sh` inside
  a named tmux session. Use a new run ID; never overwrite or reuse an existing run directory.
- Never use broad termination commands such as `pkill python`, `killall`, `ray stop --force`, or a
  process-name match. Stop only the exact tmux session or the exact managed run with
  `scripts/stop_managed.sh`, then verify PIDs, ports, Ray actors, and GPU memory.
- Do not delete, format, remount, move, or recursively clean `/media/imc/data`, the repository root,
  run directories, datasets, models, indexes, checkpoints, or unknown processes. Preserve failed-run
  evidence. Ask the user before any material deletion or overwrite.
- Keep datasets/models/checkpoints/logs under
  `/media/imc/data/project3-search-agent-rl/`. Do not commit large artifacts or secrets.
- Pin `vendor/verl-agent` to commit `20bd331bdbc9026a5668e11362178e10ab7400c8` and represent local
  upstream changes as reviewable patches. Do not silently upgrade CUDA, PyTorch, vLLM, Ray, veRL, or
  the model.
- Scale one variable at a time. Multi-GPU is allowed only after an explicit physical-GPU whitelist,
  idle checks, memory estimate, veRL/Ray resource review, and a new smoke/resume/cleanup gate.
- A successful process exit or changing weights proves an engineering run, not quality improvement.
  Final held-out and two-seed results support engineering reproduction and stable behavior changes,
  but not a stable EM improvement. Do not broaden that claim without new controlled evidence.

If a requested action conflicts with these rules, stop and explain the conflict instead of guessing.
