# Patches carried on the vLLM fork 1286ae9c

Base: `local-inference-lab/vllm` commit `1286ae9c9a3fd9376b82da39f7bc87f6a99037d9`, branch
`integration/karmic-kraken-beta`, tagged `vllm-jovian-cu134-beta-1286ae9c9a3fd9376b82da39f7bc87f6a99037d9`
(`patches/UPSTREAM_COMMITS.txt`).
The Dockerfile applies every `*.patch` here in filename order with `git apply --check` then `git apply` (new files
recorded with `git add -N`) twice: in the compile stage, before the wheel is built, and in the image's `/src/vllm`
checkout, which the pins gate compares the installed package against. `build/pins_gate.py` checks each file against
the table below (sha256), that the checkout differs from the pin by exactly these patches, that every patched file
under `vllm/` is installed byte-identical, and that each marker appears in an installed patched file. Keep one row
per line and do not put a `|` inside a cell. Drop a patch (and its row) when the pin moves past its merge.
A patch authored in this repository and not yet proposed upstream records `none` as its head SHA.

## Applied

| File | Upstream | Head SHA | Author | Scope | Form | Marker | sha256 |
|------|----------|----------|--------|-------|------|--------|--------|
| 0001-ds41-mxfp4-csf-tp3.patch | carried here (not yet proposed upstream) | none | Defilan | vllm/model_executor/model_loader/mxfp4_csf_loader.py (read_mxfp4_csf_layer TP guard) | one-line allowlist change plus a comment, against 1286ae9c | DS4.1 TP3 (three DGX Sparks) | 18be69ad65b2d96e57f843eac7187c0e03ec26de965e222b05ce7528ffbe58b1 |
| 0002-ds41-rope-table-rows-max-model-len.patch | carried here (not yet proposed upstream) | none | Defilan | vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py (DeepseekV4ScalingRotaryEmbedding table rows) | opt-in env LLMKUBE_DSV41_ROPE_ROWS=max_model_len, default unchanged, against 1286ae9c | LLMKUBE_DSV41_ROPE_ROWS | 858e586aeab1ab6769938152345e44c68d617f0dca54125a3dbe03232915302a |
| 0003-ds41-drop-unused-candidate-block-buffer.patch | carried here (not yet proposed upstream) | none | Defilan | vllm/models/deepseek_v4_1/nvidia/model.py (DeepseekV4Model candidate_block_buffer) | allocation replaced by None plus a comment, against 1286ae9c | per rank for nothing | c9a38e57de93f6dc5ad9a4df8438865f62479b137e084fd6fc77cf5f5d2c99a1 |
| 0004-worker-memory-stats-snapshots.patch | carried here (not yet proposed upstream) | none | Defilan | vllm/v1/worker/gpu_worker.py (env-gated hook), new vllm/v1/worker/llmkube_memory_stats.py | inert unless LLMKUBE_MEMORY_STATS_DIR is set, against 1286ae9c | LLMKUBE_MEMORY_STATS_DIR | 61becb0c43c0b9e85c640a838c6f754bdafd0f07b19914ae05788873ac3dc40c |

Why: the fork's MXFP4-CSF reader admits DeepSeek-V4.1 only at TP 1, 2, 4 or 8, by an explicit allowlist in
`read_mxfp4_csf_layer`; every other constraint on the path already holds at TP3. The routed-expert intermediate size
is 2304, so a TP3 rank owns 768 channels: `tp_extent` needs 32-channel alignment (2304 % 96 == 0), the compressed
scale-plane slicer needs whole 16-row slabs (rank boundaries 0, 768, 1536 are multiples of 16) and accepts any column
range, and b12x's compact W4A8 CSF geometry needs N64/K64 (w13 1536 x 5120, w2 5120 x 768 per rank). The Kimi family
in the same function already allows TP 12. The patch adds 3 for DS4.1 and nothing else. It touches Python only, so
the compiled extensions are the same as an unpatched build. `tests/test_imports_in_image.py` slices a synthetic CSF
expert set at TP3 through the fork's own loader and checks the bytes; serving it on a GB10 ring is not yet qualified
(see the image README). Upstream's branch head still has the same allowlist (checked at `7f172870`, 2026-10-04).

Why 0002: `DeepseekV4ScalingRotaryEmbedding` builds its fp32 cos/sin table over `original_max_position *
scaling_factor` positions, 1,048,576 for DeepSeek-V4.1 (compressed layers 65,536 x 16, sliding-window layers
1,048,576 x 1), and the two memoized tables cost 512 MiB per rank. The scheduler never places an input at or past
`max_model_len` (`vllm/v1/core/sched/scheduler.py:1115-1121`), so with `LLMKUBE_DSV41_ROPE_ROWS=max_model_len` the
table stops at `max_model_len + 1024` (the margin covers draft positions). Every kept row is byte-identical to the
full table's: row i depends only on i and `inv_freq`, and `inv_freq` still uses the original max position
(`yarn_find_correction_range`). The only length-dependent consumer on the GB10 path, the WO warmup's
`positions.remainder_(table.shape[0])` (`vllm/models/deepseek_v4_1/attention.py:1657`), stays in range. Saves 384 MiB
per rank at 262K. Default (unset) behaviour is the unpatched table.

Why 0003: `DeepseekV4Model` allocates `candidate_block_buffer` (`max_num_batched_tokens x candidate_topk_blocks`
int32; 4096 x 2048 = 32 MiB per rank with DeepSeek-V4.1's `text_config`) and passes it to every layer, but the only
attention class on this path, `DeepseekV41B12xAttention`, inherits `DeepseekV4Attention.__init__`, which takes the
argument and never stores or reads it (`vllm/models/deepseek_v4_1/attention.py:420`); its two-level candidate filter
keeps its own `_candidates` workspace. Setting it to None removes an allocation nothing reads.

Why 0004: the DSpark headroom work (llmkube-internal `research/2026-10-17-dsv41-dspark-headroom-engine-levers.md`)
needs the PyTorch caching allocator's allocated versus reserved bytes at each phase, which no external /proc census
can see. With `LLMKUBE_MEMORY_STATS_DIR` set, each worker appends JSON lines (phase, rank, allocator counters,
cudaMemGetInfo, RSS) after load_model, initialize_from_config and compile_or_warm_up_model, and every
`LLMKUBE_MEMORY_STATS_INTERVAL` seconds (default 60) from execute_model. It never allocates on the device and swallows
its own errors. Unset, the module is not imported.

Not carried, on purpose: gating `_mtp_hidden_buffer` (160 MiB with DSpark). DSpark does not consume its contents, but
the GPU model runner slices the buffer unconditionally when the target defines `get_mtp_target_hidden_states`
(`vllm/v1/worker/gpu/model_runner.py:966-967`), which the GB10 class inherits from
`vllm/models/deepseek_v41/nvidia/vl_model.py:326`; a None buffer would raise on the first DSpark step. A safe change
needs a model-runner change as well.
