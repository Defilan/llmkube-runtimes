# spark-ds41f patches carried on b12x 78ee52c3

Three patch files from `christopherowen/spark-ds41f` at `a2a3abae7b395f98ef64dcc77beef1584d756cd6`, unmodified
(patch headers, authorship and commit messages kept; sha256 below). spark-ds41f is Apache-2.0 since `ad4605b9`
("License the repository under Apache 2.0", PR #355); its license text is `LICENSE.spark-ds41f-Apache-2.0` at the top
of this image directory, and NOTICE attributes it. spark-ds41f's README states that each of its patch series changes
its upstream's code under that project's license, here b12x's Apache-2.0.

`build/apply_carried_patches.sh` applies them in the order of `series`, each with `git apply --check` then
`git apply`, after `patches/b12x/0001` and before b12x is installed. `build/pins_gate.py` reads this table exactly as
it reads `patches/b12x/APPLIED.md`. Keep one row per line and do not put a `|` inside a cell. Each file is also listed
in `patches/MD5SUMS.txt` as vendored material.

These are the DSv4.1 Lane A b1 lab layer's three b12x patches, in its order. Strict-apply check: on 78ee52c3 plus
`patches/b12x/0001`, all three apply with `git apply` in this order, with no offset or fuzz. All three are correctness
fixes and always on:

- 0003: DSA radix top-k breaks exact score ties by lowest logical position; before, tied candidates were chosen by
  thread timing (78% of DS4.1 layer 2's prefill rows, about 5 of 512 slots differed between identical runs).
- 0004: dense GEMM fences generic-proxy stage reads before the TMA refill; the shared-expert down projection returned
  wrong columns in about 5% of decode calls beside the routed MoE (20/12000 wrong to 0/12000 in the standalone
  stress).
- 0005: the same fence in the BF16 prefill projection, the mHC TF32 and BF16 TMA projections and the contiguous
  attention forward (beside co-resident kernels mHC TF32 returned wrong outputs in 609/12000 calls and BF16 prefill
  in up to 45/12000).

| File | Upstream | Head SHA | Author | Scope | Form | Marker | sha256 |
|------|----------|----------|--------|-------|------|--------|--------|
| 0003-dsa-topk-position-ties.patch | christopherowen/spark-ds41f patches/b12x/0003-dsa-topk-position-ties.patch at a2a3abae7b | 47c70835a65c800410c9dd2fac62fa4c1f6ddcb2 | Christopher Owen | b12x/attention/dsa_indexer/tiled_topk.py tests/attention/test_topk_position_ties.py | verbatim upstream patch file (Lane A b1); always on (correctness fix) | def _position_key( | edc77fb127475c69dd79bc6aeb8e1d39fdf3576f67612a90ede5e2e2f1df6cf2 |
| 0004-gemm-fence-stage-reads-before-tma-refill.patch | christopherowen/spark-ds41f patches/b12x/0004-gemm-fence-stage-reads-before-tma-refill.patch at a2a3abae7b | 8d08c583f6e87851bd13fd87f7361023fc1fc368 | Christopher Owen | b12x/_lib/dense_gemm.py | verbatim upstream patch file (Lane A b1); always on (correctness fix) | its TMA refill is an async-proxy write | 8bc68527c039ccf744a6584b719bd41c45253ee022331b1f55013133d11f834b |
| 0005-fence-stage-reads-three-kernels.patch | christopherowen/spark-ds41f patches/b12x/0005-fence-stage-reads-three-kernels.patch at a2a3abae7b | 6e4da1fbc5b12340adf3390a6596f24a4842ad87 | Christopher Owen | b12x/attention/_shared/contiguous/forward.py b12x/gemm/bf16_gemv/_prefill.py b12x/norm/mhc/_kernels.py | verbatim upstream patch file (Lane A b1); always on (correctness fix) | async-proxy (TMA) | d2185009e3cc1ab71fa2f55d9f6107323a9b6ce0c72fa768864b6192dc27c61d |
