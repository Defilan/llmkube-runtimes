# Upstream patches carried on b12x 0d6600e6

Base: `local-inference-lab/b12x` commit `0d6600e64955114cad44cdee94c764a8001221e2` (`patches/UPSTREAM_COMMITS.txt`).
The Dockerfile applies every `*.patch` here in filename order with `git apply --check` then `git apply`,
records each patch's new files with `git add -N`, and then installs b12x. `build/pins_gate.py` checks each file against the table
below (sha256), that the checkout differs from the pin by exactly these patches, that every patched file under
`b12x/` is installed byte-identical, and that each marker appears in an installed patched file. Keep one row per
line and do not put a `|` inside a cell. Drop a patch (and its row) when the pin moves past its merge.

## Applied

| File | Upstream | Head SHA | Author | Scope | Form | Marker | sha256 |
|------|----------|----------|--------|-------|------|--------|--------|
| 0001-roce-switchless-ring-routes.patch | local-inference-lab/b12x#457 | 4eccc7022b19ac01d19f10f1fc080196fe793dc6 | Defilan | b12x/comm/roce (RoCEnante setup and RDMA proxy) | PR diff restricted to b12x/ (tests, docs and evidence left out), checked against 0d6600e6 | three DGX Sparks cabled in a ring | c28859bef3e62ab6c6f32aaa795538692564e2788ae8591483053bac236b54ee |

Why: RoCEnante pairs local HCA h with every peer's HCA h, which only holds on a switched fabric. On the
three-Spark ring every queue pair timed out at RTR and vLLM fell back to NCCL for the tensor-parallel
all-reduce. With this patch RoCEnante connects on the ring; the evidence (GPU suite, standalone collectives
receipt, serving A/B) is in the PR.
