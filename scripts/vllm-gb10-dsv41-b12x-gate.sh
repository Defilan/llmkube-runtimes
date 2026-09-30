#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Tier-1 gate for llmkube-vllm-cuda-gb10-dsv41-b12x: re-asserts the Dockerfile's guarantees on the
# SHIPPED image. No CUDA kernel is launched; everything here runs on a driverless runner.
set -euo pipefail
IMAGE="${1:?usage: vllm-gb10-dsv41-b12x-gate.sh <image-ref>}"
run() { docker run --rm --entrypoint "$1" "${IMAGE}" "${@:2}"; }

echo "== receipts and attribution =="
run test -f /opt/llmkube/receipt.txt
run test -f /opt/llmkube/python-tree.sha256
run test -f /opt/llmkube/pip-freeze.txt
run test -f /opt/llmkube/NOTICE
for f in LICENSE.vllm-Apache-2.0 LICENSE.b12x-Apache-2.0 LICENSE.flashinfer-Apache-2.0; do
  run bash -c "grep -q 'Apache License' /opt/llmkube/$f && grep -q 'Version 2.0, January 2004' /opt/llmkube/$f"
done
run bash -c "grep -q 'Redistribution and use in source and binary forms' /opt/llmkube/LICENSE.cutlass-BSD-3-Clause"
run bash -c '! grep -rli "GNU AFFERO" /opt/llmkube/patches /opt/llmkube/build /opt/llmkube/tests'
run bash -c '! grep -rli --exclude-dir=.git "GNU AFFERO" /src/vllm /src/b12x'
echo "PASS: receipt, stamp, freeze, NOTICE, all four license texts present; no AGPL text in the image's own material or either source tree"

echo "== pins gate (checkouts at the pinned commits, installed trees identical, dist versions, install record) =="
run python3 /opt/llmkube/build/pins_gate.py --site /usr/local/lib/python3.12/dist-packages
run cat /opt/llmkube/vllm-install.txt
echo "PASS: pins"

echo "== the fork's vLLM and b12x import, V4.1 attention first =="
run python3 -c "import vllm, b12x; from vllm.models.deepseek_v4_1 import attention; print('vllm', vllm.__version__, 'b12x ok')"
run bash -c "! pip show flashinfer-python 2>/dev/null | grep -q '^Version: 0.6.18$'"
echo "PASS: imports; the base's FlashInfer 0.6.18 is gone"

run bash -c 'DSV41_VERIFY_TREE=1 /usr/local/bin/dsv41-entrypoint.sh true'
echo "PASS: entrypoint verifies the vllm+b12x tree digest"

echo "== in-image test suites (includes the fork's TP3 padding test from /src/vllm) =="
run bash -c 'DSV41_IN_IMAGE=1 python3 -m pytest -q -p no:cacheprovider /opt/llmkube/tests'
echo "PASS: Tier-1 gate complete for ${IMAGE}"
