# SPDX-License-Identifier: Apache-2.0
"""The assembled image: the fork's vLLM and b12x import, report their pins, and the fork's TP3 hook pads 64/8 to 72/9.

No CUDA kernel is launched; this runs in the docker build and on the driverless gate runner.
"""
from __future__ import annotations
import glob
import re
import importlib
import importlib.metadata as md
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
VLLM_SRC = Path(os.environ.get("DSV41_VLLM_SRC", "/src/vllm"))
FORK_TEST = "tests/v1/attention/test_b12x_sparse_mla_api.py::test_deepseek_v41_tp3_padding_uses_generic_parallel_hook"


def _pin(name: str) -> str:
    for line in (HERE.parent / "patches" / "UPSTREAM_COMMITS.txt").read_text().splitlines():
        if line.split()[:1] == [name]:
            return line.split()[2]
    raise KeyError(name)


def _sm12x_visible() -> bool:
    import torch
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] == 12


def test_the_brief_import_line():
    import vllm, b12x  # noqa: F401,E401
    from vllm.models.deepseek_v4_1 import attention  # noqa: F401


def test_versions_are_the_pinned_ones():
    import torch
    import flashinfer
    import vllm
    assert torch.__version__ == "2.13.0+cu130"
    assert flashinfer.__version__ == "0.6.18.post1"
    assert md.version("b12x") == "1.3.0"
    assert f"g{_pin('vllm')[:7]}" in vllm.__version__, vllm.__version__


def test_vllm_and_b12x_import_from_dist_packages_not_the_checkouts():
    import vllm, b12x  # noqa: E401
    for mod in (vllm, b12x):
        assert "/dist-packages/" in mod.__file__, mod.__file__


def test_b12x_registers_its_vllm_plugins():
    eps = {(e.name, e.value) for e in md.entry_points(group="vllm.general_plugins")}
    assert ("b12x_fp6", "b12x.integration.vllm.plugin:register_b12x_fp6") in eps
    assert ("b12x_loader", "b12x.integration.vllm.loader:register_b12x_loader") in eps


def test_v41_padding_hook_is_wired_for_the_model_and_its_drafter():
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP, DeepseekV41ForCausalLMConfig
    assert MODELS_CONFIG_MAP["DeepseekV41ForCausalLM"] is DeepseekV41ForCausalLMConfig
    assert MODELS_CONFIG_MAP["DSparkV41DraftModel"] is DeepseekV41ForCausalLMConfig
    assert "update_model_config_for_parallelism" in vars(DeepseekV41ForCausalLMConfig)


def gb10_loadable(elf_names):
    """Splits the SM12x cubins `cuobjdump --list-elf` reports into (loadable on GB10, not loadable).

    GB10 is sm_121. A cubin for a lower minor of the same major runs on it (CUDA binary compatibility), so plain
    sm_120 and the 12.0f family cubins the fork builds on CUDA 13.0 (CMakeLists.txt: CUDA_SUPPORTED_ARCHS lists
    12.0, not 12.1; cmake/utils.cmake maps 12.1a onto the 12.0f family) are fine. Architecture-specific "a" cubins
    run only on their exact chip, so sm_120a would not load on GB10; sm_121a would.
    """
    ok, bad = [], []
    for name in elf_names:
        m = re.search(r"sm_(12\d)([af]?)", name)
        if not m:
            continue
        arch, suffix = m.group(1), m.group(2)
        (bad if suffix == "a" and arch != "121" else ok).append(name)
    return ok, bad


def test_gb10_loadable_classifies_cubins():
    ok, bad = gb10_loadable(["x.1.sm_120.cubin", "x.2.sm_121a.cubin", "x.3.sm_120a.cubin", "x.4.sm_80.cubin",
                             "x.5.sm_120f.cubin"])
    assert ok == ["x.1.sm_120.cubin", "x.2.sm_121a.cubin", "x.5.sm_120f.cubin"]
    assert bad == ["x.3.sm_120a.cubin"]


def test_compiled_extensions_load_on_gb10():
    cuobjdump = shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"
    if not os.path.exists(cuobjdump):
        pytest.skip("cuobjdump not in this image")
    import vllm
    sos = sorted(glob.glob(os.path.join(os.path.dirname(vllm.__file__), "_C*.so")))
    assert sos, "vllm/_C*.so missing: the fork was installed without its compiled extension"
    for so in sos:
        out = subprocess.run([cuobjdump, "--list-elf", so], capture_output=True, text=True).stdout
        names = [line.split(":", 1)[1].strip() for line in out.splitlines() if "ELF file" in line and ":" in line]
        ok, bad = gb10_loadable(names)
        assert not bad, f"{os.path.basename(so)} has SM12x cubins GB10 cannot load: {bad[:5]}"
        if any(re.search(r"sm_12\d", n) for n in names):
            assert ok, f"{os.path.basename(so)}: no GB10-loadable SM12x cubin"
    assert any(
        gb10_loadable([l.split(":", 1)[1].strip() for l in subprocess.run(
            [cuobjdump, "--list-elf", so], capture_output=True, text=True).stdout.splitlines() if "ELF file" in l])[0]
        for so in sos
    ), "no _C*.so carries a GB10-loadable SM12x cubin"


def test_fork_tp3_padding_test_passes():
    """The fork's own test, from the pinned checkout, against the installed package.

    Run from / with --import-mode=importlib so /src/vllm is never put on sys.path (its vllm/ would shadow the
    installed one), and --noconftest because the fork's tests/conftest.py pulls in its whole test-dependency set.
    Without an SM12x device the platform shim stands in for the GPU; see plugins/sm12x_platform_shim.py.
    """
    assert (VLLM_SRC / FORK_TEST.split("::")[0]).is_file(), f"{FORK_TEST} is not in {VLLM_SRC}"
    shim = [] if _sm12x_visible() else ["-p", "sm12x_platform_shim"]
    env = dict(os.environ, PYTHONPATH=str(HERE / "plugins"), PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--noconftest", "--import-mode=importlib",
         f"--rootdir={VLLM_SRC}", *shim, str(VLLM_SRC / FORK_TEST)],
        cwd="/", env=env, capture_output=True, text=True)
    assert r.returncode == 0 and "1 passed" in r.stdout, (shim, r.stdout[-4000:], r.stderr[-2000:])


def test_prewarm_recorded_the_b12x_compile_env():
    text = Path("/opt/llmkube/b12x-compile-env.txt").read_text()
    assert text.startswith("B12X_COMPILE_CACHE_DIR=/opt/llmkube/b12x-cache\n"), text
    assert "CUTE_DSL_ARCH=sm_121a\n" in text, "the compile-cache key was recorded without the image's CUTE_DSL_ARCH"
    importlib.import_module("b12x._lib.compiler")


def test_image_env_defaults_point_at_real_things():
    assert os.environ.get("CUTE_DSL_ARCH") == "sm_121a"
    assert os.access(os.environ["TRITON_PTXAS_PATH"], os.X_OK), os.environ["TRITON_PTXAS_PATH"]
    cache = os.environ["B12X_COMPILE_CACHE_DIR"]
    assert os.path.isdir(cache) and os.stat(cache).st_mode & 0o1777 == 0o1777, cache


def test_b12x_loader_check_upstream_launcher_runs():
    """Verbatim from upstream's launcher (scripts/serve-ds4-flash-dspark-tp4-rdma.sh:239): --load-format b12x needs it."""
    loader_check = ('from importlib.metadata import entry_points; raise SystemExit(not any(ep.name == "b12x_loader" '
                    'for ep in entry_points(group="vllm.general_plugins")))')
    assert subprocess.run([sys.executable, "-c", loader_check]).returncode == 0


COMPILED_IN = {
    "LICENSE.flash-attention-BSD-3-Clause": "Redistribution and use in source and binary forms",
    "LICENSE.tml-fa4-BSD-3-Clause": "Redistribution and use in source and binary forms",
    "LICENSE.FlashMLA-MIT": "Permission is hereby granted",
    "LICENSE.DeepGEMM-MIT": "Permission is hereby granted",
    "LICENSE.triton-MIT": "Permission is hereby granted",
    "LICENSE.FlashKDA-MIT": "Permission is hereby granted",
    "LICENSE.DeepSelect-MIT": "Permission is hereby granted",
    "LICENSE.MSA-MIT": "Permission is hereby granted",
    "LICENSE.qutlass-Apache-2.0": "Apache License",
}


def test_attribution_for_everything_compiled_into_vllm_C_travels():
    lic = Path("/opt/llmkube/licenses")
    assert {p.name for p in lic.iterdir()} == set(COMPILED_IN), sorted(p.name for p in lic.iterdir())
    for name, marker in COMPILED_IN.items():
        assert marker in (lic / name).read_text(), name
    for top in ("NOTICE", "LICENSE.vllm-Apache-2.0", "LICENSE.b12x-Apache-2.0", "LICENSE.flashinfer-Apache-2.0",
                "LICENSE.cutlass-BSD-3-Clause"):
        assert (Path("/opt/llmkube") / top).is_file(), top
