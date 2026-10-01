# SPDX-License-Identifier: Apache-2.0
"""pins_gate.py against a synthetic image: real git checkouts, a fake dist-packages, one wrong pin at a time.

Runs on the host (git and python only) and again inside the image, where it proves the gate itself still
behaves before the gate's verdict on the real tree is trusted.
"""
from __future__ import annotations
import hashlib
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
IMAGE_DIR = HERE.parent
GATE = IMAGE_DIR / "build" / "pins_gate.py"
PIN_ENV = ("VLLM_FORK_COMMIT", "B12X_COMMIT", "VLLM_WHEEL_URL", "VLLM_WHEEL_SHA256")


def _git(cwd: Path, *args: str) -> str:
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@t", GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull)
    return subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True).stdout.strip()


def _checkout(src: Path, name: str, repo: str, files: dict[str, str]) -> str:
    d = src / name
    for rel, body in files.items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    _git(d.parent, "init", "-q", name)
    _git(d, "remote", "add", "origin", f"https://github.com/{repo}.git")
    _git(d, "add", "-A")
    _git(d, "commit", "-q", "-m", "pinned")
    return _git(d, "rev-parse", "HEAD")


def _dist(site: Path, name: str, version: str) -> None:
    info = site / f"{name.replace('-', '_')}-{version}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")


VLLM_FILES = {"vllm/__init__.py": "V = 1\n", "vllm/models/deepseek_v4_1/attention.py": "import b12x\n",
              "requirements/cuda.txt": "torch==2.13.0\n", "tests/test_x.py": "def test_x(): pass\n"}
B12X_FILES = {"b12x/__init__.py": "", "b12x/attention/__init__.py": "", "b12x/attention/mla.py": "K = 2\n"}


@pytest.fixture()
def image(tmp_path):
    """A consistent synthetic image. Tests break exactly one thing and expect the gate to name it."""
    src, site, root = tmp_path / "src", tmp_path / "site", tmp_path / "opt"
    vllm_sha = _checkout(src, "vllm", "local-inference-lab/vllm", VLLM_FILES)
    b12x_sha = _checkout(src, "b12x", "local-inference-lab/b12x", B12X_FILES)
    for name, files in (("vllm", VLLM_FILES), ("b12x", B12X_FILES)):
        for rel, body in files.items():
            if rel.startswith(name + "/"):
                p = site / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(body)
    _dist(site, "vllm", f"0.1.dev1+g{vllm_sha[:9]}.cu130")
    _dist(site, "torch", "2.13.0+cu130")
    _dist(site, "flashinfer-python", "0.6.18.post1")
    _dist(site, "b12x", "1.3.0")
    (root / "patches").mkdir(parents=True)
    (root / "patches" / "UPSTREAM_COMMITS.txt").write_text(
        f"# name repo commit\nvllm local-inference-lab/vllm {vllm_sha}\nb12x local-inference-lab/b12x {b12x_sha}\n")
    (root / "patches" / "PINNED_DISTS.txt").write_text(
        "# dist version [sha256 url]\ntorch 2.13.0+cu130\nb12x 1.3.0\n"
        f"flashinfer-python 0.6.18.post1 {'a' * 64} https://example.invalid/flashinfer_python-0.6.18.post1-py3-none-any.whl\n")
    (root / "patches" / "MD5SUMS.txt").write_text("# md5-prefix file (vendored files only; none today)\n")
    (root / "vllm-install.txt").write_text(f"source {'b' * 64} /src/vllm\n")
    return {"src": src, "site": site, "root": root, "vllm": vllm_sha, "b12x": b12x_sha}


def run_gate(image, **env):
    base = {k: v for k, v in os.environ.items() if k not in PIN_ENV}
    base.update(env)
    return subprocess.run([sys.executable, str(GATE), "--root", str(image["root"]), "--src", str(image["src"]),
                           "--site", str(image["site"])], env=base, capture_output=True, text=True)


def test_consistent_image_passes(image):
    r = run_gate(image)
    assert r.returncode == 0, r.stderr
    assert "pins gate OK" in r.stdout


def test_wrong_commit_pin_fails_and_names_it(image):
    wrong = "0" * 40
    f = image["root"] / "patches" / "UPSTREAM_COMMITS.txt"
    f.write_text(f.read_text().replace(image["b12x"], wrong))
    r = run_gate(image)
    assert r.returncode == 1
    assert "b12x" in r.stderr and wrong in r.stderr and image["b12x"] in r.stderr
    assert "vllm:" not in r.stderr, "only the broken pin should be reported"


def test_wrong_dist_version_fails_and_names_it(image):
    shutil.rmtree(next(image["site"].glob("flashinfer_python-*.dist-info")))
    _dist(image["site"], "flashinfer-python", "0.6.18")
    r = run_gate(image)
    assert r.returncode == 1
    assert "flashinfer-python" in r.stderr and "0.6.18.post1" in r.stderr


def test_modified_installed_file_fails(image):
    (image["site"] / "b12x" / "attention" / "mla.py").write_text("K = 3\n")
    r = run_gate(image)
    assert r.returncode == 1 and "b12x/attention/mla.py" in r.stderr


def test_missing_installed_file_fails(image):
    (image["site"] / "vllm" / "models" / "deepseek_v4_1" / "attention.py").unlink()
    r = run_gate(image)
    assert r.returncode == 1 and "vllm/models/deepseek_v4_1/attention.py" in r.stderr


def test_base_vllm_left_installed_fails(image):
    _dist(image["site"], "vllm", "0.20.0")
    r = run_gate(image)
    assert r.returncode == 1 and "vllm" in r.stderr and "2 installed" in r.stderr


def test_vllm_version_without_the_fork_commit_fails(image):
    shutil.rmtree(next(image["site"].glob("vllm-*.dist-info")))
    _dist(image["site"], "vllm", "0.20.0+cu130")
    r = run_gate(image)
    assert r.returncode == 1 and "0.20.0+cu130" in r.stderr


def test_dirty_checkout_fails(image):
    (image["src"] / "vllm" / "requirements" / "cuda.txt").write_text("")
    r = run_gate(image)
    assert r.returncode == 1 and "requirements/cuda.txt" in r.stderr


def test_build_arg_that_disagrees_with_the_pin_file_fails(image):
    r = run_gate(image, B12X_COMMIT="1" * 40)
    assert r.returncode == 1 and "B12X_COMMIT" in r.stderr
    assert run_gate(image, B12X_COMMIT=image["b12x"], VLLM_FORK_COMMIT=image["vllm"]).returncode == 0


def test_wheel_mode_must_carry_the_pinned_wheel_hash(image):
    good, bad = "c" * 64, "d" * 64
    (image["root"] / "vllm-install.txt").write_text(f"wheel {good} https://example.invalid/vllm.whl\n")
    ok = run_gate(image, VLLM_WHEEL_URL="https://example.invalid/vllm.whl", VLLM_WHEEL_SHA256=good)
    assert ok.returncode == 0, ok.stderr
    r = run_gate(image, VLLM_WHEEL_URL="https://example.invalid/vllm.whl", VLLM_WHEEL_SHA256=bad)
    assert r.returncode == 1 and bad in r.stderr


def test_wheel_arg_set_but_source_build_recorded_fails(image):
    r = run_gate(image, VLLM_WHEEL_URL="https://example.invalid/vllm.whl", VLLM_WHEEL_SHA256="c" * 64)
    assert r.returncode == 1 and "source" in r.stderr


def test_missing_install_record_fails(image):
    (image["root"] / "vllm-install.txt").unlink()
    r = run_gate(image)
    assert r.returncode == 1 and "vllm-install.txt" in r.stderr


def test_vendored_file_md5_checked_and_mismatch_named(image):
    vend = image["root"] / "patches" / "entry.sh"
    vend.write_text("#!/bin/sh\n")
    md5 = hashlib.md5(vend.read_bytes()).hexdigest()[:8]
    (image["root"] / "patches" / "MD5SUMS.txt").write_text(f"{md5} entry.sh\n")
    assert run_gate(image).returncode == 0
    vend.write_text("#!/bin/sh\necho changed\n")
    r = run_gate(image)
    assert r.returncode == 1 and "entry.sh" in r.stderr and md5 in r.stderr


def test_malformed_pin_line_fails(image):
    f = image["root"] / "patches" / "UPSTREAM_COMMITS.txt"
    f.write_text(f.read_text() + "tilelang 0.1.12\n")
    r = run_gate(image)
    assert r.returncode == 1 and "tilelang" in r.stderr


def test_empty_commit_pin_file_fails(image):
    (image["root"] / "patches" / "UPSTREAM_COMMITS.txt").write_text("# name repo commit\n")
    r = run_gate(image)
    assert r.returncode == 1
    assert "vllm: no entry in UPSTREAM_COMMITS.txt" in r.stderr and "b12x: no entry in UPSTREAM_COMMITS.txt" in r.stderr


def test_commit_pin_file_missing_one_entry_fails(image):
    f = image["root"] / "patches" / "UPSTREAM_COMMITS.txt"
    f.write_text("\n".join(l for l in f.read_text().splitlines() if not l.startswith("b12x ")) + "\n")
    r = run_gate(image)
    assert r.returncode == 1 and "b12x: no entry in UPSTREAM_COMMITS.txt" in r.stderr and "vllm:" not in r.stderr


def test_empty_dist_pin_file_fails(image):
    (image["root"] / "patches" / "PINNED_DISTS.txt").write_text("# dist version [sha256 url]\n")
    r = run_gate(image)
    assert r.returncode == 1
    for dist in ("torch", "flashinfer-python", "b12x"):
        assert f"{dist}: no entry in PINNED_DISTS.txt" in r.stderr, r.stderr


def test_lookalike_origin_fails(image):
    _git(image["src"] / "b12x", "remote", "set-url", "origin", "https://github.com/local-inference-lab/b12x-evil.git")
    r = run_gate(image)
    assert r.returncode == 1 and "b12x-evil" in r.stderr


def test_origin_match_is_exact_across_url_forms():
    spec = importlib.util.spec_from_file_location("pins_gate", GATE)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    repo = "local-inference-lab/vllm"
    for ok in ("https://github.com/local-inference-lab/vllm.git", "https://github.com/local-inference-lab/vllm",
               "git@github.com:local-inference-lab/vllm.git", "https://github.com/Local-Inference-Lab/vllm/"):
        assert gate.origin_matches(ok, repo), ok
    for bad in ("https://github.com/local-inference-lab/vllm-evil.git", "https://github.com/x/local-inference-lab/vllm",
                "https://gitlab.com/local-inference-lab/vllm.git", "https://github.com/local-inference-lab/vllm.git.evil"):
        assert not gate.origin_matches(bad, repo), bad


# ---- the real pin files in this directory ----

def _pins() -> dict[str, tuple[str, str]]:
    out = {}
    for line in (IMAGE_DIR / "patches" / "UPSTREAM_COMMITS.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            name, repo, commit = line.split()
            out[name] = (repo, commit)
    return out


def test_dockerfile_arg_defaults_match_upstream_commits():
    if not (IMAGE_DIR / "Dockerfile").is_file():
        pytest.skip("source-tree check; the image carries patches/ but not the Dockerfile")
    dockerfile = (IMAGE_DIR / "Dockerfile").read_text()
    args = dict(re.findall(r"^ARG (VLLM_FORK_COMMIT|B12X_COMMIT)=([0-9a-f]{40})$", dockerfile, re.M))
    pins = _pins()
    assert args == {"VLLM_FORK_COMMIT": pins["vllm"][1], "B12X_COMMIT": pins["b12x"][1]}
    assert pins["vllm"][0] == "local-inference-lab/vllm" and pins["b12x"][0] == "local-inference-lab/b12x"


def test_pinned_wheels_are_hashed_https_urls():
    rows = [l.split() for l in (IMAGE_DIR / "patches" / "PINNED_DISTS.txt").read_text().splitlines()
            if l.strip() and not l.startswith("#")]
    wheels = [r for r in rows if len(r) == 4]
    assert {r[0] for r in wheels} == {"flashinfer-python", "flashinfer-cubin", "flashinfer-jit-cache"}
    for dist, version, sha, url in wheels:
        assert re.fullmatch(r"[0-9a-f]{64}", sha), dist
        assert url.startswith("https://") and url.endswith(".whl"), dist
        assert version.split("+")[0] in url, (dist, version, url)
    assert all(len(r) in (2, 4) for r in rows), rows
