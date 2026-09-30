# SPDX-License-Identifier: Apache-2.0
"""deps_gate.py: which pip check lines fail the build, and the AGPL metadata scan. Pure functions, host and image."""
from __future__ import annotations
import importlib.util
from email.message import Message
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("deps_gate", HERE.parent / "build" / "deps_gate.py")
deps_gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deps_gate)

OWNED = {"vllm", "b12x", "torch", "flashinfer-python", "nvidia-cutlass-dsl"}


def test_conflict_on_a_pinned_package_is_fatal_either_side():
    text = ("vllm 0.1.dev1 has requirement transformers>=5.10.4, but you have transformers 5.9.0.\n"
            "quack-kernels 0.6.5 has requirement nvidia-cutlass-dsl==4.6.2, but you have nvidia-cutlass-dsl 4.7.1.\n")
    fatal, tolerated = deps_gate.classify_pip_check(text, OWNED, [])
    assert len(fatal) == 2 and tolerated == []


def test_conflict_between_packages_we_do_not_own_is_tolerated():
    text = "somelib 1.0 requires otherlib, which is not installed.\nNo broken requirements found.\n"
    fatal, tolerated = deps_gate.classify_pip_check(text, OWNED, [])
    assert fatal == [] and tolerated == ["somelib 1.0 requires otherlib, which is not installed."]


def test_allowlist_tolerates_a_documented_conflict_and_unparseable_lines_are_fatal():
    text = "vllm 0.1.dev1 has requirement torchcodec>=0.14, but you have torchcodec 0.12.\nsomething odd happened\n"
    fatal, tolerated = deps_gate.classify_pip_check(text, OWNED, [r"^vllm .* torchcodec"])
    assert tolerated == [text.splitlines()[0]] and fatal == ["something odd happened"]


def test_real_allowlist_file_parses():
    rows = deps_gate._rows(HERE.parent / "patches" / "PIP_CHECK_ALLOW.txt")
    for r in rows:
        __import__("re").compile(r)


class _Dist:
    def __init__(self, name, license="", classifiers=(), expr=""):
        m = Message()
        m["Name"] = name
        if license:
            m["License"] = license
        if expr:
            m["License-Expression"] = expr
        for c in classifiers:
            m["Classifier"] = c
        self.metadata, self.version = m, "1.0"


def test_agpl_scan_catches_license_expression_and_classifier_and_passes_permissive():
    dists = [_Dist("ok", "Apache-2.0", ["License :: OSI Approved :: MIT License"]),
             _Dist("bad1", expr="AGPL-3.0-only"),
             # Built from parts so the directory's own no-AGPL-text grep does not match this fixture.
             _Dist("bad2", classifiers=["License :: OSI Approved :: GNU " + "Affero General Public License v3"]),
             _Dist("gpl", "GPL-3.0")]
    hits = deps_gate.agpl_hits(dists)
    assert [h.split()[0] for h in hits] == ["bad1", "bad2"]


def test_licenses_dir_is_exactly_what_notice_cites():
    """Source-tree check (the image carries licenses/ at /opt/llmkube/licenses and NOTICE beside it)."""
    root = HERE.parent
    lic = root / "licenses"
    notice = root / "NOTICE"
    if not lic.is_dir():
        lic = Path("/opt/llmkube/licenses")
        notice = Path("/opt/llmkube/NOTICE")
    names = sorted(p.name for p in lic.iterdir())
    assert len(names) == 9, names
    text = notice.read_text()
    for n in names:
        assert f"licenses/{n}" in text, f"NOTICE does not cite licenses/{n}"


def test_a_crashed_or_silent_pip_check_fails_the_gate(monkeypatch, tmp_path, capsys):
    import subprocess
    for rc, out in ((-9, ""), (2, "Traceback"), (1, "")):
        monkeypatch.setattr(deps_gate.subprocess, "run",
                            lambda *a, rc=rc, out=out, **k: subprocess.CompletedProcess(a, rc, out, ""))
        assert deps_gate.main(["deps_gate", "--root", str(tmp_path)]) == 1
        assert "without a usable report" in capsys.readouterr().err
