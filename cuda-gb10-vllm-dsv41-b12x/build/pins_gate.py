#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pins gate: the image is exactly what patches/ says it is, or the build fails.

Run at build (after every install) and again by the Tier-1 gate against the shipped image. Checks:

  UPSTREAM_COMMITS.txt  "name repo commit": /src/<name> is a clean checkout of that repo at that commit, the
                        matching build ARG (VLLM_FORK_COMMIT, B12X_COMMIT) agrees when set, and every git-tracked
                        <name>/**/*.py is byte-identical in dist-packages (so the installed package IS that commit).
  PINNED_DISTS.txt      "dist version [sha256 url]": exactly one distribution of that name is installed, at that
                        version. The sha256/url column is consumed by the Dockerfile, which verifies before install.
  vllm-install.txt      "mode sha256 source" written by the vLLM install step: mode is source (compiled here) or
                        wheel (VLLM_WHEEL_URL, hash-verified); a set VLLM_WHEEL_SHA256 must match it. The single
                        installed vllm's version must carry the fork commit (setuptools-scm's +g<hash>).
  MD5SUMS.txt           "md5-prefix file" for vendored files in patches/; an empty list is valid.

Every failure names the pin it broke. Exit 0 clean, 1 on any failure, 2 on usage.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata as md
import os
import re
import subprocess
import sys
import sysconfig
from pathlib import Path

COMMIT_ENV = {"vllm": "VLLM_FORK_COMMIT", "b12x": "B12X_COMMIT"}
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")


def _rows(path: Path) -> list[list[str]]:
    return [line.split() for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout.strip()


def installed(site: Path, dist: str) -> list[str]:
    return [d.version for d in md.distributions(path=[str(site)]) if _norm(d.metadata["Name"] or "") == _norm(dist)]


def check_commits(root: Path, src: Path, site: Path) -> tuple[list[str], dict[str, str]]:
    errors: list[str] = []
    pins: dict[str, str] = {}
    for row in _rows(root / "patches" / "UPSTREAM_COMMITS.txt"):
        if len(row) != 3 or not HEX40.fullmatch(row[2]):
            errors.append(f"{row[0]}: malformed UPSTREAM_COMMITS line {' '.join(row)!r} (want: name repo <40-hex commit>)")
            continue
        name, repo, commit = row
        pins[name] = commit
        env = COMMIT_ENV.get(name)
        if env and os.environ.get(env) and os.environ[env] != commit:
            errors.append(f"{name}: build ARG {env}={os.environ[env]} disagrees with UPSTREAM_COMMITS {commit}")
        checkout = src / name
        try:
            head = _git(checkout, "rev-parse", "HEAD")
            origin = _git(checkout, "remote", "get-url", "origin")
            dirty = _git(checkout, "status", "--porcelain", "--untracked-files=no")
            tracked = _git(checkout, "ls-files", "-z", "--", f"{name}/*.py").split("\0")
        except (RuntimeError, FileNotFoundError) as e:
            errors.append(f"{name}: {checkout} is not a usable git checkout ({e})")
            continue
        if head != commit:
            errors.append(f"{name}: {checkout} is at {head}, UPSTREAM_COMMITS pins {commit} ({repo})")
            continue
        if repo not in origin:
            errors.append(f"{name}: {checkout} origin is {origin}, UPSTREAM_COMMITS pins repo {repo}")
        if dirty:
            errors.append(f"{name}: {checkout} has modified tracked files: {' '.join(dirty.split())}")
        tracked = [t for t in tracked if t]
        if not tracked:
            errors.append(f"{name}: no tracked {name}/*.py in {checkout}; the package layout changed")
        for rel in tracked:
            dst = site / rel
            if not dst.is_file():
                errors.append(f"{name}: {rel} is tracked at {commit[:12]} but not installed in {site}")
            elif dst.read_bytes() != (checkout / rel).read_bytes():
                errors.append(f"{name}: installed {rel} differs from {commit[:12]}")
    return errors, pins


def check_dists(root: Path, site: Path) -> list[str]:
    errors: list[str] = []
    for row in _rows(root / "patches" / "PINNED_DISTS.txt"):
        if len(row) not in (2, 4) or (len(row) == 4 and not HEX64.fullmatch(row[2])):
            errors.append(f"{row[0]}: malformed PINNED_DISTS line {' '.join(row)!r} (want: dist version [sha256 url])")
            continue
        dist, want = row[0], row[1]
        got = installed(site, dist)
        if len(got) != 1:
            errors.append(f"{dist}: pinned {want}, {len(got)} installed {got}")
        elif got[0] != want:
            errors.append(f"{dist}: pinned {want}, installed {got[0]}")
    return errors


def check_vllm_install(root: Path, src: Path, site: Path, commit: str) -> list[str]:
    errors: list[str] = []
    rec = root / "vllm-install.txt"
    if not rec.is_file():
        return [f"vllm: {rec} missing; the vLLM install step did not record how vllm got here"]
    fields = rec.read_text().split()
    if len(fields) != 3 or fields[0] not in ("source", "wheel") or not HEX64.fullmatch(fields[1]):
        return [f"vllm: malformed {rec}: {' '.join(fields)!r} (want: source|wheel <sha256> <origin>)"]
    mode, sha, origin = fields
    want_url, want_sha = os.environ.get("VLLM_WHEEL_URL", ""), os.environ.get("VLLM_WHEEL_SHA256", "")
    if (want_url or want_sha) and mode != "wheel":
        errors.append(f"vllm: VLLM_WHEEL_URL/VLLM_WHEEL_SHA256 are set but {rec} records a {mode} build")
    if mode == "wheel" and want_sha and sha != want_sha:
        errors.append(f"vllm: installed wheel sha256 {sha}, VLLM_WHEEL_SHA256 pins {want_sha}")
    if mode == "wheel" and want_url and origin != want_url:
        errors.append(f"vllm: installed wheel came from {origin}, VLLM_WHEEL_URL pins {want_url}")
    versions = installed(site, "vllm")
    if len(versions) != 1:
        errors.append(f"vllm: {len(versions)} installed {versions}; the base image's vllm must be gone")
    elif f"g{commit[:7]}" not in versions[0]:
        try:
            tagged = bool(_git(src / "vllm", "describe", "--tags", "--exact-match", "HEAD"))
        except RuntimeError:
            tagged = False
        if not tagged:
            errors.append(f"vllm: installed version {versions[0]} does not carry fork commit g{commit[:7]}")
    return errors


def check_md5(root: Path) -> tuple[list[str], int]:
    rows = _rows(root / "patches" / "MD5SUMS.txt")
    errors: list[str] = []
    for row in rows:
        if len(row) != 2:
            errors.append(f"MD5SUMS: malformed line {' '.join(row)!r}")
            continue
        want, name = row
        p = root / "patches" / name
        if not p.is_file():
            errors.append(f"{name}: listed in MD5SUMS.txt but missing from patches/")
            continue
        got = hashlib.md5(p.read_bytes()).hexdigest()
        if not got.startswith(want):
            errors.append(f"{name}: md5 {got[:len(want)]}, MD5SUMS.txt pins {want}")
    return errors, len(rows)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="/opt/llmkube", type=Path, help="holds patches/ and vllm-install.txt")
    ap.add_argument("--src", default="/src", type=Path, help="holds one checkout per UPSTREAM_COMMITS name")
    ap.add_argument("--site", default=sysconfig.get_paths()["purelib"], type=Path, help="dist-packages")
    a = ap.parse_args(argv[1:])
    errors, pins = check_commits(a.root, a.src, a.site)
    errors += check_dists(a.root, a.site)
    if "vllm" in pins:
        errors += check_vllm_install(a.root, a.src, a.site, pins["vllm"])
    md5_errors, vendored = check_md5(a.root)
    errors += md5_errors
    for e in errors:
        print("PIN: " + e, file=sys.stderr)
    if errors:
        return 1
    print(f"pins gate OK: {', '.join(f'{n}@{c[:12]}' for n, c in pins.items())}; "
          f"{len(_rows(a.root / 'patches' / 'PINNED_DISTS.txt'))} dists at pinned versions; {vendored} vendored files")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
