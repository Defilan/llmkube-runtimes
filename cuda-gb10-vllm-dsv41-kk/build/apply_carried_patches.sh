#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# apply_carried_patches.sh <patch-dir> <checkout>: apply every <patch-dir>/*.patch to <checkout> in filename order, then
# every series subdirectory of <patch-dir> (one holding a `series` file, e.g. spark-ds41f/), in directory-name order,
# each patch in the order its `series` file lists it. A series is a third-party stack kept byte-for-byte as upstream
# published it, so its own numbering may repeat ours and its patches may build on each other: it is applied strictly
# in its listed order, every listed file must exist, and every *.patch in that directory must be listed exactly once.
# Each patch must apply cleanly (git apply --check first; git apply never fuzzes). Its new files are then recorded with
# `git add -N` (intent to add), so a later `git clean` keeps them and build/pins_gate.py sees them as part of the
# checkout. Not `git apply --intent-to-add`: git 2.43 (the base's git) drops every other index entry when it is used
# without --index. A missing or empty patch directory applies nothing. build/pins_gate.py computes the same order.
set -eu
pdir="$1"
co="$2"

apply_one() {
  git -C "$co" apply --check "$1"
  git -C "$co" apply "$1"
  git -C "$co" apply --summary "$1" | awk '/ create mode /{print $NF}' | while read -r f; do
    git -C "$co" add -N -- "$f"
  done
  echo "applied $2 to $co"
}

for p in "$pdir"/*.patch; do
  [ -e "$p" ] || continue
  apply_one "$p" "$(basename "$p")"
done

for series in "$pdir"/*/series; do
  [ -e "$series" ] || continue
  sdir="$(dirname "$series")"
  sname="$(basename "$sdir")"
  listed="$(sed -e 's/#.*//' -e 's/[[:space:]]*$//' -e '/^$/d' "$series")"
  dups="$(printf '%s\n' "$listed" | sort | uniq -d)"
  [ -z "$dups" ] || { echo "$series lists more than once: $dups" >&2; exit 1; }
  for p in "$sdir"/*.patch; do
    [ -e "$p" ] || continue
    printf '%s\n' "$listed" | grep -qxF -- "$(basename "$p")" || { echo "$p is not listed in $series" >&2; exit 1; }
  done
  [ -n "$listed" ] || continue
  printf '%s\n' "$listed" | while read -r f; do
    [ -f "$sdir/$f" ] || { echo "$series lists $f, which is not in $sdir" >&2; exit 1; }
    apply_one "$sdir/$f" "$sname/$f"
  done
done
