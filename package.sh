#!/bin/sh
# Packaging helper: clean untracked artifacts and create a distributable zip from tracked files.
# WARNING: `git clean -fdX` is destructive to untracked files. Use with caution.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "Cleaning untracked ignored files... (git clean -fdX)"
git clean -fdX

OUT="cpp-modernizer-package-$(date +%Y%m%d%H%M%S).zip"

echo "Creating package $OUT from current HEAD (tracked files only)"
# Use git archive to avoid including any ignored/untracked files
if git rev-parse --verify HEAD >/dev/null 2>&1; then
  git archive --format=zip -o "$OUT" HEAD
  echo "Package created: $OUT"
else
  echo "No git HEAD available; falling back to zipping working tree"
  zip -r "$OUT" . -x "*.pyc" "__pycache__/*" ".pytest_cache/*" ".modernization_state.db*" "*_modernized.cpp"
  echo "Package created (fallback): $OUT"
fi
