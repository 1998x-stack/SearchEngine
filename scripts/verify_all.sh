#!/usr/bin/env bash
# 校验目录中的主题 .py 文件与主题目录一致，且每个文件均可运行。
# 用法: bash scripts/verify_all.sh
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

catalog_file="docs/superpowers/catalog.md"
if [[ ! -f "$catalog_file" ]]; then
  echo "ERROR: topic catalog not found: $catalog_file" >&2
  exit 2
fi

expected_list=$(mktemp "${TMPDIR:-/tmp}/searchengine-expected.XXXXXX")
actual_list=$(mktemp "${TMPDIR:-/tmp}/searchengine-actual.XXXXXX")
trap 'rm -f "$expected_list" "$actual_list"' EXIT

if ! sed -nE 's/^\|[[:space:]]*[0-9]+[[:space:]]*\|[[:space:]]*`(\.\/[^`]+\.py)`[[:space:]]*\|.*/\1/p' \
  "$catalog_file" | LC_ALL=C sort > "$expected_list"; then
  echo "ERROR: failed to read topic paths from catalog: $catalog_file" >&2
  exit 2
fi

expected_count=$(wc -l < "$expected_list" | tr -d '[:space:]')
if [[ "$expected_count" -eq 0 ]]; then
  echo "ERROR: topic catalog contains no numbered Python paths: $catalog_file" >&2
  exit 2
fi

if ! find . -type f -name "*.py" \
  -not -path './.git/*' \
  -not -path './docs/*' \
  -not -path './scripts/*' \
  -not -name 'main.py' \
  -print | LC_ALL=C sort > "$actual_list"; then
  echo "ERROR: failed to enumerate topic Python files; refusing to report success." >&2
  exit 2
fi

actual_count=$(wc -l < "$actual_list" | tr -d '[:space:]')
if ! diff -u "$expected_list" "$actual_list"; then
  printf 'ERROR: topic script set mismatch (catalog: %s, found: %s); refusing to report success.\n' \
    "$expected_count" "$actual_count" >&2
  exit 2
fi

pass=0
fail=0
while IFS= read -r f; do
  if python3 "$f" >/dev/null 2>&1; then
    pass=$((pass + 1))
  else
    echo "FAIL: $f"
    fail=$((fail + 1))
  fi
done < "$actual_list"

printf '通过: %s, 失败: %s\n' "$pass" "$fail"
if [[ "$fail" -eq 0 ]]; then
  printf 'ALL OK (%s/%s)\n' "$pass" "$expected_count"
else
  echo "SOME FAILED"
  exit 1
fi
