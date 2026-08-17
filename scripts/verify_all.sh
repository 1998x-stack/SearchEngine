#!/bin/bash
# 校验仓库内所有主题 .py 文件均可运行.
# 用法: bash scripts/verify_all.sh
set -u
cd "$(dirname "$0")/.." || exit 1

pass=0
fail=0
while IFS= read -r f; do
  if python3 "$f" >/dev/null 2>&1; then
    pass=$((pass + 1))
  else
    echo "FAIL: $f"
    fail=$((fail + 1))
  fi
done < <(find . -name "*.py" -not -path './.git/*' -not -name 'main.py' | sort)

echo "通过: $pass, 失败: $fail"
if [ "$fail" -eq 0 ]; then
  echo "ALL OK"
else
  echo "SOME FAILED"
  exit 1
fi