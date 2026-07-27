#!/bin/bash
# Quick human-readable snapshot of what is currently in output/.
# Reads finished files only -- safe to run on the host, no container needed.
#
# Usage: scripts/check_status.sh [questions.json path]
cd "$(dirname "$0")/.."

Q=${1:-output/pilot/questions.json}
R=$(dirname "$Q")/validation_report.json

echo "=================================================="
echo " DATASET DURUMU:  $Q"
echo "=================================================="
if [ ! -f "$Q" ]; then
  echo "   (yok -- once: python3 main.py --command generate)"
  exit 0
fi

python3 - "$Q" <<'PY'
import json
import sys
from collections import Counter

records = json.load(open(sys.argv[1], encoding="utf-8"))
print(f"   toplam soru      : {len(records)}")
for field in ("difficulty", "routing_path", "split", "review_status"):
    counts = Counter(r.get(field, "?") for r in records)
    rendered = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"   {field:<17}: {rendered}")
in_review = sum(1 for r in records if r.get("review_status") == "in_review")
if in_review:
    print(f"   -> {in_review} kayit insan incelemesi bekliyor (stage 04)")
PY

echo "--------------------------------------------------"
echo " VALIDATION: $R"
if [ -f "$R" ]; then
  python3 - "$R" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
verdict = "PASSED" if report.get("passed") else "FAILED"
print(f"   {verdict}  errors={report.get('error_count')}  warnings={report.get('warning_count')}")
snapshot = report.get("config_snapshot") or {}
if snapshot.get("code_version"):
    print(f"   code_version: {snapshot['code_version']}")
for error in (report.get("errors") or [])[:5]:
    print(f"   - {error}")
PY
else
  echo "   (yok -- once: python3 main.py --command validate)"
fi
echo "=================================================="
