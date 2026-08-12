#!/bin/bash
# End-to-end driver: check-vllm -> generate -> validate -> verify-answers -> tables -> export.
#
# Stages 01, 02, 03, 04, 06 and 07. Stage 01 runs first because every generate pass
# drafts medium/hard gold on the local vLLM servers -- failing on connectivity
# in seconds beats failing mid-pass. Stage 04 (verify-answers) runs before stage 05
# (human review) on purpose: every question a model can independently re-derive
# gets that automated check first, so a human only ever reviews what a model
# could not already settle. It is safe to run unattended -- a disagreement is
# reported, never acted on. Stage 05 (human review) is skipped here because a
# human has to fill in the worksheet between its two commands -- a driver that
# ran both would apply an empty worksheet and report every record as
# left_undecided.
#
# Every command's parameters come from scripts/tasks/*.txt; this script owns the
# one piece those files do not hold, which program each stage runs under, so the
# driver and the documented pipeline cannot drift apart.
set -u
cd "$(dirname "$0")/.."

STAGES="01_check_vllm 02_generate 03_validate 04_verify_answers 06_analysis_offline 07_export_analyzer"
LOGDIR=output/logs
mkdir -p "$LOGDIR"
STAMP=$(date +%Y%m%d_%H%M%S)
MASTER="$LOGDIR/run_all_${STAMP}.log"

say () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER"; }

program_for () {
  case "$1" in
    06_analysis_offline) echo "analysis/analysis_tables.py" ;;
    *) echo "main.py" ;;
  esac
}

say "=== run_all START (pid $$) ==="

rc_total=0
for stage in $STAGES; do
  program=$(program_for "$stage")
  say "Stage $stage ($program)"
  while IFS= read -r params; do
    [ -z "$params" ] && continue
    say "  run: python3 $program $params"
    bash -c "python3 $program $params" >> "$MASTER" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then
      say "  FAILED (rc=$rc): python3 $program $params"
      rc_total=$rc
    fi
  done < "scripts/tasks/${stage}.txt"
  say "Stage $stage done"
done

say "Human review is stage 05 and is not run here:"
say "  python3 main.py --command review-export   # then edit the worksheet"
say "  python3 main.py --command review-apply --reviewer <isim>"

say "=== run_all COMPLETE (rc=$rc_total) -> $MASTER ==="
exit "$rc_total"
