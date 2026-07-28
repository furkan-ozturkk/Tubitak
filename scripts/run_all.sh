#!/bin/bash
# End-to-end driver: check-ollama -> generate -> validate -> tables -> export.
#
# Stages 01, 02, 03, 05 and 07. Stage 01 runs first because every generate pass
# drafts medium/hard gold on the remote Ollama server -- failing on connectivity
# in seconds beats failing mid-pass. Stage 04 (human review) is skipped because
# a human has to fill in the worksheet between its two commands -- a driver that
# ran both would apply an empty worksheet and report every record as
# left_undecided. Stage 06 (verify-answers) stays manual for the same reason its
# task file gives: a disagreement needs a person before anyone edits anything.
#
# Every command comes from scripts/tasks/*.txt so the driver and the documented
# pipeline cannot drift apart. Commented lines in those files are alternatives,
# not steps, and are skipped here.
set -u
cd "$(dirname "$0")/.."

STAGES="01_check_ollama 02_generate 03_validate 05_analysis_offline 07_export_analyzer"
LOGDIR=output/logs
mkdir -p "$LOGDIR"
STAMP=$(date +%Y%m%d_%H%M%S)
MASTER="$LOGDIR/run_all_${STAMP}.log"

say () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER"; }

say "=== run_all START (pid $$) ==="

rc_total=0
for stage in $STAGES; do
  say "Stage $stage"
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    case "$line" in \#*) continue;; esac
    say "  run: $line"
    bash -c "$line" >> "$MASTER" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then
      say "  FAILED (rc=$rc): $line"
      rc_total=$rc
    fi
  done < "scripts/tasks/${stage}.txt"
  say "Stage $stage done"
done

say "Human review is stage 04 and is not run here:"
say "  python3 main.py --command review-export   # then edit the worksheet"
say "  python3 main.py --command review-apply"

say "=== run_all COMPLETE (rc=$rc_total) -> $MASTER ==="
exit "$rc_total"
