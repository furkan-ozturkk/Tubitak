#!/bin/bash
# Fast structural smoke test: does every module import, does the CLI parse, and do
# config/args.py's three validations actually reject what they claim to reject?
#
# Touches no network and no database, so it runs on the host as well as in the
# container. This is what to run after moving a module.
set -u
cd "$(dirname "$0")/.."

PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python

fails=0
check () {
  local label="$1"; shift
  if "$@" >/dev/null 2>&1; then
    echo "  ok   : $label"
  else
    echo "  FAIL : $label"
    fails=$((fails + 1))
  fi
}
check_rejects () {
  local label="$1"; shift
  if "$@" >/dev/null 2>&1; then
    echo "  FAIL : $label (was accepted, should have been rejected)"
    fails=$((fails + 1))
  else
    echo "  ok   : $label"
  fi
}

echo "=== imports ($PY) ==="
check "main.py"                  $PY -c "import main"
check "generate.py"              $PY -c "import generate"
check "validate.py"              $PY -c "import validate"
check "src.generators"           $PY -c "import src.generators"
check "src.params"               $PY -c "import src.params.results_params, src.params.scale_params"
check "analysis.analysis_tables" $PY -c "import analysis.analysis_tables"

echo "=== CLI surface ==="
check "--help"                   $PY main.py --help
for command in check-ollama generate validate verify-answers review-export review-apply; do
  check "--command $command parses" $PY -c "
from config.args import args_parser
args_parser(['--command', '$command'])
"
done

echo "=== config/args.py validations ==="
check_rejects "identical draft/review models" $PY -c "
from config.args import args_parser
args_parser(['--gold_draft_model', 'x:1b', '--groundedness_model', 'x:1b'])
"
check_rejects "window_size 0" $PY -c "
from config.args import args_parser
args_parser(['--window_size', '0'])
"
check_rejects "test_fraction 1.0" $PY -c "
from config.args import args_parser
args_parser(['--test_fraction', '1.0'])
"
check "questions/review_out default to --dataset" $PY -c "
from config.args import args_parser
args = args_parser(['--dataset', '/tmp/q.json'])
assert args.questions == ['/tmp/q.json'], args.questions
assert str(args.review_out) == '/tmp/q.json', args.review_out
"

echo "=== unit tests ==="
check "unittest discover" $PY -m unittest discover -s tests -t . -b

echo "=================================================="
if [ "$fails" -eq 0 ]; then
  echo " SMOKE OK"
else
  echo " SMOKE FAILED: $fails check(s)"
fi
exit "$fails"
