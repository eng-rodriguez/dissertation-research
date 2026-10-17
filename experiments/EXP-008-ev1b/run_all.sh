#!/usr/bin/env bash
# EXP-008 (EV-1b) driver: theta for every scenario, then the fixed-mode coverage datasets in chunks,
# then the summary under the unchanged rule with M1 as the only decision interval. A 1,000-dataset
# scenario that is INDETERMINATE (and no scenario FAIL) is extended once to 2,000, as pre-registered
# in the config; nothing else changes. Resumable: rerun the same command after an interruption;
# finished datasets are skipped. Run from the repository root.
set -euo pipefail
C=experiments/EXP-008-ev1b/config.json
W=${WORKERS:-$(python3 -c "import os; print(os.cpu_count())")}
OUT=artifacts/protocol/ev1b
SCEN="pos_hetero pos_boundary_ne33 pos_mid_hetero pos_highvar_hetero pos_sesoi null_homog null_hetero null_highvar"
n_for() { python3 -c "import json,sys; print(json.load(open('$C'))['criteria']['datasets_per_scenario'][sys.argv[1]])" "$1"; }
for s in $SCEN; do
  [ -f "$OUT/theta/$s.json" ] || python3 src/design/ev1_coverage.py theta --config $C --scenario "$s" --workers "$W"
done
for s in $SCEN; do
  n=$(n_for "$s")
  for lo in $(seq 0 100 $((n - 100))); do
    python3 src/design/ev1_coverage.py run --config $C --scenario "$s" --range "$lo:$((lo + 100))" --workers "$W"
  done
done
# The extension decision is taken once, from the first complete pass, and kept in extension.txt so a
# resumed run extends the same scenarios.
if [ ! -f "$OUT/extension.txt" ]; then
  python3 src/design/ev1_coverage.py summarise --config $C > /dev/null
  cp "$OUT/ev1_summary.json" "$OUT/ev1_summary_pass1.json"
  python3 -c "
import json; s = json.load(open('$OUT/ev1_summary.json'))
ext = [] if s['result'] == 'FAIL' else [k for k, r in s['scenarios'].items()
                                         if r['verdict'] == 'INDETERMINATE' and r['n_datasets'] == 1000]
print(' '.join(ext))" > "$OUT/extension.txt"
fi
for s in $(cat "$OUT/extension.txt"); do
  for lo in $(seq 1000 100 1900); do
    python3 src/design/ev1_coverage.py run --config $C --scenario "$s" --range "$lo:$((lo + 100))" --workers "$W"
  done
done
python3 src/design/ev1_coverage.py summarise --config $C
