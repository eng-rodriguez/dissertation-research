#!/usr/bin/env bash
# EXP-002 (EV-1) driver: theta for every scenario, then the fixed-mode coverage datasets in chunks,
# then the descriptive tuned check, then the summary. Resumable: rerun the same command after an
# interruption; finished datasets are skipped. Run from the repository root.
set -euo pipefail
C=experiments/EXP-002-ev1/config.json
W=${WORKERS:-$(python3 -c "import os; print(os.cpu_count())")}
OUT=artifacts/protocol/ev1
for s in null_homog null_hetero pos_hetero null_highvar pos_sesoi; do
  [ -f "$OUT/theta/$s.json" ] || python3 src/design/ev1_coverage.py theta --config $C --scenario $s --workers "$W"
done
for s in null_homog null_hetero pos_hetero pos_sesoi null_highvar; do
  for lo in $(seq 0 100 900); do
    python3 src/design/ev1_coverage.py run --config $C --scenario $s --range "$lo:$((lo + 100))" --workers "$W"
  done
done
python3 src/design/ev1_coverage.py run --config $C --scenario null_hetero --range 0:30 --mode tuned --workers "$W"
python3 src/design/ev1_coverage.py summarise --config $C
