#!/usr/bin/env bash
# Evaluate PRISM and the prior-free baselines for one task (or all) and print a summary.
#
#   bash scripts/test.sh <pusht|cube|tworoom|reacher|all> [--K "32 64 128"] [--seeds "0"]
#                        [--methods "vanilla warm_start prism"] [--solver mppi|cem]
#
#   vanilla     planner default N(0, σ_π²)                        eval.py
#   warm_start  prior mean μ_p, planner width σ_π                 eval_prism_head.py
#   prism       PoG-fused (μ_p, σ_p), prior scale s = 1           eval_prism_head.py
#   mean_only   execute μ_p without sampling (paper Table II)     eval_prism_head.py
#
# 50 episodes per run, goal 25 env steps ahead, H = 5, B = 5, J = 30, K_elite = 30.
# The paper's Fig. 5 is --seeds "0 1 2 3 4 5 6 7 8 9" (90 runs per task, 1-3 min each).
# Each run appends one row to results/<task>.tsv; the summary averages over all rows there.
set -euo pipefail
source "$(dirname "$0")/common.sh"

[ $# -ge 1 ] || { sed -n '2,16p' "$0"; exit 1; }
TASKS="$(expand_tasks "$1")"; shift
KS="32 64 128"; SEEDS="0"; METHODS="vanilla warm_start prism"; SOLVER=mppi
while [ $# -gt 0 ]; do
  case "$1" in
    --K) KS="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --methods) METHODS="$2"; shift 2 ;;
    --solver) SOLVER="$2"; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 1 ;;
  esac
done

cd "$REPO"
mkdir -p results logs
for TASK in $TASKS; do
  task_vars "$TASK"
  HEAD="checkpoints/prior_head_$TASK.pt"
  for K in $KS; do
    for SEED in $SEEDS; do
      for M in $METHODS; do
        common=(--config-name="$CFG" policy="$POLICY" solver="$SOLVER" solver.num_samples="$K" seed="$SEED")
        log="logs/${TASK}_${SOLVER}_${M}_K${K}_s${SEED}.log"
        echo "=== $TASK | $SOLVER | $M | K=$K | seed=$SEED   (log: $log)"
        case "$M" in
          vanilla)    cmd=(python eval.py "${common[@]}") ;;
          warm_start) cmd=(python eval_prism_head.py "${common[@]}" +head.ckpt="$HEAD" +head.injection_mode=warm_start) ;;
          prism)      cmd=(python eval_prism_head.py "${common[@]}" +head.ckpt="$HEAD" +head.injection_mode=pog +head.sigma_scale=1.0) ;;
          mean_only)  cmd=(python eval_prism_head.py "${common[@]}" +head.ckpt="$HEAD" +head.injection_mode=bc_only) ;;
          *) echo "unknown method $M" >&2; exit 1 ;;
        esac
        if ! "${cmd[@]}" > "$log" 2>&1; then
          echo "  run failed — see $log (last lines below)"; tail -n 15 "$log"; exit 1
        fi
        grep -m1 "^success_rate:" "$log"
      done
    done
  done

  python - "results/$TASK.tsv" <<'EOF'
import csv, statistics, sys
from collections import defaultdict
cells = defaultdict(list)
for r in csv.DictReader(open(sys.argv[1]), delimiter="\t"):
    cells[(r["solver"], r["method"], int(r["K"]))].append(float(r["success_rate"]))
print(f"\nSummary {sys.argv[1]}  (success rate %, mean ± std over seeds)")
for (solver, method, K), v in sorted(cells.items()):
    sd = statistics.stdev(v) if len(v) > 1 else 0.0
    print(f"  {solver:<5} {method:<11} K={K:<4} {statistics.mean(v):5.1f} ± {sd:4.1f}   (n={len(v)})")
EOF
done
