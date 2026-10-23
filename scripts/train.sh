#!/usr/bin/env bash
# Train PRISM for one task (or all).
#
#   bash scripts/train.sh <pusht|cube|tworoom|reacher|all> [--world-model] [extra args for train_prior_head.py]
#
#   1. (--world-model only) train the LeWM world model from scratch: 100 epochs,
#      written to $STABLEWM_HOME/<wm dir>/; the last epoch is copied to lewm_object.ckpt
#      (an existing lewm_object.ckpt is kept as lewm_object.ckpt.bak).
#      Without the flag, the world model from scripts/download.sh is used.
#   2. cache frozen encoder features      → $STABLEWM_HOME/jepa_cache/<task>/   (4-11 min per task)
#   3. train the prior head (β-NLL)       → checkpoints/prior_head_<task>.pt     (2-5 min per task)
set -euo pipefail
source "$(dirname "$0")/common.sh"

[ $# -ge 1 ] || { sed -n '2,13p' "$0"; exit 1; }
TASKS="$(expand_tasks "$1")"; shift
TRAIN_WM=0; HEAD_ARGS=()
for a in "$@"; do
  if [ "$a" = "--world-model" ]; then TRAIN_WM=1; else HEAD_ARGS+=("$a"); fi
done

cd "$REPO"
for TASK in $TASKS; do
  task_vars "$TASK"
  echo "=== $TASK  (STABLEWM_HOME=$STABLEWM_HOME)"

  if [ "$TRAIN_WM" = 1 ]; then
    echo "--- [1/3] world model (config/train/data/$DATA.yaml)"
    python train.py data="$DATA" subdir="$WMDIR"
    last="$(ls -v "$STABLEWM_HOME/$WMDIR"/lewm_epoch_*_object.ckpt | tail -1)"
    [ -f "$STABLEWM_HOME/$WMDIR/lewm_object.ckpt" ] && \
      mv "$STABLEWM_HOME/$WMDIR/lewm_object.ckpt" "$STABLEWM_HOME/$WMDIR/lewm_object.ckpt.bak"
    cp "$last" "$STABLEWM_HOME/$WMDIR/lewm_object.ckpt"
    echo "    $last → $STABLEWM_HOME/$WMDIR/lewm_object.ckpt"
  fi

  echo "--- [2/3] feature cache"
  python jepa_cache.py --task "$TASK"

  echo "--- [3/3] prior head"
  python train_prior_head.py --task "$TASK" ${HEAD_ARGS[@]+"${HEAD_ARGS[@]}"}
done
