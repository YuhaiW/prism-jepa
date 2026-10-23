#!/usr/bin/env bash
# Download data and pretrained weights for one task (or all).
#
#   bash scripts/download.sh <pusht|cube|tworoom|reacher|all> [--no-data] [--no-weights]
#
#   data     quentinll/lewm-<task> (Hugging Face dataset)  →  $STABLEWM_HOME/<dataset>.h5
#   weights  YuhaiW/prism-jepa-<task> (Hugging Face model)  →  $STABLEWM_HOME/<wm dir>/lewm_object.ckpt
#                                                             checkpoints/prior_head_<task>.pt
#
# Compressed dataset sizes: PushT 13 GB, Cube 46 GB, TwoRoom 3.4 GB, Reacher 24 GB
# (each expands 3-4x). Requires the `hf` CLI (huggingface_hub) and `zstd`.
set -euo pipefail
source "$(dirname "$0")/common.sh"

[ $# -ge 1 ] || { sed -n '2,12p' "$0"; exit 1; }
TASKS="$(expand_tasks "$1")"; shift
GET_DATA=1; GET_WEIGHTS=1
for a in "$@"; do
  case "$a" in
    --no-data) GET_DATA=0 ;;
    --no-weights) GET_WEIGHTS=0 ;;
    *) echo "unknown option $a" >&2; exit 1 ;;
  esac
done

mkdir -p "$STABLEWM_HOME" "$REPO/checkpoints"
for TASK in $TASKS; do
  task_vars "$TASK"
  echo "=== $TASK  (STABLEWM_HOME=$STABLEWM_HOME)"

  if [ "$GET_DATA" = 1 ]; then
    if [ -f "$STABLEWM_HOME/$H5" ]; then
      echo "  dataset present: $STABLEWM_HOME/$H5"
    else
      tmp="$STABLEWM_HOME/hf_${UP}_data"
      hf download "quentinll/lewm-$UP" --repo-type dataset --local-dir "$tmp"
      mkdir -p "$(dirname "$STABLEWM_HOME/$H5")"
      if [ "$TASK" = pusht ]; then
        zstd -d "$tmp/pusht_expert_train.h5.zst" -o "$STABLEWM_HOME/$H5"
      else
        # Archives hold a single .h5 (cube_single_expert.h5, tworoom.h5, reacher.h5).
        zstd -dc "$tmp"/*.tar.zst | tar -x -C "$tmp"
        mv "$tmp"/*.h5 "$STABLEWM_HOME/$H5"
      fi
      echo "  dataset → $STABLEWM_HOME/$H5  (archive kept in $tmp; delete it to free space)"
    fi
  fi

  if [ "$GET_WEIGHTS" = 1 ]; then
    tmp="$(mktemp -d -p "$STABLEWM_HOME")"
    hf download "YuhaiW/prism-jepa-$TASK" lewm_object.ckpt "prior_head_$TASK.pt" --local-dir "$tmp"
    mkdir -p "$STABLEWM_HOME/$WMDIR"
    mv "$tmp/lewm_object.ckpt" "$STABLEWM_HOME/$WMDIR/lewm_object.ckpt"
    mv "$tmp/prior_head_$TASK.pt" "$REPO/checkpoints/prior_head_$TASK.pt"
    rm -rf "$tmp"
    echo "  world model → $STABLEWM_HOME/$WMDIR/lewm_object.ckpt"
    echo "  prior head  → checkpoints/prior_head_$TASK.pt"
  fi
done
