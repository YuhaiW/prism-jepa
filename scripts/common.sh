# Shared per-task settings for scripts/{download,train,test}.sh. Source, don't run.
#
#   CFG     Hydra config name in config/eval/      POLICY  world model under $STABLEWM_HOME
#   WMDIR   world-model directory                  DATA    world-model training config (config/train/data/)
#   UP      upstream LeWM repo suffix (quentinll/lewm-<UP>)
#   H5      dataset path relative to $STABLEWM_HOME

export STABLEWM_HOME="${STABLEWM_HOME:-$HOME/.stable-wm}"
export PYTHONUNBUFFERED=1   # progress lines reach the log files as they are printed
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASKS_ALL="pusht cube tworoom reacher"

task_vars() {
  case "$1" in
    pusht)   CFG=pusht;   POLICY=pusht/lewm;    WMDIR=pusht;    DATA=pusht;   UP=pusht;    H5=pusht_expert_train.h5 ;;
    cube)    CFG=cube;    POLICY=cube/lewm;     WMDIR=cube;     DATA=ogb;     UP=cube;     H5=ogbench/cube_single_expert.h5 ;;
    tworoom) CFG=tworoom; POLICY=tworooms/lewm; WMDIR=tworooms; DATA=tworoom; UP=tworooms; H5=tworoom.h5 ;;
    reacher) CFG=reacher; POLICY=reacher/lewm;  WMDIR=reacher;  DATA=dmc;     UP=reacher;  H5=dmc/reacher_random.h5 ;;
    *) echo "unknown task '$1' (expected one of: $TASKS_ALL)" >&2; return 1 ;;
  esac
}

expand_tasks() {  # "all" -> every task
  if [ "$1" = "all" ]; then echo "$TASKS_ALL"; else echo "$1"; fi
}
