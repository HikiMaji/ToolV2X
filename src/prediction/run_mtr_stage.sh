#!/bin/bash
# ToolV2X MTR 阶段一键脚本。前置：cmp 环境已建好；outputs/cmp_trajs/{...}/train 与 test 都已生成。
# 用法:  bash run_mtr_stage.sh prepare      # 生成独立因果窗口（训练划分见 protocol_audit manifest）
#        bash run_mtr_stage.sh train  <cfg> # 训练某个 cfg（local_f_egoP / local_f_coopP / remote_f_egoP / remote_f_coopP）
#        bash run_mtr_stage.sh test   <cfg> [ckpt]
# 旧 GT 辅助诊断训练顺序: local_f_* 先跑完（30 epoch），remote_f_* 的 PRETRAINED_MOTION_TRANSFORMER 指向对应 local ckpt。
set -euo pipefail
unset http_proxy https_proxy
DM=/root/autodl-tmp/conda-envs/dmstrack/bin/python
CMPPY=/root/autodl-tmp/conda-envs/cmp/bin/python
SRC=/root/autodl-tmp/ToolV2X/src
OUT=/root/autodl-tmp/ToolV2X/outputs
CMP=/root/autodl-tmp/CMP

# Native CMP is still a GT-assisted diagnostic path. Never silently call it causal.
if [[ "${1:-}" == train || "${1:-}" == test ]]; then
  if [[ "${TOOLV2X_PROTOCOL:-}" != legacy-gt-diagnostic ]]; then
    echo "HOLD: native CMP inputs/aggregator do not satisfy docs/experiment_protocol.md." >&2
    echo "Historical diagnostic only: explicitly set TOOLV2X_PROTOCOL=legacy-gt-diagnostic." >&2
    exit 2
  fi
  echo "LEGACY GT-ASSISTED DIAGNOSTIC: not causal ToolV2X training/evaluation." >&2
fi
case "${1:-}" in
prepare)
  # Write separate causal windows; preserve all legacy pickles.
  manifest=$OUT/protocol_audit/split_manifest.json
  [ -f "$manifest" ] || { echo "run src/common/audit_protocol.py first" >&2; exit 2; }
  for split in train test; do
    for config in no_fusion no_fusion_cav1 cobevt; do
      $DM "$SRC/prediction/causal_windows.py" --config "$config" --split "$split" --manifest "$manifest"
    done
  done
  ;;
train)
  cfg=$2; cd $CMP
  PYTHONPATH=$CMP:$CMP/MTR $CMPPY MTR/tools/train_multiego.py --cfg_file MTR/tools/cfgs/toolv2x/$cfg.yaml \
     --launcher none --batch_size 1 --workers 4 --extra_tag default --not_eval_with_train \
     2>&1 | tee $OUT/logs/mtr_train_$cfg.log
  ;;
test)
  cfg=$2; ckpt=${3:-$CMP/MTR/output/$cfg/default/ckpt/checkpoint_epoch_30.pth}; cd $CMP
  PYTHONPATH=$CMP:$CMP/MTR $CMPPY MTR/tools/test_multiego.py --cfg_file MTR/tools/cfgs/toolv2x/$cfg.yaml \
     --launcher none --batch_size 1 --workers 4 --extra_tag default --ckpt $ckpt --save_to_file \
     2>&1 | tee $OUT/logs/mtr_test_$cfg.log
  ;;
*) echo "usage: $0 prepare | train <cfg> | test <cfg> [ckpt]"; exit 1;;
esac
