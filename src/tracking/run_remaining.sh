#!/bin/bash
unset http_proxy https_proxy
P=/root/autodl-tmp/conda-envs/dmstrack/bin/python
cd /root/autodl-tmp/ToolV2X/src/tracking
for c in no_fusion_cav1 cobevt; do
  [ -f ../../outputs/tracks/train/${c}_world.pkl ] && continue
  $P run_ab3dmot.py --config $c --split train --frame world 2>&1 | grep -E "^\[|saved"
done
echo TRACK_DONE
