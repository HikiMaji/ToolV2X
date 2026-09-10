"""Check the real CMP predictor connection; deliberately no GT scoring or routing."""
import argparse
import builtins
import json
import pickle
import sys
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from prediction import cmp_adapter as C


def run(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    model, evidence = C.load_model()
    (out / 'model_loading.json').write_text(json.dumps(evidence, indent=2) + '\n')
    manifest = json.loads((ROOT / 'outputs/protocol_audit/split_manifest.json').read_text())
    scene = manifest['validation_scenes'][0]
    frames = [t for s, t in json.loads((ROOT / 'outputs/protocol_audit/validation_frames.json').read_text()) if s == scene]
    t = frames[0]
    real_open = builtins.open

    def no_gt(file, mode='r', *args, **kwargs):
        if 'r' in mode and ('_gt' in str(file) or '/learned_capability_v1/data/' in str(file)):
            raise AssertionError('GT/label read during CMP module check')
        return real_open(file, mode, *args, **kwargs)

    rows = []
    with patch('builtins.open', side_effect=no_gt):
        for source in ('no_fusion', 'no_fusion_cav1'):
            path = ROOT / 'outputs/causal_windows_v1/train' / source / (scene + '.pkl')
            with open(path, 'rb') as f:
                artifact = pickle.load(f)
            assert artifact['meta']['gt_access'] is False
            w = artifact['windows'][t]
            begin = perf_counter()
            prediction = C.predict(model, w)
            elapsed = perf_counter() - begin
            np.savez(str(out / (source + '.npz')), **prediction)
            row = dict(source=source, scene=scene, t=t, g=w['g'], input_path=str(path),
                current_targets=len(w['track_ids']), original_MTR_targets=int(prediction['model_used'].sum()),
                stationary_short_history_fallbacks=int((~prediction['model_used']).sum()),
                modes=6, horizon_frames=50, forward_seconds=elapsed, output=str(out / (source + '.npz')))
            rows.append(row)
            print(json.dumps(row), flush=True)
    result = dict(status='PASS', scope='CMP model loading and causal-input inference connection only',
                  GT_access_during_inference=False, quality_evaluation=False, training=False,
                  full_framework_complete=False, windows=rows)
    (out / 'module_check.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    run(ap.parse_args().out)
