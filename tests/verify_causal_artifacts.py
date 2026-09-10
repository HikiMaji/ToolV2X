"""Validate real saved windows and independently recompute sampled coordinates."""
import glob
import json
import os
import pickle
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from common import v2v4real_meta as M
from prediction.causal_windows import make_window

counts = {'files': 0, 'windows': 0, 'targets': 0, 'prefix_checks': 0, 'coordinate_checks': 0}
sources = {}
for path in sorted(glob.glob(os.path.join(ROOT, 'outputs', 'causal_windows_v1', '*', '*', '*.pkl'))):
    with open(path, 'rb') as f:
        artifact = pickle.load(f)
    meta = artifact['meta']
    assert meta['gt_access'] is False and meta['coordinate_frame'] == 'ego_at_t'
    source_path = meta['source_path']
    if source_path not in sources:
        with open(source_path, 'rb') as f:
            sources[source_path] = pickle.load(f)['seqs']
    counts['files'] += 1
    counts['windows'] += len(artifact['windows'])
    times = sorted(artifact['windows'])
    sampled = {times[0], times[len(times) // 2], times[-1]}
    for t, w in artifact['windows'].items():
        n = len(w['track_ids'])
        assert w['states'].shape == (n, 11, 7)
        assert w['valid'].shape == (n, 11)
        assert np.isfinite(w['states']).all()
        assert np.all(w['states'][~w['valid']] == 0)
        assert np.all(w['valid'][:, -1])
        counts['targets'] += n
        if t not in sampled:
            continue
        g = w['g']
        seq, local, start, _ = M.seq_of(g, meta['split'])
        assert local == t
        frame_data = sources[source_path][seq]
        pose = M.load_pose(meta['split'], g)
        truncated = {k: v for k, v in frame_data.items() if k <= g}
        prefix = make_window(truncated, g, start, pose, w['source'])
        np.testing.assert_array_equal(w['states'], prefix['states'])
        np.testing.assert_array_equal(w['valid'], prefix['valid'])
        counts['prefix_checks'] += 1
        # Independent homogeneous-coordinate formula, not the production helper.
        for k, tid in enumerate(w['track_ids']):
            for h, past in enumerate(range(g - 10, g + 1)):
                rows = frame_data[past]
                matches = rows[rows[:, 0] == tid]
                assert len(matches) == int(w['valid'][k, h])
                if not len(matches):
                    continue
                row = matches[0]
                xyz = np.linalg.solve(pose, np.r_[row[1:4], 1.])[:3]
                np.testing.assert_allclose(w['states'][k, h, :3], xyz, atol=2e-3, rtol=0)
                counts['coordinate_checks'] += 1
assert counts['files'] == 78, counts
assert counts['windows'] == 17664, counts
out = os.path.join(ROOT, 'outputs', 'protocol_audit', 'artifact_validation.json')
with open(out, 'w') as f:
    json.dump(dict(status='PASS', **counts), f, indent=2)
    f.write('\n')
print(json.dumps(dict(status='PASS', **counts), indent=2))
