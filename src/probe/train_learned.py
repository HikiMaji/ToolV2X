"""Fit the preregistered shared ridge model using offline training labels only."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from probe import kinematic_tools as K
from probe import learned_predictor as L
from probe.learned_data import paired_errors

ALPHAS = (.001, .01, .1)


def sufficient_statistics(x, residual, mask):
    gram = np.zeros((50, L.FEATURES, L.FEATURES))
    rhs = np.zeros((50, L.FEATURES, 2))
    count = mask.sum(axis=0).astype(float)
    for h in range(50):
        a = np.asarray(x[mask[:, h]], dtype=float)
        b = np.asarray(residual[mask[:, h], h], dtype=float)
        gram[h] = a.T @ a
        rhs[h] = a.T @ b
    return gram, rhs, count


def solve(gram, rhs, count, alpha):
    if (count <= 0).any() or alpha <= 0:
        raise ValueError('missing horizon supervision or invalid regularization')
    penalty = np.eye(L.FEATURES)
    penalty[0, 0] = 0.
    return np.linalg.solve(gram / count[:, None, None] + alpha * penalty,
                           rhs / count[:, None, None])


def train(data_dir, out):
    data_dir, out = Path(data_dir), Path(out)
    manifest = json.loads((data_dir / 'manifest.json').read_text())
    # Exact role lists, not a directory glob that could accidentally include validation.
    current = json.loads((ROOT / 'outputs/protocol_audit/split_manifest.json').read_text())
    if manifest != current:
        raise ValueError('data manifest differs from the frozen split')
    out.mkdir(parents=True, exist_ok=False)
    gram = np.zeros((50, L.FEATURES, L.FEATURES))
    rhs, count = np.zeros((50, L.FEATURES, 2)), np.zeros(50)
    for scene in manifest['train_scenes']:
        with np.load(str(data_dir / 'train' / (scene + '.npz')), allow_pickle=False) as d:
            mask = d['mask'] & d['eligible'][:, None]
            for view in ('history', 'state'):
                a, b, c = sufficient_statistics(d['x_' + view], d['residual'], mask)
                gram += a
                rhs += b
                count += c
        print('fitted statistics: ' + scene, flush=True)
    models = {str(alpha): solve(gram, rhs, count, alpha) for alpha in ALPHAS}
    validation = []
    for scene in manifest['validation_scenes']:
        with np.load(str(data_dir / 'validation' / (scene + '.npz')), allow_pickle=False) as d:
            truth, mask = d['residual'], d['mask']
            for alpha, w in models.items():
                for view in ('history', 'state'):
                    prediction = np.einsum('nf,hfc->nhc', d['x_' + view], w)
                    prediction[~d['eligible']] = 0.
                    for source in (0, 1):
                        select = d['source'] == source
                        scores = paired_errors(np.zeros_like(truth[select]), prediction[select],
                                               truth[select], mask[select])
                        validation.append(dict(scene=scene, source=source, alpha=alpha, view=view, **scores))
    objectives = {alpha: float(np.mean([r['ADE_b'] for r in validation
        if r['alpha'] == alpha and r['view'] == 'history'])) for alpha in models}
    chosen = min(objectives, key=lambda a: (objectives[a], float(a)))
    metadata = {'predictor': L.PREDICTOR, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'features': L.FEATURES, 'chosen_alpha': float(chosen), 'candidate_alphas': list(ALPHAS),
        'train_scenes': manifest['train_scenes'], 'validation_scenes': manifest['validation_scenes'],
        'fit_sources': ['no_fusion', 'no_fusion_cav1'], 'fit_views': ['history', 'state'],
        'horizon_observation_counts_both_views': count.tolist(), 'data_dir': str(data_dir.resolve()),
        'selection': 'macro scene/source history-view observed ADE; validation only',
        'test_used': False, 'MTR': False, 'scope': 'single-target ridge capability baseline',
        'validation_objectives': objectives}
    np.savez(str(out / 'model.npz'), weights=models[chosen], metadata=np.array(json.dumps(metadata)))
    np.savez(str(out / 'fit_statistics.npz'), gram=gram, rhs=rhs, count=count)
    (out / 'training.json').write_bytes(K.encode(metadata) + b'\n')
    (out / 'validation_candidates.json').write_bytes(K.encode(validation) + b'\n')
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('data_dir')
    ap.add_argument('out')
    a = ap.parse_args()
    train(a.data_dir, a.out)
