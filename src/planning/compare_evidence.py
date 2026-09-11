"""Compare encoding at fixed information, then separately compare budget coverage."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from planning.inputs import make_prompt, pack_evidence, unpack_evidence


def compare_saved_wire(baseline, expanded, action):
    """Compare saved request/response bytes; evidence equality is insufficient.

    Requests are stored as formatted JSON by the runner; responses are the wire
    bytes. This checks file equality, not the network size of formatted requests.
    """
    if action not in ('Ego', 'P', 'F', 'PF'):
        raise ValueError('unknown action')
    compared, missing, different = [], [], []
    for tool in (() if action == 'Ego' else tuple(action)):
        for kind in ('request', 'response'):
            name = tool + '_' + kind + '.json'
            left, right = Path(baseline) / name, Path(expanded) / name
            absent = [str(path) for path in (left, right) if not path.is_file()]
            if absent:
                missing.extend(absent)
                continue
            compared.append(name)
            if left.read_bytes() != right.read_bytes():
                different.append(name)
    return dict(checked=not missing, equal=None if missing else not different,
                compared_files=compared, missing_files=missing, different_files=different,
                method='direct saved-file byte comparison')


def compare(out, baseline, expanded):
    from planning.v2vgot import V2VGoTPlanner, prompt_tokens, rounded
    from planning.run_connection import save_json, snapshot_code
    out.mkdir(parents=True, exist_ok=False)
    snapshot_code(out)
    read = lambda path: json.loads(path.read_text())
    meta, other = read(baseline / 'connection.json'), read(expanded / 'connection.json')
    assert (meta['scene'], meta['g'], meta['ego_motion']) == (other['scene'], other['g'], other['ego_motion'])
    features = dict(np.load(baseline / 'ego_features.npz', allow_pickle=False))
    for key, value in features.items():
        np.testing.assert_array_equal(value, np.load(expanded / 'ego_features.npz', allow_pickle=False)[key])
    planner = V2VGoTPlanner(evidence_format='compact')
    save_json(out / 'model_loading.json', planner.provenance)
    motion, feature_count = meta['ego_motion'], int(features['active_agent_mask'].sum()) * 270
    prototype = 'The suggested speed setting is: very slow. The suggested steering setting is: slightly right.'

    def tokens(evidence, mode):
        return max(len(prompt_tokens(planner.tokenizer, make_prompt(task, motion, evidence,
            q8_answer=prototype if task == 'Q9' else None, evidence_format=mode)))
            for task in ('Q8', 'Q9')) - 1 + feature_count

    report = dict(scope='encoding and same-frame input coverage; no task quality claim',
                  baseline=str(baseline), compact_expanded=str(expanded), scene=meta['scene'], g=meta['g'], actions={})
    for action in meta['actions']:
        wire = compare_saved_wire(baseline / action, expanded / action, action)
        full = read(baseline / action / 'evidence_full.json')
        assert full == read(expanded / action / 'evidence_full.json')
        selected = read(baseline / action / 'evidence_used.json')
        selected_new = read(expanded / action / 'evidence_used.json')
        for evidence in (full, rounded(full), selected, selected_new):
            assert unpack_evidence(pack_evidence(evidence)) == evidence
        result = planner.plan(features, motion, selected)
        assert result['evidence_used'] == selected, 'fixed-information encoding changed the selected evidence'
        save_json(out / (action + '_same_set_compact.json'), result)
        old_plan = read(baseline / action / 'plan.json')
        new_plan = read(expanded / action / 'plan.json')
        row = dict(full_records=len(full['objects']), baseline_records=len(selected['objects']),
            compact_expanded_records=len(selected_new['objects']),
            full_precision_json_tokens=tokens(full, 'json'), full_precision_compact_tokens=tokens(full, 'compact'),
            rounded_json_tokens=tokens(rounded(full), 'json'), rounded_compact_tokens=tokens(rounded(full), 'compact'),
            same_set_json_tokens=tokens(selected, 'json'), same_set_compact_tokens=tokens(selected, 'compact'),
            baseline_status=old_plan['status'], same_set_compact_status=result['status'],
            expanded_compact_status=new_plan['status'], full_wire_unchanged=wire['equal'],
            wire_comparison=wire, roundtrip_exact=True)
        report['actions'][action] = row
        save_json(out / 'comparison.json', report)
        print(json.dumps(dict(action=action, **row)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    parser.add_argument('--baseline', type=Path, default=ROOT / 'outputs/framework_connection_v3')
    parser.add_argument('--expanded', type=Path, default=ROOT / 'outputs/framework_compact_v1')
    args = parser.parse_args()
    compare(args.out, args.baseline, args.expanded)
