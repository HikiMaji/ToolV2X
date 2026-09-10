"""Independently check real saved packets, target retention, costs and forecasts."""
import argparse
import json
import pickle
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from common import v2v4real_meta as M


def verify(run_dir):
    path = Path(run_dir)
    meta = json.loads((path / 'run.json').read_text())
    decisions = [json.loads(line) for line in (path / 'decisions.jsonl').read_text().splitlines()]
    messages = [json.loads(line) for line in (path / 'messages.jsonl').read_text().splitlines()]
    assert len(decisions) == len(meta['frames']) and len(messages) == 2 * len(decisions)
    assert [r['t'] for r in decisions] == meta['frames']
    assert not meta['RSU'] and not meta['MTR'] and not meta['gt_used_in_generation']
    lookup = {(r['t'], r['service']): r for r in messages}
    assert len(lookup) == len(messages)
    windows = {}
    for source, source_path in meta['source_paths'].items():
        with open(source_path, 'rb') as f:
            windows[source] = pickle.load(f)['windows']
    counts = {'frames': len(decisions), 'messages': len(messages), 'forecast_targets_recomputed': 0,
              'peer_only_target_records': 0, 'equal_P_F_PF_frames': 0, 'causal_references_recomputed': 0,
              'max_reference_abs_error_m': 0.}
    for record in decisions:
        t, g = record['t'], record['g']
        packets = {}
        for service in ('P', 'F'):
            line = lookup[(t, service)]
            request, response = json.loads(line['request_json']), json.loads(line['response_json'])
            assert request['g'] == response['g'] == line['g'] == g
            assert line['request_bytes'] == len(line['request_json'].encode('utf-8'))
            assert line['response_bytes'] == len(line['response_json'].encode('utf-8'))
            assert line['network_ms'] is None and line['sender_detection_tracking_ms'] is None
            packets[service] = response
        peer_window = windows['no_fusion_cav1'][t]
        for packet, source in [(record['ego_forecast'], 'no_fusion'), (packets['F'], 'no_fusion_cav1')]:
            w = windows[source][t]
            by_key = {o['key']: o for o in packet['objects']}
            assert set(by_key) == {'%s:%d' % (source, i) for i in w['track_ids']}
            for i, tid in enumerate(w['track_ids']):
                obj = by_key['%s:%d' % (source, tid)]
                mask = w['valid'][i]
                if mask.sum() >= 2:
                    # Separate polynomial fitter, not the production covariance formula.
                    velocity = np.polyfit(w['time_seconds'][mask], w['states'][i, mask, :2], 1)[0]
                else:
                    velocity = np.zeros(2)
                    assert obj['fallback'] == 'stationary_short_history'
                expected = w['states'][i, -1, :2] + np.arange(1, 51)[:, None] * .1 * velocity
                np.testing.assert_allclose(obj['trajectory'], expected, atol=2e-5, rtol=0)
                counts['forecast_targets_recomputed'] += 1
        for i, obj in enumerate(packets['P']['objects']):
            assert obj['key'] == 'no_fusion_cav1:%d' % peer_window['track_ids'][i]
            np.testing.assert_array_equal(obj['history'], peer_window['states'][i])
            np.testing.assert_array_equal(obj['valid'], peer_window['valid'][i])
        ego_keys = {o['key'] for o in record['ego_forecast']['objects'] if np.linalg.norm(o['state'][:2]) >= 1.5}
        peer_keys = {o['key'] for o in packets['F']['objects'] if np.linalg.norm(o['state'][:2]) >= 1.5}
        for action, result in record['actions'].items():
            seen = [key for o in result['objects'] for key in o['evidence']]
            expected_keys = ego_keys if action == 'ego' else ego_keys | peer_keys
            assert set(seen) == expected_keys and len(seen) == len(set(seen))
            services = [] if action == 'ego' else list(action)
            assert result['used_services'] == services
            for field in ('request_bytes', 'response_bytes'):
                assert result[field] == sum(lookup[(t, s)][field] for s in services)
        a = record['actions']
        assert a['P']['objects'] == a['F']['objects'] == a['PF']['objects']
        assert a['P']['plan'] == a['F']['plan'] == a['PF']['plan']
        counts['equal_P_F_PF_frames'] += 1
        counts['peer_only_target_records'] += sum(o['evidence'][0] in peer_keys and len(o['evidence']) == 1
                                                for o in a['F']['objects'])
        pose = M.load_pose(meta['source_split'], g).astype(np.float64)
        past = np.array([(np.linalg.inv(pose) @ M.load_pose(meta['source_split'], h).astype(np.float64))[:2, 3]
                         for h in range(g - 10, g + 1)])
        velocity = np.polyfit(np.arange(-10, 1) * .1, past, 1)[0]
        expected_reference = np.arange(1, 31)[:, None] * .1 * velocity
        # Stored input poses and producer solve use float32. Allow 20 micrometres, rtol=0.
        np.testing.assert_allclose(record['reference'], expected_reference, atol=2e-5, rtol=0)
        counts['max_reference_abs_error_m'] = max(counts['max_reference_abs_error_m'],
            float(np.max(np.abs(np.asarray(record['reference']) - expected_reference))))
        counts['causal_references_recomputed'] += 1
    result = {'status': 'PASS', **counts,
              'scope': 'saved kinematic interface artifacts only; no method or safety claim'}
    (path / 'artifact_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    verify(ap.parse_args().run_dir)
