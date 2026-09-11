"""Check real archived evidence without loading a model or the external dataset."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from planning.inputs import make_prompt, parse_q8, parse_q9


def read(relative):
    return json.loads((ROOT / 'outputs' / relative).read_text())


class SavedReviewEvidenceTests(unittest.TestCase):
    def test_framework_generations_match_causal_prompts_and_independent_metrics(self):
        # These original records keep workstation paths; resolve only inside this archive.
        def archived(path):
            return ROOT / Path(path).relative_to('/root/autodl-tmp/ToolV2X')
        def jsonl(path):
            return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        records = jsonl(ROOT/'outputs/framework_generation_v1/generations.jsonl')
        labels = {r['sample_id']:r for r in jsonl(ROOT/'outputs/framework_review_2026_09_11/offline_validation_labels.jsonl')}
        metrics = {(r['sample_id'],r['policy']):r for r in jsonl(ROOT/'outputs/framework_evaluation_v1/rows.jsonl')}
        self.assertEqual(len(records), 20)
        self.assertEqual(len({(r['sample_id'],r['policy']) for r in records}), 20)
        repeated = 0
        for record in records:
            saved = json.loads(archived(record['generation_path']).read_text())
            task = json.loads(archived(saved['task_path']).read_text())
            plan = saved['plan']
            self.assertEqual(task['role'], 'validation')
            self.assertFalse(plan['q8_executed'])
            self.assertEqual(plan['q9_prompt'], make_prompt('Trajectory', task['ego_motion'],
                plan['evidence_used'], evidence_format='compact', remote_evidence=plan['remote_evidence_used']))
            points = parse_q9(plan['q9_raw'])
            label = labels[task['sample_id']]
            self.assertEqual(label['g'], task['g'])
            self.assertTrue(all(label['valid']))
            distance = np.sqrt(((points-np.asarray(label['waypoints']))**2).sum(axis=1))
            row = metrics[(record['sample_id'],record['policy'])]
            self.assertAlmostEqual(float(distance.mean()), row['ADE3'])
            self.assertAlmostEqual(float(distance[-1]), row['FDE3'])
            repeated += bool(np.max(np.linalg.norm(points-points[0],axis=1)) < 1e-6)
        self.assertEqual(repeated, 1, 'the physically degenerate raw answer must remain in the archive')

    def test_framework_archive_keeps_actual_query_costs_and_early_stops(self):
        from probe.kinematic_tools import encode
        paths = list((ROOT/'outputs/framework_debug_v2/frames').glob('validation_g*/*/task.json'))
        self.assertEqual(len(paths), 20)
        paths += [ROOT/('outputs/framework_debug_v2/frames/train_g%d/rule/task.json'%g) for g in (2932,5465)]
        for path in paths:
            task = json.loads(path.read_text())
            episode = task['episode']
            calls = [s for s in episode['steps'] if 'request' in s]
            self.assertEqual(episode['cost']['calls'], len(calls))
            self.assertEqual(episode['cost']['request_bytes'], sum(len(encode(s['request'])) for s in calls))
            self.assertEqual(episode['cost']['response_bytes'], sum(len(encode(s['response'])) for s in calls))
            self.assertEqual(episode['steps'][-1]['decision']['tool'], 'STOP')
            if task['policy'] == 'Ego':
                self.assertEqual(task['remote_reads'], [])
                self.assertEqual(calls, [])
        self.assertEqual(read('framework_debug_v2/frames/train_g2932/rule/task.json')['episode']['cost']['calls'], 0)
        self.assertEqual(read('framework_debug_v2/frames/train_g5465/rule/task.json')['episode']['cost']['calls'], 1)

    def test_full_p_recomputation_matches_f_for_the_same_peer_targets(self):
        evidence = read('framework_connection_v3/PF/evidence_full.json')
        local = {o['track_id']: o for o in evidence['objects'] if o['source'].endswith(':P_local')}
        remote = {o['track_id']: o for o in evidence['objects'] if o['source'].endswith(':F')}
        self.assertEqual(set(local), set(remote))
        self.assertEqual(len(local), 30)
        for track_id in local:
            for key in ('forecast', 'forecast_scores', 'forecast_times', 'model_used'):
                self.assertEqual(local[track_id][key], remote[track_id][key])

    def test_real_compact_failure_is_preserved_beside_parseable_baseline(self):
        for action in ('Ego', 'P', 'F', 'PF'):
            plan = read('framework_connection_v3/' + action + '/plan.json')
            parse_q8(plan['q8_raw'])
            self.assertEqual(len(parse_q9(plan['q9_raw'])), 6)
        failed = read('framework_compact_v1/P/plan.json')
        self.assertEqual(failed['status'], 'invalid_q9')
        with self.assertRaises(ValueError):
            parse_q9(failed['q9_raw'])

    def test_saved_supervision_uses_the_generated_parent_and_separate_targets(self):
        rows = [json.loads(line) for line in
                (ROOT / 'outputs/adaptation_data_v1/examples/train.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 8)
        self.assertEqual(len({row['sample_id'] for row in rows}), 1)
        meta = read('framework_compact_train_v1/connection.json')
        self.assertEqual(meta['research_split'], 'train')
        for row in rows:
            plan = read('framework_compact_train_v1/' + row['action'] + '/plan.json')
            parent = plan['q8_raw'] if row['task'] == 'Q9' else None
            prompt = make_prompt(row['task'], meta['ego_motion'], plan['evidence_used'],
                                 parent, evidence_format=row['evidence_format'])
            self.assertEqual(row['prompt'], prompt)
            self.assertEqual(row['supervision'], 'assistant target only')
            self.assertTrue(row['target'])
        forward = read('adaptation_forward_v1/forward_check.json')
        self.assertEqual(forward['optimizer_steps'], 0)


if __name__ == '__main__':
    unittest.main()
