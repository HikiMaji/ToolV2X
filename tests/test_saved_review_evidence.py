"""Check real archived evidence without loading a model or the external dataset."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from planning.inputs import make_prompt, parse_q8, parse_q9


def read(relative):
    return json.loads((ROOT / 'outputs' / relative).read_text())


class SavedReviewEvidenceTests(unittest.TestCase):
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
