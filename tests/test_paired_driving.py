"""Keep the ego information fixed when adding a remote capability."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from planning.inputs import make_prompt


class PairedInputTests(unittest.TestCase):
    def test_remote_block_preserves_literal_ego_block_and_rejects_wrong_time(self):
        ego = dict(as_of_g=50, coordinate_frame='ego_at_t', objects=[], relations=[], queries=[])
        remote = dict(ego, queries=[dict(tool='F', roi=None, response_bytes=300, status='ok')])
        motion = dict(speed_mps=4., yaw_rate_rps=0.)
        baseline = make_prompt('Trajectory', motion, ego, evidence_format='compact')
        try:
            extended = make_prompt('Trajectory', motion, ego, evidence_format='compact',
                                   remote_evidence=remote)
        except TypeError as exc:
            self.fail('independent acquired remote block missing: ' + str(exc))
        base_block = baseline.split('Output six numeric')[0]
        self.assertTrue(extended.startswith(base_block))
        self.assertEqual(extended.count('What is the suggested future trajectory'), 1)
        with self.assertRaises(ValueError):
            make_prompt('Trajectory', motion, ego, remote_evidence=dict(remote, as_of_g=51))
        with self.assertRaises(ValueError):
            make_prompt('Q8', motion, ego, remote_evidence=remote)
        with self.assertRaises(ValueError):
            make_prompt('Trajectory', motion, dict(ego, queries=remote['queries']), remote_evidence=remote)

    def test_discontinuity_rule_has_exact_closed_future_and_open_history_boundary(self):
        path = ROOT / 'scripts/prepare_paired_driving.py'
        self.assertTrue(path.is_file(), 'paired data preparation is not implemented')
        spec = importlib.util.spec_from_file_location('paired_preparation_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        jumps = [dict(scene='scene', to_g=100)]
        rejected = [g for g in range(60, 121)
                    if module.crossed_jumps(dict(scene='scene', g=g), jumps)]
        self.assertEqual(rejected, list(range(70, 110)))
        self.assertFalse(module.crossed_jumps(dict(scene='other', g=100), jumps))

    def test_direct_supervision_reuses_the_remote_block_without_q8(self):
        from planning.adaptation_data import supervised_examples
        ego = dict(as_of_g=50, coordinate_frame='ego_at_t', objects=[], queries=[])
        remote = dict(ego, queries=[dict(tool='F', roi=None, response_bytes=300, status='no_observed_targets')])
        motion = dict(speed_mps=4., yaw_rate_rps=0.)
        try:
            prompt = make_prompt('Trajectory', motion, ego, evidence_format='compact', remote_evidence=remote)
        except TypeError as exc:
            self.fail('paired prompt missing: ' + str(exc))
        plan = dict(decoding='direct', q8_executed=False, evidence_used=ego,
                    remote_evidence_used=remote, evidence_selection=dict(evidence_format='compact'), q9_prompt=prompt)
        rows = supervised_examples('train', 'sample', 'F', motion, plan,
            dict(target_q9='six points', target_q8='must not be used'), '/features.npz')
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['prompt'], prompt)
        self.assertEqual(rows[0]['task'], 'Trajectory')
        with self.assertRaises(ValueError):
            supervised_examples('validation', 'sample', 'F', motion, plan, {}, '/features.npz')


class PairedBudgetTests(unittest.TestCase):
    def test_local_selection_is_frozen_before_peer_content_and_all_modes_survive(self):
        import json
        from transformers import AutoTokenizer
        import planning
        path = ROOT / 'src/planning/paired_inputs.py'
        self.assertTrue(path.is_file(), 'paired input selector missing')
        from planning.paired_inputs import fit_local, fit_remote, input_size
        from planning.v2vgot import CHECKPOINT
        run = ROOT / 'outputs/framework_compact_train_v1'
        meta = json.loads((run / 'connection.json').read_text())
        ego = json.loads((run / 'Ego/evidence_full.json').read_text())
        combined = json.loads((run / 'F/evidence_full.json').read_text())
        remote = dict(combined, objects=[o for o in combined['objects'] if o['source'] != 'ego'])
        tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT), use_fast=False, local_files_only=True)
        local, _ = fit_local(tokenizer, meta['ego_motion'], ego, 540)
        before = copy.deepcopy(local)
        selected, _ = fit_remote(tokenizer, meta['ego_motion'], local, remote, 540)
        self.assertEqual(before, local)
        self.assertTrue(selected['objects'])
        for obj in selected['objects']:
            self.assertEqual(len(obj['forecast']), 6)
        self.assertLessEqual(input_size(tokenizer, meta['ego_motion'], local, 540, selected) + 256, 4096)
        poisoned = copy.deepcopy(remote)
        poisoned['objects'][0]['source'] = 'ego'
        with self.assertRaises(ValueError):
            fit_remote(tokenizer, meta['ego_motion'], local, poisoned, 540)


if __name__ == '__main__':
    unittest.main()
