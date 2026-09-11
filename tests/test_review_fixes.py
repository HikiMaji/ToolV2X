"""Regressions for the reviewed runner and original-tokenizer boundaries."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from common import v2v4real_meta as M
from planning import run_connection as R, v2vgot as V
from planning.inputs import FEATURE_SHAPES, make_prompt, parse_q8


class ResourceRegressionTests(unittest.TestCase):
    def test_explicit_adapter_directories_keep_candidate_checkpoints_separate(self):
        names = ('adapter_model.safetensors', 'adapter_config.json', 'non_lora_trainables.bin',
                 'tokenizer.model', 'tokenizer_config.json', 'special_tokens_map.json')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clip = root / 'clip'
            clip.mkdir()
            for name in ('config.json', 'preprocessor_config.json', 'pytorch_model.bin'):
                (clip / name).write_text('{}')
            adapters = []
            for candidate in ('got', 'llm'):
                checkpoint = root / candidate
                checkpoint.mkdir()
                for name in names + ('config.json',):
                    (checkpoint / name).write_text('{}')
                try:
                    adapter = V.local_adapter(checkpoint, clip,
                        directory=root / 'runs' / candidate / 'llava-toolv2x-lora-ego')
                except TypeError as exc:
                    self.fail('candidate-specific adapter directory missing: ' + str(exc))
                adapters.append(adapter)
                self.assertEqual((adapter / names[0]).resolve(), checkpoint / names[0])
            self.assertNotEqual(adapters[0], adapters[1])
            self.assertEqual((adapters[0] / names[0]).resolve(), root / 'got' / names[0])

    def test_prepare_reads_features_and_motion_from_configured_root(self):
        class StopBeforeModels(Exception):
            pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'data/train_no_fusion_keep_all/npy'
            features = data / 'co_llm/ego'
            features.mkdir(parents=True)
            for key, shape in FEATURE_SHAPES.items():
                np.save(features / ('0000_' + key + '.npy'), np.full(shape, 91., np.float32))
            np.save(features / '0000_detection_box_score.npy', np.zeros((0, 8), np.float32))
            (data / 'ego').mkdir()
            np.save(data / 'ego/0000_lidar_pose.npy', np.eye(4))
            out = root / 'out'
            out.mkdir()
            motion_paths = []
            real_motion = R.load_ego_motion

            def read_motion(*args):
                motion, paths = real_motion(*args)
                motion_paths.extend(paths)
                return motion, paths

            with patch.object(M, 'V2VGOT_ROOT', str(root / 'data')), \
                    patch.object(R, 'select_decision', return_value='scene'), \
                    patch.object(R, 'load_window', return_value={'g': 0}), \
                    patch.object(R, 'load_ego_motion', side_effect=read_motion), \
                    patch.object(R, 'load_projector', side_effect=StopBeforeModels):
                with self.assertRaises(StopBeforeModels):
                    R.prepare(out, 0, 0, ['Ego'])
            with np.load(out / 'ego_features.npz') as saved:
                self.assertTrue(np.all(saved['regression_map'][0, 0] == 91.))
            self.assertEqual(motion_paths, [str(data / 'ego/0000_lidar_pose.npy')])

    def test_adapter_links_resolve_for_relative_absolute_and_user_paths(self):
        names = ('adapter_model.safetensors', 'adapter_config.json', 'non_lora_trainables.bin',
                 'tokenizer.model', 'tokenizer_config.json', 'special_tokens_map.json')
        with tempfile.TemporaryDirectory(dir=str(Path.home())) as directory:
            root = Path(directory)
            checkpoint, clip = root / 'resources/checkpoint', root / 'resources/clip'
            for folder, files in ((checkpoint, names + ('config.json',)),
                                  (clip, ('config.json', 'preprocessor_config.json', 'pytorch_model.bin'))):
                folder.mkdir(parents=True)
                for name in files:
                    (folder / name).write_text('{}')
            previous = Path.cwd()
            try:
                os.chdir(root)
                for index, prefix in enumerate((root, Path('.'), Path('~') / root.name)):
                    with self.subTest(prefix=str(prefix)), patch.object(V, 'ROOT', root / ('project_' + str(index))):
                        adapter = V.local_adapter(prefix / 'resources/checkpoint', prefix / 'resources/clip')
                        for name in names:
                            self.assertTrue((adapter / name).is_file(), name)
                            self.assertEqual((adapter / name).resolve(), checkpoint / name)
                        config = json.loads((adapter / 'config.json').read_text())
                        self.assertEqual(config['mm_vision_tower'], str(clip))
                with patch.object(V, 'ROOT', root / 'project'):
                    conflict = root / 'project/models/llava-toolv2x-lora-ego' / names[0]
                    conflict.parent.mkdir(parents=True)
                    conflict.write_text('unrelated weights')
                    with self.assertRaises(FileExistsError):
                        V.local_adapter(checkpoint, clip)
                    self.assertEqual(conflict.read_text(), 'unrelated weights')
            finally:
                os.chdir(previous)


class ContextBudgetRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tokenizer = AutoTokenizer.from_pretrained(str(V.CHECKPOINT), use_fast=False, local_files_only=True)
        run = ROOT / 'outputs/framework_connection_v3'
        cls.motion = json.loads((run / 'connection.json').read_text())['ego_motion']
        cls.evidence = json.loads((run / 'P/evidence_full.json').read_text())
        cls.raw8 = 'The suggested speed setting is: very slow. The suggested steering setting is: slightly right.'
        while len(cls.tokenizer.encode(cls.raw8 + ' Additional caution.', add_special_tokens=False)) <= 127:
            cls.raw8 += ' Additional caution.'

    def test_selection_allows_a_long_valid_generated_q8_parent(self):
        parse_q8(self.raw8)
        self.assertLessEqual(len(self.tokenizer.encode(self.raw8, add_special_tokens=False)), 128)
        view, report = V.fit_evidence(self.tokenizer, self.motion, self.evidence, 540)
        q9 = make_prompt('Q9', self.motion, view, self.raw8)
        self.assertLessEqual(len(V.prompt_tokens(self.tokenizer, q9)) - 1 + 540 + 256, 4096)
        self.assertLessEqual(report['input_token_bound'] + report['reserved_generation_tokens'], 4096)

    def test_comparison_report_persists_the_measured_wire_difference(self):
        from planning.compare_evidence import compare
        from probe.kinematic_tools import encode
        from test_vehicle_tools import fixture_prediction, window
        from tools.vehicle import VehicleTools, make_evidence
        response = VehicleTools(lambda: window(), fixture_prediction, 'scene', 10, provider='peer').query('P')
        packet = json.loads(response['wire'])
        packet['objects'][0]['history_scores'][0] = .8
        local = window('ego')
        evidence = make_evidence(local, fixture_prediction(local), [packet], fixture_prediction)
        planner = SimpleNamespace(tokenizer=self.tokenizer, provenance={'test_only': True},
            plan=lambda features, motion, selected: dict(status='test_generation_skipped', evidence_used=selected))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline, expanded = root / 'baseline', root / 'expanded'
            for run, score in ((baseline, .8), (expanded, .9)):
                (run / 'P').mkdir(parents=True)
                R.save_json(run / 'connection.json', dict(scene='scene', g=10, ego_motion=self.motion, actions={'P': {}}))
                np.savez(run / 'ego_features.npz', active_agent_mask=np.ones((1, 2, 1), bool))
                for name in ('evidence_full.json', 'evidence_used.json'):
                    R.save_json(run / 'P' / name, evidence)
                R.save_json(run / 'P/plan.json', {'status': 'test_generation_skipped'})
                R.save_json(run / 'P/P_request.json', response['request'])
                packet['objects'][0]['history_scores'][0] = score
                (run / 'P/P_response.json').write_bytes(encode(packet))
            with patch.object(V, 'V2VGoTPlanner', return_value=planner), \
                    patch.object(R, 'snapshot_code'), contextlib.redirect_stdout(io.StringIO()):
                compare(root / 'comparison', baseline, expanded)
            row = json.loads((root / 'comparison/comparison.json').read_text())['actions']['P']
            self.assertFalse(row['full_wire_unchanged'])
            self.assertTrue(row['wire_comparison']['checked'])
            self.assertEqual(row['wire_comparison']['different_files'], ['P_response.json'])

    def test_q9_overflow_keeps_q8_and_infer_continues_to_the_next_action(self):
        planner = object.__new__(V.V2VGoTPlanner)
        planner.tokenizer, planner.context_limit, planner.evidence_format = self.tokenizer, 4096, 'json'
        planner.model = SimpleNamespace(device='cpu', dtype=torch.float32)
        planner.provenance = {'test_only_generation_boundary': True}
        features = {'regression_map': np.zeros((1, 2, 1, 14, 50, 88), np.float32),
                    'classification_map': np.zeros((1, 2, 1, 2, 50, 88), np.float32),
                    'detection_box_score': np.zeros((1, 2, 1, 50, 8), np.float32),
                    'active_agent_mask': np.ones((1, 2, 1), bool)}
        raw8 = self.raw8 + ' Additional caution.' * 1500
        calls = []

        def generate(current_features, prompt, limit):
            calls.append(limit)
            if len(calls) == 1:
                # Force the residual overflow guard; this is not a real 128-token generation.
                return raw8, {'output_tokens': 128, 'test_only': True}
            if len(calls) == 2:
                return V.V2VGoTPlanner._generate(planner, current_features, prompt, limit)
            return 'invalid next action', {'output_tokens': 3, 'test_only': True}

        planner._generate = generate
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            metadata = dict(evidence_format='json', ego_motion=self.motion, actions={'P': {}, 'Ego': {}})
            for action in metadata['actions']:
                (out / action).mkdir()
                R.save_json(out / action / 'evidence_full.json', self.evidence)
            with patch.object(R, 'V2VGoTPlanner', return_value=planner), contextlib.redirect_stdout(io.StringIO()):
                try:
                    R.infer(out, features, metadata)
                except ValueError as exc:
                    self.fail('Q9 overflow escaped instead of saving the action: ' + str(exc))
            failed = json.loads((out / 'P/plan.json').read_text())
            self.assertEqual(failed['status'], 'q9_context_overflow')
            self.assertEqual(failed['q8_raw'], raw8)
            self.assertIn(raw8, failed['q9_prompt'])
            self.assertEqual(failed['q8_cost']['output_tokens'], 128)
            self.assertFalse(failed['q9_executed'])
            self.assertTrue(failed['language_model_executed'])
            self.assertGreater(failed['q9_budget']['input_tokens'] + failed['q9_budget']['max_new_tokens'],
                               failed['q9_budget']['context_limit'])
            self.assertNotIn('q9_raw', failed)
            following = json.loads((out / 'Ego/plan.json').read_text())
            self.assertEqual(following['status'], 'invalid_q8')
            self.assertFalse(metadata['all_actions_parsed'])
            self.assertEqual(calls, [128, 256, 128])

    def test_short_history_fallback_never_counts_as_mtr_computation(self):
        from prediction import cmp_adapter as C
        from test_vehicle_tools import window
        from tools.vehicle import VehicleTools
        w = window()
        w['valid'][:, :-1] = False
        w['states'][:, :-1] = 0.
        w['scores'][:, :-1] = 0.
        service = VehicleTools(lambda: w, lambda value: C.predict(None, value), 'scene', 10, provider='peer')
        response = service.query('F')
        self.assertTrue(all(not obj['model_used'] for obj in json.loads(response['wire'])['objects']))
        self.assertEqual(response['cost']['model_targets_computed'], 0)
        self.assertEqual(response['cost']['fallback_targets_computed'], 2)


if __name__ == '__main__':
    unittest.main()
