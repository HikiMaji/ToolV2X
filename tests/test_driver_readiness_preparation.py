"""Resource-free checks for the frozen shared-driver preparation package."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'configs/structured_driver_readiness_v1'
SOURCE = ROOT / '.superpowers/sdd/2026-09-15-driver-readiness/source_causal_frames.jsonl'
HELPER = ROOT / 'scripts/prepare_driver_readiness.py'


def module():
    spec = importlib.util.spec_from_file_location('prepare_driver_readiness', HELPER)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


class DriverReadinessPreparationTests(unittest.TestCase):
    def copy_package(self):
        temporary = tempfile.TemporaryDirectory()
        target = Path(temporary.name) / PACKAGE.name
        shutil.copytree(PACKAGE, target)
        return temporary, target

    def test_frozen_package_validates_actual_configs_selection_and_budget(self):
        result = module().validate_package(PACKAGE, source_frames=SOURCE)
        self.assertEqual(result['frame_counts'], {'train': 2987, 'validation': 108})
        self.assertEqual(result['acceptance_counts'], {'train': 16, 'validation': 4})
        self.assertEqual(result['one_shot']['candidate_ids'],
                         ['initial', 'slower', 'constant_motion'])
        self.assertLessEqual(result['one_shot']['request_bytes'], 16384)
        self.assertLessEqual(result['one_shot']['request_bytes'] +
                             result['one_shot']['outer_response_cap'], 196608)

    def test_frame_contract_rejects_role_forbidden_field_and_source_drift(self):
        for mutation in ('role', 'label', 'source'):
            temporary, target = self.copy_package()
            with temporary:
                path = target / 'frames.jsonl'
                rows = path.read_text().splitlines()
                first = json.loads(rows[0])
                if mutation == 'role':
                    first['role'] = 'test'
                elif mutation == 'label':
                    first['future_label'] = [[0., 0.]]
                else:
                    first['ego_motion']['speed_mps'] += 1.
                rows[0] = json.dumps(first)
                path.write_text('\n'.join(rows) + '\n')
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    module().validate_package(target, source_frames=SOURCE)

    def test_altered_acceptance_identity_fails(self):
        temporary, target = self.copy_package()
        with temporary:
            path = target / 'acceptance_frames.json'
            selected = json.loads(path.read_text())
            selected[0]['g'] += 1
            path.write_text(json.dumps(selected))
            with self.assertRaises(ValueError):
                module().validate_package(target)

    def test_portable_validation_keeps_frozen_population_count(self):
        temporary, target = self.copy_package()
        with temporary:
            frames = (target / 'frames.jsonl').read_text().splitlines()
            del frames[1]
            (target / 'frames.jsonl').write_text('\n'.join(frames) + '\n')
            source_path = target / 'source_metadata.json'
            source = json.loads(source_path.read_text())
            source['counts']['train'] -= 1
            source['existing_exclusion_counts']['train']['kept'] -= 1
            source_path.write_text(json.dumps(source))
            readiness_path = target / 'readiness.json'
            readiness_path.write_text(json.dumps(module().readiness(source)))
            with self.assertRaises(ValueError):
                module().validate_package(target)

    def test_portable_validation_rejects_each_stable_identity_collision(self):
        rows = module().read_jsonl(PACKAGE / 'frames.jsonl')
        selected = json.loads((PACKAGE / 'acceptance_frames.json').read_text())
        config = json.loads((PACKAGE / 'readiness.json').read_text())
        duplicate_sample = copy.deepcopy(rows)
        duplicate_sample[2] = copy.deepcopy(duplicate_sample[1])
        duplicate_sample[2]['g'] = rows[2]['g']
        same_scene_g = copy.deepcopy(rows)
        same_scene_g[2]['g'] = same_scene_g[1]['g']
        for name, changed in (('duplicate_sample_changed_g', duplicate_sample),
                              ('same_scene_g_alias', same_scene_g)):
            with self.subTest(name=name), self.assertRaises(ValueError):
                module().validate_frames(changed, selected, config)

    def test_public_validate_only_cli_needs_no_private_preparation_inputs(self):
        commands = (
            [sys.executable, str(HELPER), str(PACKAGE), '--validate-only'],
            [sys.executable, str(HELPER), str(PACKAGE), '--validate-only',
             '--source-frames', str(SOURCE)],
        )
        for command, compared in zip(commands, (False, True)):
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            with self.subTest(source_comparison=compared):
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout)['source_comparison'], compared)

    def test_driver_mode_and_actual_budget_mutations_fail(self):
        for filename, mutate in (
            ('training_v2.json', lambda value: value['driver_spec'].__setitem__('p_processing', 'local_mtr')),
            ('readiness.json', lambda value: value['execution_spec'].__setitem__('max_episode_bytes', 1000)),
        ):
            temporary, target = self.copy_package()
            with temporary:
                path = target / filename
                value = json.loads(path.read_text())
                mutate(value)
                path.write_text(json.dumps(value))
                with self.subTest(filename=filename), self.assertRaises(ValueError):
                    module().validate_package(target)

    def test_bootstrap_is_expected_only_and_cannot_certify_eligible_labels(self):
        bootstrap = json.loads((PACKAGE / 'bootstrap_training_expected_v2.json').read_text())
        readiness = json.loads((PACKAGE / 'readiness.json').read_text())
        self.assertEqual((bootstrap['epochs'], bootstrap['validation']['interval_batches']), (3, 8))
        self.assertEqual(readiness['bootstrap']['expected_training_rows'], 16)
        self.assertFalse(readiness['bootstrap']['eligible_label_count_certified'])
        self.assertEqual(readiness['bootstrap']['runtime_interval_rule'],
                         'ceil(eligible_train_rows/2)')
        self.assertEqual(readiness['lifecycle'], 'prepared_not_executed')
        self.assertIsNone(readiness['runtime_load_identity'])


if __name__ == '__main__':
    unittest.main()
