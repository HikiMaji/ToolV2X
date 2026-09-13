"""T9 run-entry freezing contracts; no real model or recording execution."""
import builtins
import copy
import json
from pathlib import Path
from types import ModuleType
import tempfile
import sys
import unittest
from unittest.mock import patch

from common.audit_protocol import recording
from planning.method_controls import control_spec
from test_method_episode import EvidenceDriver, limits
from test_query_data import utility
from test_task_spec import provenance


class FakePolicy:
    def __init__(self, saved):
        self.kind = saved['kind']
        self.config = copy.deepcopy(saved['config'])
        self.provenance = copy.deepcopy(saved['provenance'])
        self.calls = 0

    def validate_runtime(self, **metadata):
        json.dumps(metadata, allow_nan=False)

    def __call__(self, state):
        self.calls += 1
        return dict(tool='STOP', mode=None, reason='frozen_fixture_value')


class MethodRunSpecTests(unittest.TestCase):
    def predictor_identity(self):
        return dict(version='toolv2x_predictor_load_identity_v1',
            checkpoint_metadata=dict(epoch=5, it=5196, version='fixture_mtr_v1'),
            model_class='CMP MotionTransformer', state_items_loaded=880,
            missing_keys=[], unexpected_keys=[], parameter_count=68514524)

    def mtr_provenance(self, path='/machine/a/best_model.pth', **changes):
        value=dict(checkpoint=path, config='/machine/a/config.yaml', source='/machine/a/model.py',
            device='cuda', checkpoint_metadata=dict(epoch=5, it=5196, version='fixture_mtr_v1'),
            model_class='CMP MotionTransformer', state_items_loaded=880,
            missing_keys=[], unexpected_keys=[], parameter_count=68514524)
        value.update(changes)
        return value

    def rows(self):
        scene = 'testoutput_CAV_data_2022-03-15-10-09-50_0'
        return [dict(sample_id='sample0', scene=scene, g=10, local_frame=10,
            role='validation', physical_split='train', ego_motion=dict(speed_mps=2., yaw_rate_rps=0.),
            feature_read_paths=['feature'], motion_read_paths=['motion'])]

    def spec(self, rows=None):
        rows = rows or self.rows()
        group = recording(rows[0]['scene'])
        driver = EvidenceDriver([]).provenance
        predictor = dict(binding_id='run-local-id', model_version=provenance()['prediction'],
            settings=dict(predictor_load_identity=self.predictor_identity()))
        return dict(version='toolv2x_method_run_spec_v1', limits=limits(policy_id='method_value_v1'),
            local_provenance=provenance(), control=control_spec('feedback'), utility_spec=utility(),
            recording_folds={group:'fold0'}, recording_roles={group:'validation'},
            runtime_binding=dict(driver=driver, predictor=predictor),
            query_spec=dict(state_version='toolv2x_query_state_v1',
                feature_version='toolv2x_query_features_v1', action_version='toolv2x_value_actions_v1'))

    def binding(self, spec):
        from planning.query_data import _semantic_binding
        return _semantic_binding(spec['runtime_binding'], spec)

    def saved(self, spec, kind, marker):
        policy_id = (spec['control']['bundle']['continuation_policy_id']
            if kind == 'bundle_terminal' and spec['control']['name'] == 'one_shot'
            else spec['limits']['policy_id'])
        return dict(version='fixture_checkpoint_v1', kind=kind, marker=marker,
            weights=[1., 2., marker], config=dict(binding=self.binding(spec),
                utility_spec=spec['utility_spec'], policy_id=policy_id),
            provenance=dict(version='fixture_value_source_v1', policy_kind=kind, marker=marker))

    def one_shot_spec(self):
        spec = self.spec()
        spec['control'] = control_spec('one_shot', bundle=dict(
            continuation_policy_id='bundle_value_v1', candidate_sources=['initial'],
            slowdown_scale=.5, max_candidates=1, wrapper_reserve_bytes=512,
            extra_generation=None))
        return spec

    @staticmethod
    def loader(path, *, expected_binding=None, policy_kind=None):
        saved = json.loads(Path(path).read_text())
        if saved['config']['binding'] != expected_binding:
            raise ValueError('binding drift')
        if saved['kind'] != policy_kind:
            raise ValueError('wrong policy kind')
        return FakePolicy(saved)

    @staticmethod
    def reader(path):
        return json.loads(Path(path).read_text())

    def test_freeze_is_json_only_and_records_exact_sample_role_and_fold(self):
        from planning.method_run_spec import freeze_method_run_spec
        rows = self.rows(); spec = self.spec(rows)
        spec['runtime_binding']['driver']['checkpoint'] = '/machine/local/driver'
        original_import = builtins.__import__
        def no_torch(name, *args, **kwargs):
            if name == 'torch' or name.startswith('torch.'):
                raise AssertionError('pure run specification imported torch')
            return original_import(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=no_torch):
            frozen = freeze_method_run_spec(spec, rows)
        self.assertEqual(frozen['samples'], [dict(sample_id='sample0', scene=rows[0]['scene'], g=10,
            local_frame=10, physical_recording=recording(rows[0]['scene']), role='validation', fold='fold0')])
        self.assertEqual(frozen['value_checkpoints'], {})
        self.assertNotIn('/machine/local/driver', json.dumps(frozen))
        self.assertNotIn('run-local-id', json.dumps(frozen))
        json.dumps(frozen, allow_nan=False)

        bad = copy.deepcopy(spec); bad.pop('utility_spec')
        with self.assertRaises(ValueError): freeze_method_run_spec(bad, rows)
        bad = copy.deepcopy(spec); bad['recording_roles'][recording(rows[0]['scene'])] = 'train'
        with self.assertRaises(ValueError): freeze_method_run_spec(bad, rows)
        duplicate = [rows[0], dict(rows[0], scene=rows[0]['scene'].replace('_0', '_1'), g=11, local_frame=11)]
        with self.assertRaisesRegex(ValueError, 'sample'):
            freeze_method_run_spec(spec, duplicate)
        bad = copy.deepcopy(spec); bad['recording_roles']['another_recording'] = 'train'
        with self.assertRaisesRegex(ValueError, 'roles|folds'):
            freeze_method_run_spec(bad, rows)

    def test_predictor_load_identity_uses_checkpoint_metadata_and_ignores_paths(self):
        from planning.method_run_spec import predictor_load_identity
        original_import = builtins.__import__
        def no_torch(name, *args, **kwargs):
            if name == 'torch' or name.startswith('torch.'):
                raise AssertionError('predictor identity imported torch')
            return original_import(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=no_torch):
            left = predictor_load_identity(self.mtr_provenance('/machine/a/model.pth'))
            relocated_source = self.mtr_provenance('/machine/b/model.pth')
            relocated_source.update(config='/machine/b/config.yaml', source='/machine/b/model.py')
            relocated = predictor_load_identity(relocated_source)
            changed = predictor_load_identity(self.mtr_provenance('/machine/a/model.pth',
                checkpoint_metadata=dict(epoch=6, it=6200, version='fixture_mtr_v1')))
        self.assertEqual(left, self.predictor_identity())
        self.assertEqual(left, relocated)
        self.assertNotEqual(left, changed)
        self.assertNotIn('/machine/', json.dumps(left))

    def test_snapshot_loads_only_role_appropriate_kinds_and_saves_provenance(self):
        from planning.method_run_spec import prepare_value_checkpoints
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); spec = self.one_shot_spec(); frozen = __import__(
                'planning.method_run_spec', fromlist=['freeze_method_run_spec']).freeze_method_run_spec(spec, self.rows())
            request = root/'request.pt'; bundle = root/'bundle.pt'
            request.write_text(json.dumps(self.saved(spec, 'shared', 1)))
            bundle.write_text(json.dumps(self.saved(spec, 'bundle_terminal', 2)))
            loaded_paths=[];read_paths=[]
            def load_snapshot(path,**kwargs):
                loaded_paths.append(Path(path));return self.loader(path,**kwargs)
            def read_snapshot(path):
                read_paths.append(Path(path));return self.reader(path)
            manifest, policies = prepare_value_checkpoints(
                dict(request=request, bundle=bundle), root/'run', self.binding(spec), spec['utility_spec'],
                policy_loader=load_snapshot, checkpoint_reader=read_snapshot)
            frozen['value_checkpoints'] = manifest
            self.assertEqual(set(policies), {'request', 'bundle'})
            self.assertEqual(manifest['request']['policy_kind'], 'shared')
            self.assertEqual(manifest['bundle']['policy_kind'], 'bundle_terminal')
            self.assertEqual(manifest['request']['provenance'], policies['request'].provenance)
            self.assertFalse(Path(manifest['request']['snapshot']).is_absolute())
            self.assertEqual(self.reader(root/'run'/manifest['request']['snapshot'])['weights'], [1., 2., 1])
            self.assertEqual(loaded_paths, [root/'run/value_checkpoints/request.pt', root/'run/value_checkpoints/bundle.pt'])
            self.assertEqual(read_paths, loaded_paths)
            wrong = root/'wrong.pt'; wrong.write_text(json.dumps(self.saved(spec, 'terminal', 3)))
            with self.assertRaisesRegex(ValueError, 'kind'):
                prepare_value_checkpoints(dict(request=wrong), root/'bad', self.binding(spec), spec['utility_spec'],
                    policy_loader=self.loader, checkpoint_reader=self.reader)
            ordinary = self.spec(); unsupported = root/'unsupported.pt'
            unsupported.write_text(json.dumps(self.saved(ordinary, 'bundle_terminal', 4)))
            with self.assertRaisesRegex(ValueError, 'one_shot'):
                prepare_value_checkpoints(dict(bundle=unsupported), root/'unsupported', self.binding(ordinary),
                    ordinary['utility_spec'], policy_loader=self.loader, checkpoint_reader=self.reader)

    def test_resume_rejects_replaced_checkpoint_at_the_same_path(self):
        from planning.method_run_spec import prepare_value_checkpoints
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); spec = self.spec(); selected = root/'selected.pt'
            selected.write_text(json.dumps(self.saved(spec, 'shared', 1)))
            manifest, _ = prepare_value_checkpoints(dict(request=selected), root/'first', self.binding(spec),
                spec['utility_spec'], policy_loader=self.loader, checkpoint_reader=self.reader)
            selected.write_text(json.dumps(self.saved(spec, 'shared', 9)))
            with self.assertRaisesRegex(ValueError, 'changed'):
                prepare_value_checkpoints(dict(request=selected), root/'second', self.binding(spec),
                    spec['utility_spec'], resume_from=root/'first', previous_manifest=manifest,
                    policy_loader=self.loader, checkpoint_reader=self.reader)
            selected.write_text(json.dumps(self.saved(spec, 'shared', 1)))
            escaped = copy.deepcopy(manifest); escaped['request']['snapshot'] = str(selected)
            with self.assertRaisesRegex(ValueError, 'snapshot'):
                prepare_value_checkpoints(dict(request=selected), root/'third', self.binding(spec),
                    spec['utility_spec'], resume_from=root/'first', previous_manifest=escaped,
                    policy_loader=self.loader, checkpoint_reader=self.reader)

    def test_runtime_binding_is_checked_before_original_models_are_prepared(self):
        from planning.method_run_spec import validate_runtime_binding
        spec = self.spec(); wrong = copy.deepcopy(spec['runtime_binding'])
        wrong['predictor']['model_version']['revision'] = 'changed'
        touched = []
        def prepare_models():
            touched.append(True)
        with self.assertRaisesRegex(ValueError, 'runtime binding'):
            validate_runtime_binding(wrong, spec)
        wrong = copy.deepcopy(spec['runtime_binding'])
        wrong['predictor']['settings']['predictor_load_identity']['checkpoint_metadata']['epoch'] = 6
        with self.assertRaisesRegex(ValueError, 'runtime binding'):
            validate_runtime_binding(wrong, spec)
        self.assertEqual(touched, [])
        self.assertEqual(validate_runtime_binding(spec['runtime_binding'], spec), self.binding(spec))

    def test_interact_uses_explicit_shared_checkpoint_and_freezes_it_before_runtime(self):
        from planning import run_framework as runner
        from planning.context import build_task_plan_input
        from test_method_episode import InteractionRunnerTests
        helper = InteractionRunnerTests(); helper.setUp()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); data, rows, spec_path = helper.make_data(root)
            runtime = helper.runtime(rows)
            from tools.task_spec import FrozenPredictor
            runtime.predictor = FrozenPredictor(helper.predictor,
                provenance()['prediction'],dict(predictor_load_identity=self.predictor_identity()))
            spec = self.spec(rows)
            spec['runtime_binding'] = dict(driver=runtime.driver.provenance,
                predictor=runtime.predictor.descriptor)
            spec_path.write_text(json.dumps(spec))
            selected = root/'selected.pt'; selected.write_text(json.dumps(self.saved(spec, 'shared', 1)))
            loaded = []
            def loader(path, **kwargs):
                policy = self.loader(path, **kwargs); loaded.append(policy); return policy
            def build(*args, **kwargs):
                kwargs['token_counter'] = lambda prompt: len(prompt)//4
                return build_task_plan_input(*args, **kwargs)
            with patch('planning.query_value.load_query_policy', side_effect=loader), \
                    patch('planning.query_value._torch_load', side_effect=self.reader), \
                    patch.object(runner, '_load_interaction_runtime', return_value=runtime) as prepare, \
                    patch('planning.method_episode.build_task_plan_input', side_effect=build):
                runner.interact(data, root/'run', Path('/fixture/driver'), spec_path,
                    per_recording=0, value_checkpoint=selected)
            self.assertEqual(prepare.call_count, 1)
            self.assertEqual(loaded[0].calls, len(rows))
            frozen = json.loads((root/'run/run_spec.json').read_text())
            self.assertEqual(frozen['value_checkpoints']['request']['provenance'], loaded[0].provenance)
            self.assertTrue((root/'run'/frozen['value_checkpoints']['request']['snapshot']).is_file())
            task = json.loads((root/'run/sample_000000/task.json').read_text())
            self.assertEqual(task['episode']['stop_reason'], 'frozen_fixture_value')
            self.assertEqual(task['episode']['value_policy'], loaded[0].provenance)

    def test_bad_selected_checkpoint_fails_before_runtime_or_output_creation(self):
        from planning import run_framework as runner
        from test_method_episode import InteractionRunnerTests
        helper = InteractionRunnerTests(); helper.setUp()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); data, rows, spec_path = helper.make_data(root)
            runtime = helper.runtime(rows); spec = self.spec(rows)
            spec['runtime_binding']['driver'] = runtime.driver.provenance
            spec_path.write_text(json.dumps(spec))
            selected = root/'selected.pt'; selected.write_text('{}')
            with patch('planning.query_value.load_query_policy', side_effect=ValueError('bad checkpoint')), \
                    patch.object(runner, '_load_interaction_runtime') as prepare:
                with self.assertRaisesRegex(ValueError, 'bad checkpoint'):
                    runner.interact(data, root/'run', Path('/fixture/driver'), spec_path,
                        per_recording=0, value_checkpoint=selected)
            prepare.assert_not_called()
            self.assertFalse((root/'run').exists())

    def test_bundle_collection_entry_passes_original_runtime_factory_to_exact_collector_api(self):
        from planning import run_framework as runner
        from test_method_episode import InteractionRunnerTests
        helper = InteractionRunnerTests(); helper.setUp()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); data, rows, spec_path = helper.make_data(root)
            raw = json.loads(spec_path.read_text())
            raw['control'] = control_spec('one_shot', bundle=dict(
                continuation_policy_id='diagnostic_conditional_v1',
                candidate_sources=['initial'], slowdown_scale=.5, max_candidates=1,
                wrapper_reserve_bytes=512, extra_generation=None))
            raw['version'] = 'toolv2x_bundle_collection_v1'
            raw['recording_folds'] = {recording(rows[0]['scene']):'fold0'}
            spec_path.write_text(json.dumps(raw))
            calls = []
            module = ModuleType('planning.bundle_data')
            def collect_bundle_branches(online_index, out, runtime, controls_spec):
                calls.append((online_index, out, runtime, controls_spec))
                return dict(status='synthetic_contract_only')
            module.collect_bundle_branches = collect_bundle_branches
            with patch.dict(sys.modules, {'planning.bundle_data':module}):
                result = runner.collect_bundle_method(data, root/'bundle', Path('/fixture/driver'),
                    spec_path, role='validation', per_recording=0)
            self.assertEqual(result, dict(status='synthetic_contract_only'))
            self.assertEqual(calls[0][0], rows)
            self.assertEqual(calls[0][1], root/'bundle')
            self.assertTrue(callable(calls[0][2]))
            self.assertEqual(calls[0][3], raw)

    def test_bundle_only_run_rejects_non_diagnostic_outer_policy_before_runtime(self):
        from planning import run_framework as runner
        from test_method_episode import InteractionRunnerTests
        helper = InteractionRunnerTests(); helper.setUp()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); data, rows, spec_path = helper.make_data(root)
            runtime = helper.runtime(rows); spec = self.one_shot_spec()
            spec['runtime_binding']['driver'] = runtime.driver.provenance
            spec_path.write_text(json.dumps(spec))
            bundle = root/'bundle.pt'; bundle.write_text(json.dumps(self.saved(spec, 'bundle_terminal', 1)))
            with patch.object(runner, '_load_interaction_runtime') as prepare:
                with self.assertRaisesRegex(ValueError, 'diagnostic'):
                    runner.interact(data, root/'run', Path('/fixture/driver'), spec_path,
                        per_recording=0, bundle_value_checkpoint=bundle)
            prepare.assert_not_called()
            self.assertFalse((root/'run').exists())

    def test_resume_rejects_non_python_mtr_config_change_before_runtime(self):
        from planning import run_framework as runner
        from planning.context import build_task_plan_input
        from test_method_episode import InteractionRunnerTests
        helper = InteractionRunnerTests(); helper.setUp()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); data, rows, spec_path = helper.make_data(root)
            def build(*args, **kwargs):
                kwargs['token_counter'] = lambda prompt: len(prompt)//4
                return build_task_plan_input(*args, **kwargs)
            with patch.object(runner, '_load_interaction_runtime', return_value=helper.runtime(rows)), \
                    patch('planning.method_episode.build_task_plan_input', side_effect=build):
                runner.interact(data, root/'old', Path('/fixture/driver'), spec_path, per_recording=0)
            yaml = root/'old/code_snapshot/vendor/cmp_mtr/configs/v2v4real_multiego_no_coop.yaml'
            yaml.write_text(yaml.read_text() + '\n# changed after execution\n')
            with patch.object(runner, '_load_interaction_runtime') as prepare:
                with self.assertRaisesRegex(ValueError, 'source changed'):
                    runner.interact(data, root/'new', Path('/fixture/driver'), spec_path,
                        per_recording=0, resume_from=root/'old')
            prepare.assert_not_called()
            self.assertFalse((root/'new').exists())


if __name__ == '__main__':
    unittest.main()
