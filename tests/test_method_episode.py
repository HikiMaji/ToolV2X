"""T4 execution contracts; synthetic driver/predictor, actual service and receiver."""
import builtins
import copy
import importlib
import io
from types import SimpleNamespace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from test_evidence_ledger import LedgerFixture
from test_vehicle_tools import window
from test_task_spec import provenance, spec_dict


def limits(**changes):
    value = dict(version='toolv2x_interaction_v1', max_calls=2, policy_id='fixture_feedback',
        driver_version=dict(name='evidence_dependent_fixture', revision='v1'),
        execution_spec=spec_dict(max_request_bytes=30000, max_response_bytes=30000,
                                 max_episode_bytes=100000),
        receiver_spec=dict(version='toolv2x_receiver_v1', context_limit=12000,
                           generation_reserve=256, peer_reserve=6000, numeric_decimal_places=2))
    value.update(changes)
    return value


class EvidenceDriver:
    """Test-only external-model stand-in; result depends on actual admitted objects."""
    tokenizer = None
    context_limit = 12000
    provenance = dict(model_class='synthetic_contract_driver', decoding='greedy')

    def __init__(self, events, invalid_stage=None, raise_stage=None):
        self.events, self.inputs = events, []
        self.invalid_stage, self.raise_stage = invalid_stage, raise_stage

    def plan_prepared(self, features, prepared):
        stage = len(self.inputs)
        self.events.append('driver:%d' % stage)
        self.inputs.append(copy.deepcopy(prepared))
        if stage == self.raise_stage:
            raise RuntimeError('fixture generation interrupted')
        remote = prepared['remote_evidence_used']
        # Only a near target changes the generated plan; far/empty paid results do not.
        changed = remote is not None and any(o['box'][0] < 30 for o in remote['objects'])
        tau = [[float(i + 1), .1 * (i + 1) if changed else 0.] for i in range(6)]
        raw = 'The suggested trajectory is: ' + repr(tau)
        if stage == self.invalid_stage:
            raw = 'malformed original model answer'
        return dict(prepared, status='invalid_q9' if stage == self.invalid_stage else 'parsed',
                    q9_raw=raw, waypoints=tau, q9_executed=True, language_model_executed=True,
                    q9_cost=dict(seconds=float(stage + 1), input_tokens=prepared['evidence_selection']['input_tokens'],
                                 output_tokens=36, feature_tokens=270))


def feedback(state):
    if state['remaining_budget']['calls'] == 2:
        return dict(tool='P', mode='current', reason='inspect')
    changed = state['previous_plan']['waypoints'] != state['current_plan']['waypoints']
    return dict(tool='F' if changed else 'STOP', mode='change' if changed else None,
                reason='actual_revision' if changed else 'unchanged')


class MethodEpisodeTests(LedgerFixture):
    def runner(self):
        self.assertTrue((ROOT / 'src/planning/method_episode.py').exists(), 'T4 episode runner missing')
        return importlib.import_module('planning.method_episode').run_task_episode

    def execute(self, peer=None, policy=feedback, config=None, progress=None, driver=None, service=None):
        self.events = [] if driver is None else driver.events
        self.driver = driver or EvidenceDriver(self.events)
        self.service_used = service or self.service(peer)
        actual = self.service_used.query_task
        def query(request):
            self.events.append('query:%s_%s' % (request['tool'], request['mode']))
            return actual(request)
        self.service_used.query_task = query
        ego = window('ego')
        return self.runner()(ego, self.predictor(ego), dict(speed_mps=2., yaw_rate_rps=0.),
            dict(active_agent_mask=np.ones((1, 2, 1), dtype=bool) & np.array([[[True], [False]]])),
            self.service_used, self.predictor, self.driver, policy, config or limits(),
            on_progress=progress, local_provenance=provenance(), sample_id='sample', branch_id='feedback',
            token_counter=lambda prompt: len(prompt) // 4)

    def test_fixed_diagnostic_ids_schedule_only_current_receipt_count(self):
        from planning.method_episode import DIAGNOSTIC_POLICY_IDS, diagnostic_policy

        self.assertEqual(DIAGNOSTIC_POLICY_IDS,
            ('stop', 'p_current', 'p_current_f_change', 'f_current', 'p_current_f_current'))

        class NoForecastPreview(dict):
            def __getitem__(self, key):
                if 'forecast' in key:
                    raise AssertionError('diagnostic selector read a forecast preview')
                return super().__getitem__(key)

        actions = [dict(tool='STOP', mode=None), dict(tool='P', mode='current'),
                   dict(tool='F', mode='current'), dict(tool='F', mode='change')]
        expected = {
            'stop': [('STOP', None), ('STOP', None), ('STOP', None)],
            'p_current': [('P', 'current'), ('STOP', None), ('STOP', None)],
            'f_current': [('F', 'current'), ('STOP', None), ('STOP', None)],
            'p_current_f_current': [('P', 'current'), ('F', 'current'), ('STOP', None)],
        }
        for policy_id, schedule in expected.items():
            for receipts, wanted in enumerate(schedule):
                state = NoForecastPreview(policy_id=policy_id,
                    response_receipts=[dict(receipt_id='r%d' % i) for i in range(receipts)],
                    previous_plan=dict(waypoints=[[99., 99.]] * 6) if receipts == 0 else None,
                    available_actions=actions)
                decision = diagnostic_policy(state)
                with self.subTest(policy_id=policy_id, receipts=receipts):
                    self.assertEqual((decision['tool'], decision['mode']), wanted)
                    self.assertIn(dict(tool=decision['tool'], mode=decision['mode']), actions)

        changed = NoForecastPreview(policy_id='p_current_f_change', response_receipts=[{}],
            previous_plan=dict(waypoints=[[1., 0.]] * 6), available_actions=actions)
        self.assertEqual(diagnostic_policy(changed)['mode'], 'change')
        unavailable = NoForecastPreview(policy_id='f_current', response_receipts=[],
            previous_plan=None, available_actions=[dict(tool='STOP', mode=None),
                                                   dict(tool='P', mode='current')])
        self.assertEqual(diagnostic_policy(unavailable)['tool'], 'STOP')

    def test_actual_response_drives_revision_then_second_change_request(self):
        result = self.execute()
        self.assertEqual(self.events, ['driver:0', 'query:P_current', 'driver:1', 'query:F_change', 'driver:2'])
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['events'][-1]['kind'], 'STOP')
        self.assertEqual(result['requests'][1]['tau_old'], result['plans'][0]['output']['waypoints'])
        self.assertEqual(result['requests'][1]['tau_new'], result['plans'][1]['output']['waypoints'])
        self.assertTrue(result['requests'][1]['acquired_field_manifest'])
        self.assertEqual(result['final_plan_id'], result['plans'][2]['plan_id'])
        self.assertEqual(result['cost']['driver_attempts'], 3)
        self.assertEqual(result['cost']['generation_seconds'], 6.)
        self.assertEqual(len(result['ledger_snapshots'][-1]['acquired_fields']), 6)
        self.assertEqual(result['limits'], limits())
        json.dumps(result, allow_nan=False)

    def test_change_first_response_changes_second_action_to_stop(self):
        peer = window()
        peer['states'][:, :, 0] = 100.
        result = self.execute(peer)
        self.assertEqual(self.events, ['driver:0', 'query:P_current', 'driver:1'])
        self.assertEqual(result['stop_reason'], 'unchanged')
        self.assertEqual(result['plans'][0]['output']['waypoints'], result['plans'][1]['output']['waypoints'])
        self.assertEqual(result['final_plan_id'], result['plans'][1]['plan_id'])

    def test_zero_one_two_caps_and_stop_never_add_an_extra_generation(self):
        for cap in (0, 1, 2):
            with self.subTest(cap=cap):
                result = self.execute(policy=lambda s: dict(tool='P', mode='current', reason='repeat'),
                                      config=limits(max_calls=cap))
                self.assertEqual(len(result['plans']), cap + 1)
                self.assertEqual(len(result['requests']), cap)
                self.assertEqual(len(self.service_used.task_records), cap)
        result = self.execute(policy=lambda s: dict(tool='STOP', mode=None, reason='enough'))
        self.assertEqual(len(result['plans']), 1)
        self.assertEqual(self.service_used.task_records, [])

    def test_invalid_initial_or_intermediate_answer_preserves_raw_and_stops(self):
        for stage in (0, 1):
            with self.subTest(stage=stage):
                driver = EvidenceDriver([], invalid_stage=stage)
                result = self.execute(driver=driver)
                self.assertEqual(result['status'], 'invalid_plan')
                self.assertEqual(len(self.service_used.task_records), stage)
                self.assertIsNone(result['final_plan_id'])
                self.assertEqual(result['plans'][-1]['output']['q9_raw'], 'malformed original model answer')
                self.assertEqual(result['cost']['driver_attempts'], stage + 1)
                if stage:
                    self.assertGreater(result['cost']['response_bytes'], 0)

    def test_parsed_but_abnormal_path_does_not_become_task_parameters(self):
        driver = EvidenceDriver([])
        actual = driver.plan_prepared
        def abnormal(features, prepared):
            out = actual(features, prepared)
            out['waypoints'] = [[10000., 0.]] * 6
            out['q9_raw'] = 'The suggested trajectory is: ' + repr(out['waypoints'])
            return out
        driver.plan_prepared = abnormal
        result = self.execute(driver=driver)
        self.assertEqual(result['status'], 'invalid_plan')
        self.assertEqual(self.service_used.task_records, [])

    def test_policy_gets_only_visible_snapshot_and_cannot_choose_a_path_or_manifest(self):
        seen = []
        def policy(state):
            seen.append(copy.deepcopy(state))
            state['current_plan']['waypoints'][0][0] = 12345.
            if len(seen) == 1:
                return dict(tool='P', mode='current', reason='inspect')
            return dict(tool='STOP', mode=None, reason='done')
        result = self.execute(policy=policy)
        self.assertEqual(result['requests'][0]['tau_new'][0], [1., 0.])
        self.assertEqual(seen[0]['acquired_fields'], [])
        self.assertEqual(seen[0]['response_receipts'], [])
        self.assertFalse({'service', 'features', 'offline_labels', 'tree', 'window_loader'} & set(seen[0]))
        for extra in ('tau_new', 'acquired_field_manifest'):
            with self.subTest(extra=extra):
                bad = self.execute(policy=lambda s: dict(tool='P', mode='current', reason='bad', **{extra: []}))
                self.assertEqual(bad['status'], 'policy_error')
                self.assertEqual(self.service_used.task_records, [])

    def test_first_change_and_unchanged_change_are_illegal(self):
        result = self.execute(policy=lambda s: dict(tool='F', mode='change', reason='no_old_plan'))
        self.assertEqual(result['status'], 'policy_error')
        self.assertEqual(self.service_used.task_records, [])
        peer = window(); peer['states'][:, :, 0] = 100.
        result = self.execute(peer, policy=lambda s: dict(tool='P' if s['previous_plan'] is None else 'F',
                    mode='current' if s['previous_plan'] is None else 'change', reason='ask'))
        self.assertEqual(result['status'], 'policy_error')
        self.assertEqual(len(self.service_used.task_records), 1)

    def test_budget_preflight_uses_full_manifest_and_never_reads_unpaid_peer(self):
        config = limits()
        config['execution_spec']['max_episode_bytes'] = 1
        result = self.execute(config=config)
        self.assertEqual(result['stop_reason'], 'byte_budget_exhausted')
        self.assertEqual(self.service_used.task_records, [])
        self.assertEqual(len(result['plans']), 1)
        config = limits(); config['execution_spec']['max_request_bytes'] = 1
        result = self.execute(config=config)
        self.assertEqual(result['stop_reason'], 'request_byte_limit')
        self.assertEqual(self.service_used.task_records, [])

    def test_service_and_driver_failures_keep_spent_costs_without_fake_completion(self):
        def broken():
            raise RuntimeError('failed private loading')
        from tools.vehicle import VehicleTools
        service = VehicleTools(broken, self.predictor, 'scene', 10, 'peer', task_provenance=provenance())
        result = self.execute(service=service)
        self.assertEqual(result['status'], 'service_error')
        self.assertGreater(result['cost']['request_bytes'], 0)
        self.assertFalse(result['cost']['complete'])
        driver = EvidenceDriver([], raise_stage=1)
        result = self.execute(driver=driver)
        self.assertEqual(result['status'], 'driver_error')
        self.assertEqual(len(self.service_used.task_records), 1)
        self.assertGreater(result['cost']['response_bytes'], 0)
        self.assertEqual(result['cost']['driver_attempts'], 2)
        self.assertIsNone(result['cost']['generation_seconds'])

    def test_response_persisted_before_receiver_and_interruption_is_not_completion(self):
        saved = []
        def interrupt(snapshot):
            saved.append(snapshot)
            if snapshot['events'][-1]['kind'] == 'response_received':
                raise KeyboardInterrupt('simulated process interruption')
        with self.assertRaises(KeyboardInterrupt):
            self.execute(progress=interrupt)
        latest = saved[-1]
        self.assertEqual(latest['status'], 'running')
        self.assertEqual(len(latest['responses']), 1)
        self.assertEqual(len(latest['plans']), 1)
        self.assertEqual(latest['ledger_snapshots'][-1]['acquired_fields'], [])
        self.assertGreater(latest['cost']['response_bytes'], 0)
        self.assertEqual(saved[0]['responses'], [])


    def test_paid_response_and_e_survive_receiver_prediction_failure(self):
        actual = self.predictor._predict
        def broken(w):
            if w['source'] == 'peer':
                raise RuntimeError('receiver MTR failure')
            return actual(w)
        self.predictor._predict = broken
        result = self.execute()
        self.assertEqual(result['status'], 'receiver_error')
        self.assertEqual(len(result['responses']), 1)
        self.assertEqual(len(result['ledger_snapshots'][-1]['acquired_fields']), 4)
        self.assertEqual(result['ledger_snapshots'][-1]['derived_fields'], [])
        self.assertGreater(result['cost']['response_bytes'], 0)
        self.assertEqual(len(result['plans']), 1)

    def test_invalid_wire_keeps_raw_bytes_cost_and_never_becomes_a_revision(self):
        service = self.service()
        query = service.query_task
        def invalid(request):
            result = query(request)
            result['wire'] = b'broken response with paid bytes'
            result['cost']['response_bytes'] = len(result['wire'])
            return result
        service.query_task = invalid
        result = self.execute(service=service)
        self.assertEqual(result['status'], 'receiver_error')
        self.assertEqual(bytes.fromhex(result['responses'][0]['wire_hex']), b'broken response with paid bytes')
        self.assertEqual(result['ledger_snapshots'][-1]['acquired_fields'], [])
        self.assertEqual(len(result['plans']), 1)


    def test_missing_response_cost_preserves_bytes_and_fails_without_new_plan(self):
        service = self.service()
        query = service.query_task
        actual_wires = []
        def missing_cost(request):
            result = query(request)
            actual_wires.append(result['wire'])
            del result['cost']
            return result
        service.query_task = missing_cost
        result = self.execute(service=service)
        self.assertEqual(result['status'], 'receiver_error')
        self.assertEqual(bytes.fromhex(result['responses'][0]['wire_hex']), actual_wires[0])
        self.assertEqual(result['cost']['response_bytes'], len(actual_wires[0]))
        self.assertGreater(result['cost']['request_bytes'], 0)
        self.assertFalse(result['cost']['complete'])
        self.assertEqual(len(result['plans']), 1)

    def test_online_episode_does_not_open_future_labels(self):
        run = self.runner()
        original = builtins.open
        def guarded(path, *args, **kwargs):
            if 'offline_labels' in str(path):
                raise AssertionError('online execution opened labels')
            return original(path, *args, **kwargs)
        with patch('builtins.open', guarded):
            one = self.execute()
            two = self.execute()
        self.assertEqual(one['requests'][0]['tau_new'], two['requests'][0]['tau_new'])
        self.assertEqual(one['plans'][-1]['output']['waypoints'], two['plans'][-1]['output']['waypoints'])

    def test_same_instance_configuration_cannot_drift_between_revisions(self):
        driver = EvidenceDriver([])
        def mutate(state):
            driver.context_limit = 9999
            return dict(tool='P', mode='current', reason='changed_driver')
        result = self.execute(driver=driver, policy=mutate)
        self.assertEqual(result['status'], 'driver_error')
        self.assertEqual(len(result['plans']), 1)


class InteractionRunnerTests(LedgerFixture):
    def runtime(self, rows):
        driver = EvidenceDriver([])
        def load(row, directory):
            ego = window('ego')
            peer = window()
            for w in (ego, peer):
                w.update(scene=row['scene'], g=row['g'])
            return dict(local_window=ego, local_prediction=self.predictor(ego),
                motion=row['ego_motion'], features=dict(active_agent_mask=np.array([[[True], [False]]])),
                service=self.service_for(peer), metadata=dict(feature_read_paths=row['feature_read_paths']))
        return SimpleNamespace(driver=driver, predictor=self.predictor, load_inputs=load,
                               provenance=dict(kind='test_runtime_no_models'))

    def service_for(self, peer):
        from tools.vehicle import VehicleTools
        return VehicleTools(lambda: peer, self.predictor, peer['scene'], peer['g'], 'peer',
                            task_provenance=provenance())

    def make_data(self, root):
        data = root/'data'
        (data/'online_index').mkdir(parents=True)
        (data/'offline_labels').mkdir()
        (data/'offline_labels/validation.jsonl').write_text('future A')
        (data/'config.json').write_text(json.dumps(dict(mtr_checkpoint='/fixture/model')))
        scene = 'testoutput_CAV_data_2022-03-15-10-09-50_0'
        rows = [dict(sample_id='sample%d'%i, role='validation', physical_split='train', scene=scene,
            local_frame=10+i, g=10+i, ego_motion=dict(speed_mps=2., yaw_rate_rps=0.),
            feature_read_paths=['features%d'%i], motion_read_paths=['motion%d'%i]) for i in range(2)]
        (data/'online_index/validation.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        (data/'online_index/train.jsonl').write_text('')
        config = dict(limits=limits(policy_id='p_current_f_change'), local_provenance=provenance())
        spec = root/'spec.json'; spec.write_text(json.dumps(config))
        return data, rows, spec

    def test_interact_forbids_labels_uses_one_driver_and_archives_real_boundaries(self):
        from planning import run_framework as runner
        self.assertTrue(hasattr(runner, 'interact'), 'T4 interact entry missing')
        from planning.context import build_task_plan_input
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, rows, spec = self.make_data(root)
            raw_open = io.open
            def guarded(path, *a, **kw):
                if 'offline_labels' in str(path):
                    raise AssertionError('online runner opened future labels')
                return raw_open(path, *a, **kw)
            prompts = []
            for n in (0, 1):
                (data/'offline_labels/validation.jsonl').write_text('changed future %d'%n)
                runtime = self.runtime(rows)
                def build(*a, **kw):
                    kw['token_counter'] = lambda p: len(p)//4
                    return build_task_plan_input(*a, **kw)
                with patch.object(runner, '_load_interaction_runtime', return_value=runtime) as load, \
                     patch('planning.method_episode.build_task_plan_input', side_effect=build), \
                     patch('io.open', guarded), patch('builtins.open', guarded):
                    runner.interact(data, root/('out%d'%n), Path('/fixture/driver'), spec, per_recording=0)
                self.assertEqual(load.call_count, 1)
                out = root/('out%d'%n)
                progress = json.loads((out/'progress.json').read_text())
                self.assertEqual(progress['status'], 'completed')
                self.assertEqual(progress['terminal_tasks'], 2)
                records = runner.read_jsonl(out/'tasks.jsonl')
                tasks = [json.loads((out/r['path']).read_text()) for r in records]
                self.assertTrue(all(t['episode']['status'] == 'completed' for t in tasks))
                first = tasks[0]['episode']
                self.assertEqual(len(first['plans']), 3)
                self.assertEqual(first['events'][0]['kind'], 'episode_started')
                self.assertEqual(first['events'][-1]['kind'], 'STOP')
                prompts.append([p['prepared']['q9_prompt'] for p in first['plans']])
            self.assertEqual(prompts[0], prompts[1])

    def test_resume_reuses_terminal_prefix_including_failures_and_preserves_partial_tail(self):
        from planning import run_framework as runner
        self.assertTrue(hasattr(runner, 'completed_method_prefix'), 'T4 resume prefix missing')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, rows, spec = self.make_data(root)
            config = dict(version='toolv2x_interact_run_v1', fixture=True)
            (root/'config.json').write_text(json.dumps(config))
            (root/'selected_index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            (root/'progress.json').write_text(json.dumps(dict(terminal_tasks=1, status='running')))
            task = dict(row=rows[0], status='failed', episode=dict(version='toolv2x_episode_v2',
                sample_id=rows[0]['sample_id'], scene=rows[0]['scene'], g=rows[0]['g'], status='invalid_plan',
                final_plan_id=None, events=[dict(kind='failed')]))
            (root/'first.json').write_text(json.dumps(task))
            records = [dict(sample_id=rows[0]['sample_id'], path='first.json')]
            (root/'tasks.jsonl').write_text(json.dumps(records[0])+'\n'+ '{"partial":')
            reused = runner.completed_method_prefix(root, rows, config)
            self.assertEqual(len(reused), 1)
            self.assertEqual(reused[0]['status'], 'failed')
            self.assertTrue((root/'tasks.jsonl').read_text().endswith('{"partial":'))
            with self.assertRaises(ValueError):
                runner.completed_method_prefix(root, rows, dict(config, fixture=False))
            task['episode']['status'] = 'running'
            (root/'first.json').write_text(json.dumps(task))
            with self.assertRaises(ValueError):
                runner.completed_method_prefix(root, rows, config)


    def test_interruption_resume_reuses_only_first_terminal_sample_without_loading_its_model_again(self):
        from planning import run_framework as runner
        self.assertTrue(hasattr(runner, 'interact'))
        from planning.context import build_task_plan_input
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, rows, spec = self.make_data(root)
            original_save = runner._save_method_json
            def save(path, value):
                original_save(path, value)
                ep = value.get('episode')
                if ep and ep['sample_id'] == rows[1]['sample_id'] and ep['events'][-1]['kind'] == 'response_received':
                    raise KeyboardInterrupt('interrupted second sample')
            def build(*a, **kw):
                kw['token_counter'] = lambda p: len(p)//4
                return build_task_plan_input(*a, **kw)
            with patch.object(runner, '_load_interaction_runtime', return_value=self.runtime(rows)), \
                 patch.object(runner, '_save_method_json', side_effect=save), \
                 patch('planning.method_episode.build_task_plan_input', side_effect=build):
                with self.assertRaises(KeyboardInterrupt):
                    runner.interact(data, root/'old', Path('/fixture/driver'), spec, per_recording=0)
            self.assertEqual(json.loads((root/'old/progress.json').read_text())['terminal_tasks'], 1)
            partial = (root/'old/sample_000001/task.json').read_bytes()
            old_first = json.loads((root/'old/sample_000000/task.json').read_text())
            runtime = self.runtime(rows)
            with patch.object(runner, '_load_interaction_runtime', return_value=runtime), \
                 patch('planning.method_episode.build_task_plan_input', side_effect=build):
                runner.interact(data, root/'new', Path('/fixture/driver'), spec, per_recording=0, resume_from=root/'old')
            self.assertEqual(len(runtime.driver.inputs), 3)
            self.assertEqual((root/'old/sample_000001/task.json').read_bytes(), partial)
            new_first = json.loads((root/'new/sample_000000/task.json').read_text())
            self.assertEqual(new_first['episode'], old_first['episode'])
            self.assertEqual(new_first['inputs']['artifact_root'], str(root/'old/sample_000000'))
            config = json.loads((root/'new/config.json').read_text())
            self.assertEqual(len(runner.completed_method_prefix(root/'new', rows, config)), 2)
