"""T5 offline evaluation of actual T4 archives; no model or training dependency."""
import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import test_method_episode as fixtures
from test_task_spec import provenance
from test_vehicle_tools import window, fixture_prediction
from tools.task_spec import FrozenPredictor
from tools.vehicle import VehicleTools
from planning.method_episode import run_task_episode, diagnostic_policy

SCENE = 'testoutput_CAV_data_2022-03-15-10-09-50_0'
POLICY = 'p_current_f_change'


def task_fixture(max_calls=2, invalid_stage=None, raise_stage=None, progress=None, scene=SCENE):
    predictor = FrozenPredictor(fixture_prediction, provenance()['prediction'], dict(adapter='fixture'))
    ego, peer = window('ego'), window()
    ego['scene'] = peer['scene'] = scene
    config = fixtures.limits(max_calls=max_calls, policy_id=POLICY)
    service = VehicleTools(lambda: peer, predictor, scene, 10, 'peer', task_provenance=provenance())
    motion = dict(speed_mps=2., yaw_rate_rps=0.)
    sample = scene+':10'
    ep = run_task_episode(ego, predictor(ego), motion, dict(active_agent_mask=np.array([[[True], [False]]])),
        service, predictor, fixtures.EvidenceDriver([], invalid_stage, raise_stage), diagnostic_policy, config,
        local_provenance=provenance(), sample_id=sample, branch_id=POLICY,
        token_counter=lambda p: len(p)//4, on_progress=progress)
    task = dict(row=dict(sample_id=sample, scene=scene, g=10, local_frame=10, role='validation',
                        physical_split='train', ego_motion=motion),
        episode=ep, status='completed' if ep['status']=='completed' else 'failed',
        inputs=dict(local_prediction_seconds=.5, local_model_targets=2))
    label = dict(sample_id=sample, scene=scene, role='validation', g=10, times_seconds=[.5,1.,1.5,2.,2.5,3.],
                 waypoints=[[float(i+1), 0.] for i in range(6)], valid=[True]*6)
    return task, label


def fixed_measurements(task):
    """Known synthetic durations in real archive shape, separate from model output quality."""
    ep = task['episode']
    for e in ep['cost_events']:
        if e['kind'] == 'driver':
            e['attempt_seconds'] = float(e['stage']+1)
            e['generation_cost']['seconds'] = float(e['stage'])+.5
            ep['plans'][e['stage']]['driver_attempt_seconds'] = e['attempt_seconds']
            ep['plans'][e['stage']]['output']['q9_cost']['seconds'] = e['generation_cost']['seconds']
        elif e['kind'] == 'service':
            e['attempt_seconds'] = 2.
            e['service_cost']['service_seconds'] = 1.5
            e['service_cost']['model_seconds'] = float(e['stage'])
        elif e['kind'] == 'receiver':
            e['seconds'] = 3.
        else:
            e['seconds'] = .1
    if ep['ledger_snapshots'][-1]['receiver_events']:
        for i, e in enumerate(ep['ledger_snapshots'][-1]['receiver_events']):
            e['model_seconds'] = 2. if i == 0 else 0.
    return task


class MethodEvaluationTests(unittest.TestCase):
    def api(self):
        module = importlib.import_module('evaluation.framework')
        self.assertTrue(hasattr(module, 'evaluate_method_task'), 'T5 v2 evaluation missing')
        return module

    def test_three_driver_calls_sum_outer_times_and_keep_model_subitems_separate(self):
        task, label = task_fixture()
        fixed_measurements(task)
        row = self.api().evaluate_method_task(task, label)
        self.assertEqual(row['driver_calls'], 3)
        self.assertEqual(row['driver_seconds'], 6.)
        self.assertEqual(row['generation_seconds'], 4.5)
        self.assertEqual(row['service_seconds'], 4.)
        self.assertEqual(row['peer_model_seconds'], 1.)
        self.assertEqual(row['receiver_seconds'], 6.)
        self.assertEqual(row['receiver_model_seconds'], 2.)
        self.assertAlmostEqual(row['total_compute_seconds'], 16.8)
        self.assertTrue(row['cost_complete'])
        self.assertEqual(row['request_bytes'], sum(len(json.dumps(q, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode())
                                                   for q in task['episode']['requests']))
        self.assertEqual(row['response_bytes'], sum(len(bytes.fromhex(r['wire_hex'])) for r in task['episode']['responses']))
        self.assertEqual(len(row['plans']), 3)
        self.assertTrue(row['task_success'])
        self.assertAlmostEqual(row['ADE3'], .35)
        self.assertAlmostEqual(row['plans'][0]['raw_ADE3'], 0.)

    def test_stop_counts_one_driver_and_no_remote_work_without_guessing_gpu_flops(self):
        task, label = task_fixture(max_calls=0)
        row = self.api().evaluate_method_task(fixed_measurements(task), label)
        self.assertEqual(row['driver_calls'], 1)
        self.assertEqual(row['driver_seconds'], 1.)
        self.assertEqual(row['calls'], 0)
        self.assertEqual(row['response_bytes'], 0)
        self.assertEqual(row['receiver_model_seconds'], 0.)
        self.assertAlmostEqual(row['total_compute_seconds'], 1.6)
        self.assertIsNone(row['policy_seconds'])
        self.assertIsNone(row['end_to_end_seconds'])

    def test_middle_invalid_output_keeps_all_cost_and_never_uses_previous_valid_as_final(self):
        task, label = task_fixture(invalid_stage=1)
        row = self.api().evaluate_method_task(task, label)
        self.assertFalse(row['parse_valid'])
        self.assertFalse(row['task_success'])
        self.assertIsNone(row['ADE3'])
        self.assertEqual(row['driver_calls'], 2)
        self.assertGreater(row['response_bytes'], 0)
        self.assertEqual(row['plans'][0]['raw_ADE3'], 0.)
        self.assertEqual(row['plans'][1]['raw_answer'], 'malformed original model answer')

    def test_unknown_generation_preserves_measured_time_bytes_and_excludes_unknown_tokens(self):
        task, label = task_fixture(raise_stage=1)
        module = self.api()
        row = module.evaluate_method_task(task, label)
        self.assertFalse(row['cost_complete'])
        self.assertIsNone(row['output_tokens'])
        self.assertIsNone(row['generation_seconds'])
        self.assertGreater(row['known_cost']['output_tokens'], 0)
        self.assertGreater(row['driver_seconds'], 0)
        self.assertGreater(row['response_bytes'], 0)
        stats = module.summarize_method([row])['groups'][0]['costs']['output_tokens']
        self.assertEqual(stats['known_count'], 0)
        self.assertEqual(stats['unknown_count'], 1)
        self.assertIsNone(stats['mean_known'])

    def test_interruption_after_response_keeps_wire_but_pending_receiver_is_unknown(self):
        saved = []
        def stop(ep):
            saved.append(ep)
            if ep['events'][-1]['kind']=='response_received':
                raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            task_fixture(progress=stop)
        task, label = task_fixture(max_calls=0)
        task.update(episode=saved[-1], status='running')
        row = self.api().evaluate_method_task(task, label)
        self.assertFalse(row['task_success'])
        self.assertEqual(row['artifact_status'], 'incomplete')
        self.assertGreater(row['response_bytes'], 0)
        self.assertIsNone(row['receiver_seconds'])
        self.assertIsNone(row['total_compute_seconds'])
        self.assertFalse(row['cost_complete'])


    def test_failed_input_build_preserves_measured_duration_in_cost_events(self):
        task, label = task_fixture(max_calls=0)
        # Fail at the actual receiver configuration boundary, before any driver call.
        from unittest.mock import patch
        original = fixtures.limits
        def invalid(**kwargs):
            value = original(**kwargs)
            value['receiver_spec']['peer_reserve'] = 12000
            return value
        with patch.object(fixtures, 'limits', side_effect=invalid):
            failed, label = task_fixture()
        ep = failed['episode']
        self.assertEqual(ep['error']['stage'], 'input_build')
        self.assertTrue(any(e['kind']=='input_build' for e in ep['cost_events']),
                        'failed input construction lost its stage duration')
        row = self.api().evaluate_method_task(failed, label)
        self.assertGreater(row['input_build_seconds'], 0)
        self.assertEqual(row['input_build_seconds'], ep['cost']['input_build_seconds'])
        self.assertFalse(row['task_success'])
        self.assertFalse(row['cost_complete'])

    def test_requested_but_unrecorded_service_cost_is_unknown_not_free(self):
        snapshots = []
        def stop(ep):
            snapshots.append(ep)
            if ep['events'][-1]['kind']=='request_started':
                raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            task_fixture(progress=stop)
        task, label = task_fixture(max_calls=0)
        task.update(episode=snapshots[-1], status='running')
        row = self.api().evaluate_method_task(task, label)
        self.assertEqual(row['calls'], 1)
        self.assertIsNone(row['request_bytes'])
        self.assertIsNone(row['response_bytes'])
        self.assertIsNone(row['service_seconds'])
        self.assertEqual(row['known_cost']['response_bytes'], 0)

    def test_no_labels_or_invalid_last_label_never_changes_parse_or_execution_success(self):
        task, label = task_fixture()
        module = self.api()
        missing = module.evaluate_method_task(task, None)
        self.assertTrue(missing['parse_valid'])
        self.assertTrue(missing['task_success'])
        self.assertEqual(missing['label_status'], 'missing')
        self.assertIsNone(missing['ADE3'])
        label['valid'][-1] = False
        partial = module.evaluate_method_task(task, label)
        self.assertAlmostEqual(partial['ADE3'], .3)
        self.assertIsNone(partial['FDE3'])
        label['valid'] = [False]*6
        absent = module.evaluate_method_task(task, label)
        self.assertIsNone(absent['ADE3'])
        self.assertTrue(absent['parse_valid'])

    def test_abnormal_but_parsed_answer_retains_raw_error_and_fails_admissibility(self):
        task, label = task_fixture(max_calls=0)
        p = task['episode']['plans'][0]
        p['output']['waypoints'] = [[10000., 0.]]*6
        p['output']['q9_raw'] = 'The suggested trajectory is: '+repr(p['output']['waypoints'])
        p['status'] = 'invalid_plan'
        task.update(status='failed')
        task['episode'].update(status='invalid_plan', final_plan_id=None)
        task['episode']['events'][-1]['kind'] = 'failed'
        row = self.api().evaluate_method_task(task, label)
        self.assertTrue(row['parse_valid'])
        self.assertFalse(row['admissibility'])
        self.assertFalse(row['task_success'])
        self.assertAlmostEqual(row['raw_ADE3'], 9996.5)
        self.assertIsNone(row['ADE3'])

    def test_same_global_frame_different_recording_label_mismatch_is_explicit(self):
        task, label = task_fixture()
        label['scene'] = 'testoutput_CAV_data_2022-03-21-09-50-20_0'
        row = self.api().evaluate_method_task(task, label)
        self.assertEqual(row['label_status'], 'identity_mismatch')
        self.assertIsNone(row['ADE3'])
        self.assertTrue(row['parse_valid'])

    def test_duplicate_stage_or_false_final_pointer_cannot_be_scored_as_success(self):
        task, label = task_fixture()
        for alter in ('duplicate', 'wrong_final'):
            bad = copy.deepcopy(task)
            if alter=='duplicate':
                bad['episode']['plans'].append(copy.deepcopy(bad['episode']['plans'][0]))
            else:
                bad['episode']['final_plan_id'] = 'plan_0'
            with self.subTest(alter=alter):
                result = self.api().evaluate_method_task(bad, label)
                self.assertFalse(result['task_success'])
                self.assertEqual(result['artifact_status'], 'invalid_artifact')


    def test_conflicting_artifact_records_are_preserved_but_excluded_from_means(self):
        module = self.api()
        task, label = task_fixture(max_calls=0)
        for damage in ('duplicate_plan','wrong_identity','conflicting_duration'):
            bad = copy.deepcopy(task)
            if damage=='duplicate_plan':
                bad['episode']['plans'].append(copy.deepcopy(bad['episode']['plans'][0]))
            elif damage=='wrong_identity':
                bad['episode']['scene'] = 'wrong_recording'
            else:
                next(e for e in bad['episode']['cost_events'] if e['kind']=='driver')['attempt_seconds'] = 9999.
            with self.subTest(damage=damage):
                row = module.evaluate_method_task(bad, label)
                self.assertEqual(row['artifact_status'], 'invalid_artifact')
                report = module.summarize_method([row])['groups'][0]
                self.assertEqual(report['stages'][0]['raw_metrics']['ADE3']['known_count'], 0)
                self.assertGreater(report['stages'][0]['excluded_artifact_plans'], 0)
                self.assertEqual(report['costs']['driver_seconds']['known_count'], 0)
                self.assertTrue(row['plans'])
                self.assertIsNotNone(row['reported_cost'])

    def test_missing_preparation_record_does_not_invent_zero_model_cost(self):
        task, label = task_fixture()
        task.update(status='failed', episode=None, inputs={}, error=dict(stage='local_preparation'))
        row = self.api().evaluate_method_task(task, label, policy_id=POLICY, branch_id=POLICY)
        self.assertFalse(row['task_success'])
        self.assertIsNone(row['local_model_seconds'])
        self.assertIsNone(row['total_compute_seconds'])
        self.assertFalse(row['cost_complete'])

    def test_old_saved_160_answers_keep_identical_raw_ade_fde_in_new_stage_reader(self):
        module = self.api()
        records = [json.loads(s) for s in (ROOT/'outputs/framework_epoch01_quick_eval_2026_09_12/review_predictions.jsonl').read_text().splitlines()]
        from evaluation.planning import evaluate_plan
        self.assertEqual(len(records), 160)
        for r in records:
            output = dict(decoding='direct', q8_executed=False, q9_executed=True,
                          q9_raw=r['raw_answer'], waypoints=r['waypoints'])
            label = dict(valid=r['truth_valid'], times_seconds=r['times_seconds'], waypoints=r['truth'])
            old = evaluate_plan(output, label, dict(speed_mps=None))
            new = module.method_plan_row(dict(plan_id='archived_single', stage=0, output=output), label,
                dict(speed_mps=None), fixtures.limits()['execution_spec'])
            self.assertAlmostEqual(new['raw_ADE3'], old['ADE3'], places=10)
            self.assertAlmostEqual(new['raw_FDE3'], r['FDE3'], places=10)


class MethodArchiveTests(unittest.TestCase):
    def api(self):
        module = importlib.import_module('evaluation.framework')
        self.assertTrue(hasattr(module, 'evaluate_method'), 'T5 archive evaluator missing')
        return module

    def archive(self, root, tasks, labels):
        (root/'labels').mkdir()
        (root/'run').mkdir()
        run = root/'run'
        policy = tasks[0]['episode']['policy_id']
        (run/'config.json').write_text(json.dumps(dict(version='toolv2x_interact_run_v1',
            spec=dict(limits=tasks[0]['episode']['limits']))))
        (run/'selected_index.jsonl').write_text(''.join(json.dumps(t['row'])+'\n' for t in tasks))
        (run/'progress.json').write_text(json.dumps(dict(status='completed', terminal_tasks=len(tasks))))
        records=[]
        for i,t in enumerate(tasks):
            name='sample%d.json'%i
            (run/name).write_text(json.dumps(t))
            records.append(dict(sample_id=t['row']['sample_id'], path=name))
        (run/'tasks.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        (root/'labels/validation.jsonl').write_text(''.join(json.dumps(l)+'\n' for l in labels))
        return run, records

    def test_expected_denominator_keeps_initial_invalid_missing_and_duplicate_artifacts(self):
        module = self.api()
        cases = [task_fixture(max_calls=0, invalid_stage=0, scene=SCENE),
                 task_fixture(max_calls=0, scene='testoutput_CAV_data_2022-03-17-16-06-11_0'),
                 task_fixture(max_calls=0, scene='testoutput_CAV_data_2022-03-21-09-50-20_0')]
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);run, records=self.archive(root,[t for t,l in cases],[l for t,l in cases])
            (run/'sample1.json').unlink()
            with (run/'tasks.jsonl').open('a') as h:
                h.write(json.dumps(records[2])+'\n'+ '{"interrupted":')
            report = module.evaluate_method(run, root/'eval', root/'labels')
            rows=[json.loads(s) for s in (root/'eval/rows.jsonl').read_text().splitlines()]
            self.assertEqual(report['attempts'], 3)
            self.assertEqual(report['task_successes'], 0)
            self.assertEqual(len(rows), 3)
            self.assertEqual([r['artifact_status'] for r in rows], ['recorded_failure','missing_artifact','duplicate_artifact'])
            self.assertTrue(report['archive_issues'])
            self.assertEqual(len(report['by_recording']),3)

    def test_labels_are_joined_by_identity_and_evaluation_does_not_read_artifact_features(self):
        module=self.api()
        cases=[task_fixture(max_calls=0, scene=s) for s in (SCENE,'testoutput_CAV_data_2022-03-17-16-06-11_0')]
        cases[1][1]['waypoints']=[[float(i+2),0.] for i in range(6)]
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);run,_=self.archive(root,[t for t,l in cases],[l for t,l in reversed(cases)])
            for i,(task,label) in enumerate(cases):
                task['inputs']['artifact_root']='/nonexistent/original_resumed_run'
                (run/('sample%d.json'%i)).write_text(json.dumps(task))
            module.evaluate_method(run,root/'eval',root/'labels')
            rows=[json.loads(s) for s in (root/'eval/rows.jsonl').read_text().splitlines()]
            self.assertEqual([r['ADE3'] for r in rows],[0.,1.])
            self.assertEqual([r['g'] for r in rows],[10,10])
            self.assertEqual(len({(r['sample_id'],r['policy_id'],r['branch_id']) for r in rows}),2)
