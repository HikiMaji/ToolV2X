"""Request-decision accounting regressions; synthetic driver, no model imports."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

import test_evidence_ledger as ledger_fixture
import test_method_episode as fixture
from evaluation.framework import _method_cost
from planning.method_controls import (capture_control_prefix, control_spec,
                                      make_control_policy)
from planning.method_episode import run_task_episode
from tools.vehicle import VehicleTools


class DecisionAccountingTests(unittest.TestCase):
    def setUp(self):
        ledger_fixture.LedgerFixture.setUp(self)

    def execute(self, policy, *, control=None, limits=None, progress=None,
                prefix=None, service=None, driver=None):
        ego = fixture.window('ego')
        service = service or VehicleTools(lambda: fixture.window(), self.predictor,
            'scene', 10, 'peer', task_provenance=fixture.provenance())
        driver = driver or fixture.EvidenceDriver([])
        inputs = dict(local_window=ego, local_prediction=self.predictor(ego),
            motion=dict(speed_mps=2., yaw_rate_rps=0.),
            features=dict(active_agent_mask=np.array([[[True], [False]]])),
            service=service, predictor=self.predictor, driver=driver,
            policy=make_control_policy(control, policy) if control else policy,
            limits=limits or fixture.limits(), local_provenance=fixture.provenance(),
            sample_id='sample', branch_id='accounting', token_counter=lambda s: len(s)//4,
            control_spec=control, on_progress=progress)
        return run_task_episode(**inputs, prefix=prefix), inputs, service, driver

    @staticmethod
    def cost(episode):
        return _method_cost(dict(episode=episode,
                                 inputs=dict(local_prediction_seconds=0.)))

    def test_none_and_explicit_feedback_both_measure_policy_time_without_persistence(self):
        for control in (None, control_spec('feedback')):
            with self.subTest(control=control and control['name']):
                clock = [0.]
                def policy(state):
                    clock[0] += 5.
                    return dict(tool='STOP', mode=None, reason='timed_fixture')
                def persist(episode):
                    if episode['events'][-1]['kind'] == 'decision':
                        clock[0] += 7.
                with patch('planning.method_episode.perf_counter', side_effect=lambda: clock[0]):
                    episode, _, _, _ = self.execute(policy, control=control, progress=persist)
                cost = self.cost(episode)
                self.assertGreaterEqual(cost['control_seconds'], 5.)
                self.assertEqual(cost['control_seconds'], 5.)
                self.assertTrue(cost['cost_complete'], cost)
                self.assertIn('compute_accounting_version', episode)
                self.assertFalse(any(e['kind'] == 'control' for e in
                                     episode['decisions'][0]['state']['cost_ledger']))

    def test_no_call_budget_stop_has_one_explicit_zero_duration_decision(self):
        def forbidden(state):
            self.fail('forced budget STOP called the policy')
        with patch('planning.method_episode.perf_counter', return_value=0.):
            episode, _, _, _ = self.execute(forbidden, limits=fixture.limits(max_calls=0))
        events = [e for e in episode['cost_events'] if e['kind'] == 'control']
        self.assertEqual([(e['stage'], e['seconds'], e['complete']) for e in events],
                         [(0, 0., True)])
        cost = self.cost(episode)
        self.assertEqual(cost['control_seconds'], 0.)
        self.assertTrue(cost['cost_complete'], cost)

    def test_policy_exception_before_decision_append_keeps_known_incomplete_prefix(self):
        clock = [0.]
        def broken(state):
            clock[0] += 3.
            raise RuntimeError('policy failed')
        with patch('planning.method_episode.perf_counter', side_effect=lambda: clock[0]):
            episode, _, _, _ = self.execute(broken)
        self.assertEqual(episode['decisions'], [])
        event = [e for e in episode['cost_events'] if e['kind'] == 'control'][0]
        self.assertEqual((event['stage'], event['seconds'], event['complete']), (0, 3., False))
        cost = self.cost(episode)
        self.assertIsNone(cost['control_seconds'])
        self.assertEqual(cost['known_cost']['control_seconds'], 3.)
        self.assertFalse(cost['cost_complete'])

    def test_dispatch_failure_keeps_decision_stage_incomplete(self):
        clock = [0.]
        driver = fixture.EvidenceDriver([])
        def policy(state):
            clock[0] += 2.
            driver.context_limit = 9999
            return dict(tool='P', mode='current', reason='drift')
        with patch('planning.method_episode.perf_counter', side_effect=lambda: clock[0]):
            episode, _, _, _ = self.execute(policy, driver=driver)
        self.assertEqual(episode['error']['stage'], 'frozen_configuration_or_dispatch')
        self.assertEqual(len(episode['decisions']), 1)
        event = [e for e in episode['cost_events'] if e['kind'] == 'control'][0]
        self.assertEqual((event['stage'], event['seconds'], event['complete']), (0, 2., False))
        cost = self.cost(episode)
        self.assertIsNone(cost['control_seconds'])
        self.assertEqual(cost['known_cost']['control_seconds'], 2.)
        self.assertFalse(cost['cost_complete'])

    def test_progress_interruption_preserves_known_but_incomplete_decision_stage(self):
        saved, clock = [], [0.]
        def policy(state):
            clock[0] += 4.
            return dict(tool='STOP', mode=None, reason='timed_fixture')
        def interrupt(episode):
            if episode['events'][-1]['kind'] == 'decision':
                saved.append(episode)
                raise KeyboardInterrupt()
        with patch('planning.method_episode.perf_counter', side_effect=lambda: clock[0]):
            with self.assertRaises(KeyboardInterrupt):
                self.execute(policy, progress=interrupt)
        event = [e for e in saved[-1]['cost_events'] if e['kind'] == 'control'][0]
        self.assertEqual((event['stage'], event['seconds'], event['complete']), (0, 4., False))
        cost = self.cost(saved[-1])
        self.assertIsNone(cost['control_seconds'])
        self.assertEqual(cost['known_cost']['control_seconds'], 4.)
        self.assertIsNone(cost['total_compute_seconds'])
        self.assertFalse(cost['cost_complete'])

    def test_prefix_reuse_keeps_existing_stage_and_accounts_only_new_decisions(self):
        control = control_spec('feedback')
        episode, inputs, service, driver = self.execute(fixture.feedback, control=control)
        inputs.update(service=VehicleTools(lambda: fixture.window(), self.predictor,
            'scene', 10, 'peer', task_provenance=fixture.provenance()), driver=fixture.EvidenceDriver([]))
        prefix = capture_control_prefix(**inputs, after_calls=1)
        resumed, _, _, _ = self.execute(fixture.feedback, control=control, prefix=prefix,
            service=copy.deepcopy(inputs['service']), driver=inputs['driver'])
        stages = [e['stage'] for e in resumed['cost_events'] if e['kind'] == 'control']
        self.assertEqual(stages, [0, 1, 2])
        self.assertEqual(len(stages), len(set(stages)))
        self.assertTrue(self.cost(resumed)['cost_complete'])

    def test_exact_repeat_accounts_only_the_final_actual_decision(self):
        control = control_spec('exact_repeat')
        episode, _, _, _ = self.execute(fixture.feedback, control=control)
        self.assertEqual(len(episode['plans']), 3)
        self.assertEqual([e['stage'] for e in episode['cost_events'] if e['kind'] == 'control'], [2])
        self.assertTrue(self.cost(episode)['cost_complete'])

    def test_missing_and_legacy_timing_are_unknown_but_explicit_old_timing_remains_usable(self):
        episode, _, _, _ = self.execute(
            lambda state: dict(tool='STOP', mode=None, reason='fixture'))
        old_explicit = copy.deepcopy(episode)
        old_explicit.pop('compute_accounting_version')
        self.assertTrue(self.cost(old_explicit)['cost_complete'])

        missing = copy.deepcopy(old_explicit)
        missing['cost_events'] = [e for e in missing['cost_events'] if e['kind'] != 'control']
        missing_cost = self.cost(missing)
        self.assertIsNone(missing_cost['control_seconds'])
        self.assertEqual(missing_cost['known_cost']['control_seconds'], 0)
        self.assertFalse(missing_cost['cost_complete'])

        legacy = copy.deepcopy(old_explicit)
        event = next(e for e in legacy['cost_events'] if e['kind'] == 'control')
        event.pop('timing_version')
        event['seconds'] = 9.
        legacy_cost = self.cost(legacy)
        self.assertIsNone(legacy_cost['control_seconds'])
        self.assertEqual(legacy_cost['known_cost']['control_seconds'], 0)
        self.assertEqual(legacy_cost['reported_control_seconds'], 9.)
        self.assertFalse(legacy_cost['cost_complete'])
