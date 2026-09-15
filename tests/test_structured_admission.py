"""Torch-free audit of evidence actually admitted by numeric episodes."""
import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from test_evidence_ledger import LedgerFixture
from test_method_episode import EvidenceDriver, limits
from test_structured_inputs import empty_window, one_track
from test_task_spec import provenance, spec_dict
from test_vehicle_tools import window
from planning.method_controls import control_spec

SCENE = 'testoutput_CAV_data_2022-03-15-10-09-50_0'


class NumericContractDriver:
    """Contract fixture only; it is not trainable-model or effect evidence."""
    def __init__(self, spec):
        self.spec = spec
        self.provenance = dict(driver_kind='structured', decoding='numeric',
            driver_spec=spec.to_dict(), model_version=dict(name='numeric_contract_fixture',
                revision='v1', training=dict(status='test_only_untrained', optimizer_steps=0)))

    def prepare_input(self, features, motion, ledger, receiver_spec, **prior):
        from planning.structured_inputs import build_structured_plan_input
        if receiver_spec != self.spec.to_dict():
            raise ValueError('fixture receiver specification changed')
        return build_structured_plan_input(motion, ledger, self.spec,
            ego_history=features.get('ego_history'), **prior)

    def plan_prepared(self, features, prepared):
        tensors = prepared['tensor_inputs']
        has_remote_tensor = any(group['use'] == 'tensor'
            for group in prepared['admission_report']['field_groups'])
        x = 0. if has_remote_tensor else 20.
        previous = prepared['previous_plan']
        y = 0. if previous is None else previous[0][1] + .1
        waypoints = [[x, y]] * 6
        count = (220 + sum(tensors['entity_mask']) +
                 int(np.asarray(tensors['forecast_mask'], bool).any(axis=-1).sum()) + 9)
        return dict(output_version='toolv2x_numeric_plan_v1', driver_kind='structured',
            status='valid', waypoints=waypoints, prepared_input=copy.deepcopy(prepared),
            driver_cost=dict(seconds=0., numeric_token_count=count, output_points=6,
                             model_executed=True),
            parent_refs=copy.deepcopy(prepared['admission_report']['admitted_field_refs']))


class StructuredAdmissionContractTests(LedgerFixture):
    def test_structured_audit_module_and_api_exist(self):
        module = ROOT / 'src/evaluation/structured.py'
        self.assertTrue(module.is_file(), 'structured evidence-use audit module is missing')
        from evaluation.structured import audit_structured_episode
        self.assertTrue(callable(audit_structured_episode))

    def episode(self, local, peer, *, max_entities=2, max_forecasts=4,
                max_targets=1, tools=('P',), p_processing='observations_only',
                control=None, ego_history=None):
        from planning.method_episode import run_task_episode
        from planning.structured_inputs import StructuredDriverSpec
        from tools.vehicle import VehicleTools
        local, peer = copy.deepcopy(local), copy.deepcopy(peer)
        local['scene'] = peer['scene'] = SCENE
        spec = StructuredDriverSpec(max_entities=max_entities,
            max_forecast_sets_per_entity=max_forecasts, p_processing=p_processing)
        driver = NumericContractDriver(spec)
        service = VehicleTools(lambda: peer, self.predictor, peer['scene'], peer['g'],
                               peer['source'], task_provenance=provenance())
        def policy(state):
            index = len(state['response_receipts'])
            if index < len(tools):
                return dict(tool=tools[index], mode='current', reason='fixture_request')
            return dict(tool='STOP', mode=None, reason='fixture_complete')
        execution = spec_dict(max_targets=max_targets, max_request_bytes=30000,
            max_response_bytes=30000, max_episode_bytes=100000,
            max_plan_acceleration_mps2=100.)
        config = limits(version='toolv2x_interaction_v2', receiver_spec=spec.to_dict(),
            execution_spec=execution, driver_version={k:driver.provenance['model_version'][k]
                for k in ('name', 'revision')})
        features = dict(active_agent_mask=np.array([[[True], [False]]], dtype=bool))
        if ego_history is not None:
            features['ego_history'] = copy.deepcopy(ego_history)
        return run_task_episode(local, self.predictor(local),
            dict(speed_mps=2., yaw_rate_rps=0.), features, service, self.predictor,
            driver, policy, config, local_provenance=provenance(), sample_id='sample',
            branch_id='structured_audit', control_spec=control)

    @staticmethod
    def task(ep):
        return dict(row=dict(sample_id='sample', scene=ep['scene'], g=ep['g'], role='train'),
                    episode=ep, status='completed' if ep['status'] == 'completed' else 'failed',
                    inputs=dict(local_prediction_seconds=.1))

    def test_capacity_reports_direct_and_indirect_remote_refs_at_real_slots(self):
        from evaluation.framework import evaluate_method_task
        from evaluation.structured import audit_structured_episode
        for capacity in (1, 2):
            with self.subTest(capacity=capacity):
                ep = self.episode(one_track('ego', 10.), one_track('peer', 30.),
                                  max_entities=capacity)
                audit = audit_structured_episode(ep)
                stage = audit[1]
                self.assertEqual([ref['field_kind'] for ref in stage['known_remote_refs']],
                                 ['anchor', 'history'])
                self.assertEqual(stage['new_remote_refs'], stage['known_remote_refs'])
                self.assertEqual(stage['previously_acquired_remote_refs'], [])
                direct = stage['direct_remote_primary_refs']
                self.assertEqual([ref['field_kind'] for ref in direct],
                                 [] if capacity == 1 else ['history'])
                if capacity == 1:
                    self.assertEqual(stage['indirect_only_remote_refs'],
                                     stage['known_remote_refs'])
                    self.assertFalse(any(row['source_role'] == 'remote'
                                         for row in stage['observation_coverage']))
                else:
                    remote = next(row for row in stage['observation_coverage']
                                  if row['source_role'] == 'remote')
                    self.assertEqual(remote['tensor_location'],
                                     dict(array='observations', entity=1, source_slot=1))
                    self.assertEqual(remote['valid_steps'], list(range(11)))
                    self.assertEqual(remote['timepoints_seconds'],
                                     (np.arange(-10, 1) / 10.).tolist())
                self.assertEqual(stage['counts']['direct_remote_primary_refs'], len(direct))
                self.assertEqual(stage['capacities']['max_entities'], capacity)
                self.assertEqual(evaluate_method_task(self.task(ep), None)['structured_audit'], audit)
                json.dumps(audit, allow_nan=False)

    def test_remote_only_f_and_ambiguous_p_report_actual_mask_coverage(self):
        from evaluation.structured import audit_structured_episode
        remote_f = audit_structured_episode(self.episode(
            empty_window('ego'), one_track('peer', 30.), tools=('F',)))[1]
        self.assertEqual([ref['field_kind'] for ref in remote_f['direct_remote_primary_refs']],
                         ['forecast'])
        self.assertEqual(remote_f['association_status_counts'], {'unmatched': 1})
        forecast = remote_f['forecast_coverage'][0]
        self.assertEqual(forecast['source_role'], 'remote')
        self.assertEqual(forecast['context_scope'], 'provider_full_at_t')
        self.assertEqual(forecast['valid_modes'], list(range(6)))
        self.assertEqual(forecast['valid_timepoints_per_mode'], [6] * 6)
        self.assertEqual(forecast['valid_mode_timepoints'], 36)

        matched_f = audit_structured_episode(self.episode(
            one_track('ego', 10.), one_track('peer', 10.2, 99),
            max_forecasts=2, tools=('F',)))[1]
        self.assertEqual({row['source_role'] for row in matched_f['forecast_coverage']},
                         {'local', 'remote'})
        self.assertEqual({row['context_scope'] for row in matched_f['forecast_coverage']},
                         {'ego_full_at_t', 'provider_full_at_t'})
        capped_f = audit_structured_episode(self.episode(
            one_track('ego', 10.), one_track('peer', 10.2, 99),
            max_forecasts=1, tools=('F',)))[1]
        self.assertEqual(capped_f['capacities']['structured_capacity_filtered_forecast_sets'], 1)
        self.assertEqual(capped_f['direct_remote_primary_refs'], [])

        peer = window('peer')
        peer['states'][:, :, 0] = [[9.9] * 11, [10.1] * 11]
        ambiguous = audit_structured_episode(self.episode(
            one_track('ego', 10.), peer, max_entities=3, max_targets=2,
            tools=('P',)))[1]
        self.assertEqual(ambiguous['association_status_counts'], {'ambiguous': 3})
        self.assertEqual(len(ambiguous['direct_remote_primary_refs']), 2)

        empty = audit_structured_episode(self.episode(
            empty_window('ego'), empty_window('peer'), tools=()))[0]
        self.assertEqual(empty['capacities']['tensor_entities'], 0)
        self.assertEqual(empty['observation_coverage'], [])
        self.assertEqual(empty['forecast_coverage'], [])

        masked_local = one_track('ego', 10.)
        masked_local['valid'][0, :-2] = False
        masked = audit_structured_episode(self.episode(
            masked_local, empty_window('peer'), tools=()))[0]
        local_observation = next(row for row in masked['observation_coverage']
                                 if row['source_role'] == 'local')
        self.assertEqual(local_observation['valid_steps'], [9, 10])
        self.assertEqual(local_observation['timepoints_seconds'], [-.1, 0.])

        ego_history = dict(states=[[0., 0., 0.]] * 11, valid=[True] * 11,
            times=(np.arange(-10, 1) / 10.).tolist(), read_paths=[])
        filtered = audit_structured_episode(self.episode(
            one_track('ego', .1, 3), empty_window('peer'), tools=(),
            ego_history=ego_history))[0]
        self.assertEqual(filtered['capacities']['ego_filtered_entities'], 1)
        self.assertEqual(filtered['capacities']['tensor_entities'], 0)

    def test_full_p_and_equivalent_f_keep_multirole_dependency_closure(self):
        from evaluation.structured import audit_structured_episode
        ep = self.episode(empty_window('ego'), one_track('peer', 30.),
                          tools=('P', 'F'), p_processing='local_mtr')
        audit = audit_structured_episode(ep)
        after_p, after_f = audit[1], audit[2]
        self.assertEqual({ref['field_kind'] for ref in after_p['known_acquired_remote_refs']},
                         {'anchor', 'history'})
        self.assertEqual({ref['field_kind'] for ref in after_p['new_acquired_remote_refs']},
                         {'anchor', 'history'})
        self.assertEqual(after_p['previously_acquired_remote_refs'], [])
        self.assertEqual([ref['field_kind'] for ref in
                          after_p['known_receiver_derived_remote_refs']], ['forecast'])
        self.assertEqual(after_p['new_receiver_derived_remote_refs'],
                         after_p['known_receiver_derived_remote_refs'])
        self.assertEqual(after_p['previously_receiver_derived_remote_refs'], [])
        self.assertEqual({key:after_p['counts'][key] for key in (
            'new_acquired_remote_refs', 'previously_acquired_remote_refs',
            'new_receiver_derived_remote_refs', 'previously_receiver_derived_remote_refs')},
            dict(new_acquired_remote_refs=2, previously_acquired_remote_refs=0,
                 new_receiver_derived_remote_refs=1,
                 previously_receiver_derived_remote_refs=0))
        self.assertEqual([ref['field_kind'] for ref in after_p['direct_remote_primary_refs']],
                         ['history', 'forecast'])
        self.assertEqual({ref['field_kind'] for ref in after_p['direct_dependency_closure_refs']},
                         {'anchor', 'history', 'forecast'})
        self.assertEqual(len(after_p['forecast_coverage']), 1)
        self.assertEqual(len(after_p['forecast_coverage'][0]['primary_refs']), 1)
        self.assertEqual(len(after_f['forecast_coverage']), 1)
        self.assertEqual(len(after_f['forecast_coverage'][0]['primary_refs']), 2)
        self.assertEqual({ref['field_kind'] for ref in after_f['direct_remote_primary_refs']},
                         {'history', 'forecast'})
        self.assertTrue(after_f['prior_dependency_refs'])
        self.assertEqual([ref['field_kind'] for ref in after_f['new_acquired_remote_refs']],
                         ['forecast'])
        self.assertEqual({ref['field_kind'] for ref in
                          after_f['previously_acquired_remote_refs']}, {'anchor', 'history'})
        self.assertEqual(after_f['new_receiver_derived_remote_refs'], [])
        self.assertEqual(after_f['previously_receiver_derived_remote_refs'],
                         after_f['known_receiver_derived_remote_refs'])
        self.assertEqual(after_f['counts']['known_acquired_remote_refs'], 3)
        self.assertEqual(after_f['counts']['known_receiver_derived_remote_refs'], 1)
        self.assertEqual(after_f['counts']['new_acquired_remote_refs'], 1)
        self.assertEqual(after_f['counts']['previously_acquired_remote_refs'], 2)
        self.assertEqual(after_f['counts']['new_receiver_derived_remote_refs'], 0)
        self.assertEqual(after_f['counts']['previously_receiver_derived_remote_refs'], 1)

    def test_repeat_refinement_prior_only_and_reference_only_differences(self):
        from evaluation.structured import audit_structured_episode
        for name, expected_prior_changes in (
                ('exact_repeat', [None, False, False]),
                ('self_refinement', [None, True, True])):
            with self.subTest(name=name):
                ep = self.episode(one_track('ego', 10.), empty_window('peer'), tools=(),
                    control=control_spec(name, driver_calls=3, repeat_after_calls=0))
                audit = audit_structured_episode(ep)
                self.assertEqual([row['tensor_changed_from_previous'] for row in audit],
                                 [None, False, False])
                self.assertEqual([row['prior_changed_from_previous'] for row in audit],
                                 expected_prior_changes)
                self.assertEqual([row['output_changed_from_previous'] for row in audit],
                                 [None, name == 'self_refinement', name == 'self_refinement'])
                self.assertTrue(all(row['output_valid'] for row in audit))

        reference_only = audit_structured_episode(self.episode(
            one_track('ego', 10.), one_track('peer', 30.), tools=('P', 'P'),
            p_processing='local_mtr'))
        self.assertEqual(reference_only[2]['new_remote_refs'], [])
        self.assertEqual(reference_only[2]['previously_acquired_remote_refs'],
                         reference_only[2]['known_acquired_remote_refs'])
        self.assertEqual(reference_only[2]['new_acquired_remote_refs'], [])
        self.assertEqual(reference_only[2]['new_receiver_derived_remote_refs'], [])
        self.assertEqual(reference_only[2]['previously_receiver_derived_remote_refs'],
                         reference_only[2]['known_receiver_derived_remote_refs'])
        self.assertEqual(reference_only[2]['counts']['new_acquired_remote_refs'], 0)
        self.assertEqual(reference_only[2]['counts']['new_receiver_derived_remote_refs'], 0)
        self.assertEqual(len(reference_only[2]['new_receipt_ids']), 1)
        self.assertFalse(reference_only[2]['tensor_changed_from_previous'])

        peer = window('peer')
        prior_only = audit_structured_episode(self.episode(
            empty_window('ego'), peer, max_entities=1, max_forecasts=1,
            tools=('F', 'F')))[2]
        prior_only_forecasts = [ref for ref in prior_only['prior_only_refs']
                                if ref['field_kind'] == 'forecast']
        self.assertEqual(len(prior_only_forecasts), 1)
        self.assertNotIn(prior_only_forecasts[0], prior_only['direct_primary_refs'])

    def test_role_transition_reports_pf_drop_and_fp_prior_dependency(self):
        from evaluation.structured import audit_structured_episode
        history = dict(states=[[0., 0., 0.]] * 11, valid=[True] * 11,
            times=(np.arange(-10, 1) / 10.).tolist(), read_paths=[])
        for tools in (('P', 'F'), ('F', 'P')):
            with self.subTest(tools=tools):
                ep = self.episode(empty_window('ego'), one_track('peer', .1, y=0.),
                                  tools=tools, ego_history=history)
                original = copy.deepcopy(ep)
                rows = audit_structured_episode(ep)
                self.assertEqual(ep, original, 'derived audit must not mutate archived input')
                self.assertEqual(ep['status'], 'completed')
                self.assertEqual(rows[-1]['exclusion_reason_version'],
                                 'toolv2x_structured_exclusion_reasons_v1')
                details = {x['ref']['field_kind']: x for x in rows[-1]['non_direct_primary_fields']
                           if x['source_role'] == 'remote'}
                self.assertEqual(set(details), {'history', 'forecast'})
                self.assertEqual({x['reason'] for x in details.values()}, {'ego_filter'})
                self.assertTrue(details['history']['admitted_dependency'])
                self.assertFalse(details['history']['dropped'])
                self.assertEqual(details['forecast']['dropped'], tools == ('P', 'F'))
                self.assertEqual(details['forecast']['admitted_dependency'], tools == ('F', 'P'))
                self.assertEqual(details['forecast']['legacy_reason'],
                                 'structured_capacity' if tools == ('P', 'F') else None)
                self.assertEqual(rows[-1]['direct_remote_primary_refs'], [])
                if tools == ('F', 'P'):
                    entity0 = ep['plans'][1]['prepared']['entities'][0]
                    entity1 = ep['plans'][2]['prepared']['entities'][0]
                    self.assertEqual(entity0['role'], 'unresolved')
                    self.assertEqual(entity1['role'], 'ego')
                    self.assertEqual(entity0['aliases'], entity1['aliases'])
                    self.assertEqual(entity0['forecast_sets'], entity1['forecast_sets'])
                    self.assertEqual([x['field_kind'] for x in rows[1]['direct_remote_primary_refs']],
                                     ['forecast'])
                    self.assertIn(details['forecast']['ref'], rows[-1]['prior_dependency_closure_refs'])

    def test_forecast_exclusions_distinguish_entity_and_forecast_set_capacity(self):
        from evaluation.structured import audit_structured_episode
        for peer_x, entities, forecasts, reason in (
                (30., 1, 4, 'entity_capacity'),
                (10.2, 2, 1, 'forecast_set_capacity')):
            with self.subTest(reason=reason):
                ep = self.episode(one_track('ego', 10.), one_track('peer', peer_x, 99),
                    tools=('F',), max_entities=entities, max_forecasts=forecasts)
                rows = audit_structured_episode(ep)
                item = next(x for x in rows[-1]['non_direct_primary_fields']
                            if x['source_role'] == 'remote' and x['ref']['field_kind'] == 'forecast')
                self.assertEqual(item['reason'], reason)
                self.assertTrue(item['dropped'])
                self.assertFalse(item['admitted_dependency'])
                self.assertEqual(item['legacy_reason'], 'structured_capacity')

    def test_corrupt_locations_and_receipts_are_rejected_and_old_got_stays_unbound(self):
        from evaluation.framework import evaluate_method_task
        from evaluation.structured import audit_structured_episode
        ep = self.episode(one_track('ego', 10.), one_track('peer', 30.), tools=('P',))
        bad_location = copy.deepcopy(ep)
        group = next(group for group in bad_location['plans'][1]['prepared'][
            'admission_report']['field_groups'] if group['tensor_locations'])
        group['tensor_locations'][0]['source_slot'] = 99
        bad_location['plans'][1]['output']['prepared_input'] = copy.deepcopy(
            bad_location['plans'][1]['prepared'])
        with self.assertRaises(ValueError):
            audit_structured_episode(bad_location)
        bad_receipt = copy.deepcopy(ep)
        bad_receipt['ledger_snapshots'][1]['receipts'][0]['wire_text'] = '{}'
        with self.assertRaises(ValueError):
            audit_structured_episode(bad_receipt)

        events = []
        from tools.vehicle import VehicleTools
        old_driver = EvidenceDriver(events)
        old_local, old_peer = window('ego'), empty_window('peer')
        old_local['scene'] = old_peer['scene'] = SCENE
        old_service = VehicleTools(lambda: old_peer, self.predictor, SCENE, 10, 'peer',
                                   task_provenance=provenance())
        old_ep = __import__('planning.method_episode', fromlist=['run_task_episode']).run_task_episode(
            old_local, self.predictor(old_local), dict(speed_mps=2., yaw_rate_rps=0.),
            dict(active_agent_mask=np.array([[[True], [False]]], dtype=bool)),
            old_service, self.predictor, old_driver,
            lambda state: dict(tool='STOP', mode=None, reason='fixture_complete'), limits(),
            local_provenance=provenance(), sample_id='sample', branch_id='old_got',
            token_counter=lambda prompt: len(prompt) // 4)
        old = evaluate_method_task(self.task(old_ep), None)
        self.assertTrue(old['task_success'])
        self.assertIsNone(old['structured_audit'])


if __name__ == '__main__':
    unittest.main()
