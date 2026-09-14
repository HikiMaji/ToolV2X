"""Synthetic CPU contracts for the untrained shared numeric planner."""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import planning.structured_driver as driver
from planning.structured_inputs import StructuredDriverSpec
import test_structured_inputs as task1_fixtures
from tools.task_spec import ExecutionSpec


def tiny_spec(**changes):
    values = StructuredDriverSpec(hidden_dim=16, attention_heads=4,
        interaction_layers=1, max_entities=3,
        max_forecast_sets_per_entity=2).to_dict()
    values.update(changes)
    return StructuredDriverSpec.from_dict(values)


def features(*, previous_active=True, ordered=False):
    if ordered:
        regression = torch.arange(123200, dtype=torch.float32).reshape(
            1, 2, 1, 14, 50, 88)
        classification = 200000. + torch.arange(17600, dtype=torch.float32).reshape(
            1, 2, 1, 2, 50, 88)
    else:
        generator = torch.Generator().manual_seed(17)
        regression = torch.randn((1, 2, 1, 14, 50, 88), generator=generator)
        classification = torch.randn((1, 2, 1, 2, 50, 88), generator=generator)
    return dict(regression_map=regression, classification_map=classification,
        active_agent_mask=np.array([[[True], [previous_active]]], dtype=bool))


def fixture(spec=None, *, empty=False, previous=False, ego_history=None):
    spec = spec or tiny_spec()
    receiver = task1_fixtures.StructuredReceiverTests()
    local = (task1_fixtures.empty_window() if empty else
             task1_fixtures.one_track('ego', 10., 7, score=.8))
    ledger, predictor = receiver.ledger(local)
    if not empty:
        ledger, _ = receiver.acquire(ledger, predictor,
            task1_fixtures.one_track('peer', 10.2, 99, score=.99), tool='F')
    prepared = receiver.build(ledger, spec=spec, ego_history=ego_history)
    if previous:
        parent = prepared['admission_report']['admitted_field_refs'][0]
        prepared = receiver.build(ledger, spec=spec, ego_history=ego_history,
            previous_plan=[[float(i + 1), i / 4.] for i in range(6)],
            previous_parent_refs=[parent])
    return prepared, ledger


def clone(batch):
    return {key: value.clone() for key, value in batch.items()}


class StructuredDriverTests(unittest.TestCase):
    def test_public_api_exists(self):
        for name in ('StructuredPlannerNetwork', 'collate_structured_inputs',
                     'StructuredPlanner', 'validate_numeric_output'):
            with self.subTest(name=name):
                self.assertTrue(hasattr(driver, name), 'missing API: ' + name)

    def test_collator_matches_patch_layout_and_rejects_incompatible_inputs(self):
        spec = tiny_spec()
        prepared, _ = fixture(spec)
        value = features(previous_active=False, ordered=True)
        batch = driver.collate_structured_inputs([value], [prepared])
        merged = torch.cat([value['regression_map'], value['classification_map']], dim=3)
        torch.testing.assert_close(batch['scene_patches'][0, 0, 0],
                                   merged[0, 0, 0, :, :5, :4].reshape(-1))
        self.assertEqual(tuple(batch['scene_patches'].shape), (1, 2, 220, 320))
        self.assertFalse(batch['scene_mask'][0, 1].any())
        self.assertEqual(batch['scene_patches'][0, 1].count_nonzero().item(), 0)

        wrong_shape = features()
        wrong_shape['regression_map'] = wrong_shape['regression_map'][..., :-1, :]
        with self.assertRaises(ValueError):
            driver.collate_structured_inputs([wrong_shape], [prepared])
        other = copy.deepcopy(prepared)
        other['driver_spec']['hidden_dim'] = 32
        with self.assertRaises(ValueError):
            driver.collate_structured_inputs([features(), features()], [prepared, other])

    def test_real_network_is_deterministic_permutation_invariant_and_mask_safe(self):
        torch.manual_seed(23)
        spec = tiny_spec()
        prepared, _ = fixture(spec)
        batch = driver.collate_structured_inputs([features(previous_active=False)], [prepared])
        model = driver.StructuredPlannerNetwork(spec).eval()
        expected = model(batch)
        torch.testing.assert_close(model(batch), expected, rtol=0., atol=0.)
        self.assertEqual(tuple(expected.shape), (1, 6, 2))
        self.assertTrue(torch.isfinite(expected).all())

        changed = clone(batch)
        changed['observations'] = changed['observations'].flip(2)
        changed['observation_mask'] = changed['observation_mask'].flip(2)
        changed['observation_sources'] = changed['observation_sources'].flip(2)
        changed['forecasts'] = changed['forecasts'].flip(3)
        changed['forecast_mask'] = changed['forecast_mask'].flip(3)
        torch.testing.assert_close(model(changed), expected, rtol=1e-5, atol=1e-5)
        changed = clone(batch)
        changed['scene_patches'][~changed['scene_mask']] = 999.
        obs_valid = changed['observation_mask'].unsqueeze(-1).expand_as(changed['observations'])
        future_valid = changed['forecast_mask'].unsqueeze(-1).expand_as(changed['forecasts'])
        changed['observations'][~obs_valid] = 999.
        changed['forecasts'][~future_valid] = -999.
        torch.testing.assert_close(model(changed), expected, rtol=0., atol=0.)

    def test_authentic_pf_build_is_invariant_to_entity_rows_and_distinct_mode_order(self):
        from planning.evidence import apply_response, known_field_manifest, new_ledger
        from tools.task_spec import FrozenPredictor
        from tools.vehicle import VehicleTools

        spec = tiny_spec()
        permutation = np.array([5, 2, 4, 1, 3, 0])

        def make_prediction(order):
            def predict(window):
                value = task1_fixtures.fixture_prediction(window)
                for mode in range(6):
                    value['means'][:, mode, :, 0] += mode * .5
                    value['means'][:, mode, :, 1] += mode * .125
                value['scores'][:] = np.array([.05, .1, .15, .2, .25, .25])
                value['means'] = value['means'][:, order].copy()
                value['scores'] = value['scores'][:, order].copy()
                return value
            return predict

        def build(reverse_rows, mode_order):
            local = task1_fixtures.window('ego')
            remote = task1_fixtures.window('peer')
            if reverse_rows:
                for window in (local, remote):
                    for name in ('track_ids', 'states', 'valid', 'scores'):
                        window[name] = window[name][::-1].copy()
            provenance = task1_fixtures.provenance()
            predictor = FrozenPredictor(make_prediction(mode_order),
                provenance['prediction'], dict(adapter='cmp_causal_window_v1', batch_size=1))
            ledger = new_ledger(local, predictor(local), predictor=predictor,
                local_provenance=provenance, p_processing='observations_only',
                include_local_history=True)
            service = VehicleTools(lambda: remote, predictor, remote['scene'], remote['g'],
                                   remote['source'], task_provenance=provenance)
            ledger = apply_response(ledger,
                service.query_task(task1_fixtures.request('P', 'p0', max_targets=4)), predictor)
            request = task1_fixtures.request('F', 'f0', max_targets=4)
            request['acquired_field_manifest'] = known_field_manifest(ledger)
            ledger = apply_response(ledger, service.query_task(request), predictor)
            prepared = task1_fixtures.StructuredReceiverTests().build(ledger, spec=spec)
            return local, remote, prepared

        local_a, remote_a, prepared_a = build(False, np.arange(6))
        local_b, remote_b, prepared_b = build(True, permutation)
        self.assertFalse(np.array_equal(local_a['track_ids'], local_b['track_ids']))
        self.assertFalse(np.array_equal(remote_a['track_ids'], remote_b['track_ids']))
        self.assertEqual([entity['entity_id'] for entity in prepared_a['entities']],
                         [entity['entity_id'] for entity in prepared_b['entities']])
        batch_a = driver.collate_structured_inputs([features()], [prepared_a])
        batch_b = driver.collate_structured_inputs([features()], [prepared_b])
        torch.testing.assert_close(batch_a['observations'], batch_b['observations'])
        self.assertFalse(torch.equal(batch_a['forecasts'], batch_b['forecasts']))
        torch.manual_seed(31)
        model = driver.StructuredPlannerNetwork(spec).eval()
        torch.testing.assert_close(model(batch_a), model(batch_b), rtol=1e-5, atol=1e-5)

    def test_each_numeric_branch_has_gradient_and_masked_values_do_not(self):
        torch.manual_seed(29)
        spec = tiny_spec()
        ego_history = dict(states=[[0., 0., 0.]] * 11,
            valid=[False] * 10 + [True], times=(np.arange(-10, 1) / 10.).tolist(),
            read_paths=[])
        prepared, _ = fixture(spec, previous=True, ego_history=ego_history)
        value = features(previous_active=False)
        value.update(ego_pose_history=np.zeros((11, 3), np.float32),
            ego_pose_history_valid=np.array([False] * 10 + [True]),
            ego_pose_history_times=np.arange(-10, 1, dtype=np.float32) / 10.)
        batch = driver.collate_structured_inputs([value], [prepared])
        for name in ('scene_patches', 'observations', 'forecasts', 'ego_motion',
                     'ego_history', 'previous_plan'):
            batch[name].requires_grad_()
        model = driver.StructuredPlannerNetwork(spec)
        model(batch).square().sum().backward()
        obs_valid = batch['observation_mask'].unsqueeze(-1).expand_as(batch['observations'])
        future_valid = batch['forecast_mask'].unsqueeze(-1).expand_as(batch['forecasts'])
        for name, mask in (('scene_patches', batch['scene_mask']),
                           ('observations', obs_valid), ('forecasts', future_valid)):
            self.assertGreater(batch[name].grad[mask].abs().sum().item(), 0., name)
            self.assertEqual(batch[name].grad[~mask].count_nonzero().item(), 0, name)
        for source_slot in range(2):
            self.assertGreater(batch['observations'].grad[0, 0, source_slot].abs().sum().item(),
                               0., 'local and paid observation sources must both reach loss')
        for forecast_set in range(2):
            self.assertGreater(batch['forecasts'].grad[0, 0, forecast_set].abs().sum().item(),
                               0., 'local and paid forecast sets must both reach loss')
        self.assertGreater(batch['previous_plan'].grad.abs().sum().item(), 0.)
        self.assertGreater(batch['ego_motion'].grad.abs().sum().item(), 0.)
        self.assertGreater(batch['ego_history'].grad[batch['ego_history_mask']].abs().sum().item(), 0.)
        self.assertGreater(model.prior_encoder[0].weight.grad.abs().sum().item(), 0.)

    def test_empty_and_single_point_batches_train_without_batchnorm_failure(self):
        spec = tiny_spec()
        prepared, _ = fixture(spec, empty=True)
        batch = driver.collate_structured_inputs([features()], [prepared])
        batch['scene_mask'].fill_(False)
        model = driver.StructuredPlannerNetwork(spec).train()
        result = model(batch)
        self.assertTrue(torch.isfinite(result).all())
        result.sum().backward()
        one, _ = fixture(spec)
        batch = driver.collate_structured_inputs([features()], [one])
        batch['observation_mask'].fill_(False)
        batch['observation_mask'][0, 0, 0, 0] = True
        batch['forecast_mask'].fill_(False)
        model.zero_grad(set_to_none=True)
        result = model(batch)
        self.assertTrue(torch.isfinite(result).all())
        result.sum().backward()

    def test_wrapper_preserves_history_provenance_cost_and_numeric_validation(self):
        spec = tiny_spec()
        _, ledger = fixture(spec)
        value = features()
        value.update(ego_pose_history=np.zeros((11, 3), np.float32),
            ego_pose_history_valid=np.array([False] * 10 + [True]),
            ego_pose_history_times=np.arange(-10, 1, dtype=np.float32) / 10.)
        version = dict(name='structured_planner_network', revision='test_v1',
            training=dict(status='initialized_untrained', optimizer_steps=0))
        planner = driver.StructuredPlanner(spec, model_version=version)
        prepared = planner.prepare_input(value, dict(speed_mps=4., yaw_rate_rps=.1),
                                         ledger, spec)
        self.assertEqual(prepared['ego_history_used']['read_paths'], [])
        self.assertEqual(planner.provenance, dict(driver_kind='structured',
            driver_spec=spec.to_dict(), decoding='numeric', model_version=version))
        output = planner.plan_prepared(value, prepared)
        self.assertEqual(set(output), {'output_version', 'driver_kind', 'status',
            'waypoints', 'prepared_input', 'driver_cost', 'parent_refs'})
        self.assertEqual(output['prepared_input'], prepared)
        self.assertEqual(output['parent_refs'],
                         prepared['admission_report']['admitted_field_refs'])
        self.assertEqual(set(output['driver_cost']),
            {'seconds', 'numeric_token_count', 'output_points', 'model_executed'})
        self.assertTrue(output['driver_cost']['model_executed'])
        self.assertGreater(output['driver_cost']['numeric_token_count'], 6)
        execution = ExecutionSpec(max_plan_speed_mps=10000.,
                                  max_plan_acceleration_mps2=10000.)
        self.assertEqual(driver.validate_numeric_output(output, prepared, execution),
                         output['waypoints'])
        bad = copy.deepcopy(output)
        bad['driver_cost']['model_executed'] = False
        with self.assertRaises(ValueError):
            driver.validate_numeric_output(bad, prepared, execution)
        with self.assertRaises(ValueError):
            driver.StructuredPlanner(spec, model_version=dict(name='planner',
                revision='v1', training=dict(status='', optimizer_steps=1)))
        bad = copy.deepcopy(output)
        bad['waypoints'] = np.asarray(bad['waypoints'])
        with self.assertRaises(ValueError):
            driver.validate_numeric_output(bad, prepared, execution)
        with self.assertRaises(ValueError):
            planner.prepare_input(features(), dict(speed_mps=1., yaw_rate_rps=0.),
                                  ledger, tiny_spec(hidden_dim=32))
        with self.assertRaises(ValueError):
            driver.StructuredPlanner(spec, model_version=dict(name='/tmp/model',
                revision='v1', training=dict(status='trained', optimizer_steps=1)))


if __name__ == '__main__':
    unittest.main()
