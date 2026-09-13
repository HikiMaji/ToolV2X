"""Protocol tests: time-aligned tasks, causal tracking-state proxy and receipts."""
import copy
import importlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from test_vehicle_tools import window, fixture_prediction


def spec_dict(**changes):
    spec = dict(version='toolv2x_execution_v1', profile='contract_v1', sigma_m=5.,
        max_targets=4, max_request_bytes=4096, max_response_bytes=8192,
        max_episode_bytes=24576, ego_geometry='circumscribed_circle',
        ego_length_m=4.8, ego_width_m=2., max_plan_speed_mps=80.,
        max_plan_acceleration_mps2=30.)
    spec.update(changes)
    return spec


def task_request(tool='P', mode='current', request_id='q0', x=2., **spec_changes):
    return dict(version='toolv2x_task_v2', request_id=request_id, provider='peer',
        scene='scene', g=10, coordinate_frame='ego_at_t', tool=tool, mode=mode,
        times=[.5, 1., 1.5, 2., 2.5, 3.], tau_new=[[x, 0.]] * 6,
        tau_old=[[0., 0.]] * 6 if mode == 'change' else None,
        execution_spec=spec_dict(**spec_changes), acquired_field_manifest=[])


def provenance():
    return dict(version='toolv2x_provenance_v1',
        tracking=dict(name='causal_tracker', revision='fixture_v1'),
        prediction=dict(name='context_sensitive_fixture', revision='v1'),
        context=dict(name='causal_tracking_window', revision='fixture_v1'))


def field_ref(kind='history'):
    return dict(provider='peer', scene='scene', g=10, track_handle=7,
        field_kind=kind, producer_version=provenance()['tracking'],
        context_version=provenance()['context'])


class TaskSpecTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT / 'src/tools/task_spec.py').is_file(), 'T1 task specification is missing')
        self.api = importlib.import_module('tools.task_spec')

    def test_execution_spec_roundtrip_and_experimental_parameters(self):
        values = spec_dict(sigma_m=2., max_targets=1, ego_width_m=3.)
        spec = self.api.ExecutionSpec.from_dict(values)
        self.assertEqual(spec.to_dict(), values)
        self.assertEqual(json.loads(json.dumps(spec.to_dict())), values)
        for key, value in [('sigma_m', 0), ('max_targets', True), ('max_response_bytes', -1),
                           ('ego_length_m', float('nan')), ('max_plan_speed_mps', float('inf')),
                           ('version', 'unknown'), ('ego_geometry', 'unimplemented')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.api.ExecutionSpec.from_dict(dict(values, **{key: value}))
        with self.assertRaises(ValueError):
            self.api.ExecutionSpec.from_dict(dict(values, extra=1))

    def test_difference_precedes_time_and_mode_aggregation(self):
        current, change = self.api.relation_scores(
            np.array([[[1., 0.], [0., 0.]]]), np.array([[[0., 1.], [0., 0.]]]))
        np.testing.assert_array_equal(current, [1.])
        np.testing.assert_array_equal(change, [1.])

    def test_near_relation_change_beats_far_distance_change(self):
        old = np.exp(-np.array([2., 100.])[:, None, None] ** 2 / 50.)
        new = np.exp(-np.array([4., 80.])[:, None, None] ** 2 / 50.)
        _, change = self.api.relation_scores(old, new)
        self.assertGreater(change[0], change[1])

    def test_history_gap_uses_actual_dt_and_short_history_is_marked(self):
        w = window()
        w['valid'][:] = False
        w['valid'][:, -1] = True
        w['valid'][0, 6] = True
        w['states'][0, 6, 0] = 0.
        xy, status = self.api.history_proxy(w)
        np.testing.assert_allclose(xy[0, 0, :, 0], [4.5, 7., 9.5, 12., 14.5, 17.])
        np.testing.assert_allclose(xy[1, 0, :, 0], [20.] * 6)
        self.assertEqual(status, ['causal_tracking_state_motion_proxy', 'single_state_static_proxy'])

    def test_tracking_proxy_rejects_future_and_nonboolean_validity(self):
        for mutate in ('future', 'mask', 'ids'):
            w = window()
            if mutate == 'future':
                w['time_seconds'][-1] = 1e-9
            elif mutate == 'mask':
                w['valid'] = w['valid'].astype(int)
            else:
                w['track_ids'] = np.array([7.5, 9.])
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.api.history_proxy(w)

    def test_window_rejects_extra_future_fields_and_inconsistent_eligibility(self):
        w = window()
        w['gt_future'] = [[123., 456.]]
        with self.assertRaises(ValueError):
            self.api.history_proxy(w)
        w = window()
        w['eligible'] = np.array([False, True])
        with self.assertRaises(ValueError):
            self.api.history_proxy(w)

    def test_request_rejects_wrong_time_coordinates_future_and_unchanged_change(self):
        self.api.validate_task_request(task_request(), {})
        for key, value in [('coordinate_frame', 'world'), ('g', True), ('times', [1.] * 6),
                           ('tau_new', [[float('nan'), 0.]] * 6), ('tau_new', [[1., 0.]] * 5),
                           ('tool', 'I'), ('mode', 'unknown')]:
            req = task_request()
            req[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.api.validate_task_request(req, {})
        req = task_request(mode='change')
        req['tau_old'] = req['tau_new']
        with self.assertRaises(ValueError):
            self.api.validate_task_request(req, {})
        with self.assertRaises(ValueError):
            self.api.validate_task_request(dict(task_request(), gt_future=[]), {})

    def test_trajectory_limits_come_from_execution_spec(self):
        req = task_request(x=10., max_plan_speed_mps=30., max_plan_acceleration_mps2=100.)
        self.api.validate_task_request(req, {})
        req['execution_spec']['max_plan_speed_mps'] = 10.
        with self.assertRaises(ValueError):
            self.api.validate_task_request(req, {})

    def test_receipt_is_provider_recorded_and_field_exact(self):
        ref = field_ref()
        receipts = {'issued-receipt': [copy.deepcopy(ref)]}
        req = task_request()
        req['acquired_field_manifest'] = [dict(receipt_id='issued-receipt', ref=ref)]
        self.api.validate_task_request(req, receipts)
        with self.assertRaises(ValueError):
            self.api.validate_task_request(req, {})
        for key, value in [('provider', 'other'), ('scene', 'other'), ('g', 11),
                           ('track_handle', 9), ('field_kind', 'forecast'),
                           ('context_version', dict(name='causal_tracking_window', revision='other'))]:
            bad = copy.deepcopy(req)
            bad['acquired_field_manifest'][0]['ref'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.api.validate_task_request(bad, receipts)
        bad = copy.deepcopy(req)
        bad['acquired_field_manifest'].append(copy.deepcopy(bad['acquired_field_manifest'][0]))
        with self.assertRaises(ValueError):
            self.api.validate_task_request(bad, receipts)

    def test_field_identity_is_structural_and_has_no_request_or_path_identity(self):
        ref = field_ref()
        key = self.api.field_key(ref)
        self.assertEqual(key, self.api.field_key(json.loads(json.dumps(ref))))
        self.assertNotEqual(key, self.api.field_key(dict(ref, provider='other')))
        with self.assertRaises(ValueError):
            self.api.field_key(dict(ref, request_id='q0'))
        bad = copy.deepcopy(ref)
        bad['producer_version']['revision'] = '/absolute/model.pth'
        with self.assertRaises(ValueError):
            self.api.field_key(bad)

    def test_task_changes_ranking_and_ties_ignore_input_order(self):
        w = window()
        first = self.api.rank_targets(w, task_request(x=2.))
        second = self.api.rank_targets(w, task_request(x=20., max_plan_acceleration_mps2=100.))
        self.assertEqual([r['track_handle'] for r in first], [7, 9])
        self.assertEqual([r['track_handle'] for r in second], [9, 7])
        w['states'][1, :, 0] = 2.
        for name in ('states', 'valid', 'scores', 'track_ids'):
            w[name] = w[name][::-1].copy()
        tied = self.api.rank_targets(w, task_request())
        self.assertEqual([r['track_handle'] for r in tied], [7, 9])

    def test_sigma_and_ego_geometry_affect_relation_scores(self):
        w = window()
        narrow = self.api.rank_targets(w, task_request(x=10., sigma_m=1., ego_length_m=1., ego_width_m=1., max_plan_acceleration_mps2=100.))
        wide = self.api.rank_targets(w, task_request(x=10., sigma_m=10., ego_length_m=1., ego_width_m=1., max_plan_acceleration_mps2=100.))
        large = self.api.rank_targets(w, task_request(x=10., sigma_m=1., ego_length_m=20., ego_width_m=2., max_plan_acceleration_mps2=100.))
        self.assertGreater(wide[0]['score'], narrow[0]['score'])
        self.assertEqual(large[0]['score'], 1.)

    def test_f_requires_aligned_full_predictor_output(self):
        w = window()
        req = task_request(tool='F')
        with self.assertRaises(ValueError):
            self.api.rank_targets(w, req)
        forecast = fixture_prediction(w)
        self.assertEqual(self.api.rank_targets(w, req, forecast)[0]['track_handle'], 7)
        forecast['track_ids'] = forecast['track_ids'][::-1]
        with self.assertRaises(ValueError):
            self.api.rank_targets(w, req, forecast)

    def test_geometry_is_stable_across_dtypes_and_large_finite_coordinates(self):
        scores = []
        for dtype in (np.float16, np.float32, np.float64):
            w = window()
            w['states'] = w['states'].astype(dtype)
            scores.append(self.api.rank_targets(w, task_request())[1]['score'])
        self.assertEqual(scores, [scores[0]] * 3)
        w = window()
        w['states'] = w['states'].astype(np.float64)
        w['states'][1, :, 0] = 1e155
        ranks = self.api.rank_targets(w, task_request(sigma_m=1e155))
        self.assertAlmostEqual(ranks[1]['score'], np.exp(-.5), places=11)

    def test_f_fallback_matches_causal_history_and_static_anchor(self):
        for case in ('false_with_history', 'true_without_history', 'moving_fallback'):
            w = window()
            if case != 'false_with_history':
                w['valid'][0, :-1] = False
            forecast = fixture_prediction(w)
            if case != 'true_without_history':
                forecast['model_used'][0] = False
            with self.subTest(case=case), self.assertRaises(ValueError):
                self.api.rank_targets(w, task_request(tool='F'), forecast)

    def test_f_accepts_valid_list_backed_window(self):
        w = window()
        forecast = fixture_prediction(w)
        w['states'] = w['states'].tolist()
        self.assertEqual(self.api.rank_targets(w, task_request(tool='F'), forecast)[0]['track_handle'], 7)

    def test_numeric_arrays_reject_mixed_boolean_leaves(self):
        for value in ([True, 0.], (0., np.bool_(False)), np.array([True, False])):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.api._array(value, (2,))
        np.testing.assert_array_equal(self.api._array([1, 0.], (2,)), [1., 0.])

    def test_causal_window_rejects_boolean_numeric_lists(self):
        for field in ('states', 'scores', 'time_seconds'):
            w = window()
            w[field] = w[field].tolist()
            if field == 'states':
                w[field][0][0][1] = False
            elif field == 'scores':
                w[field][0][0] = True
            else:
                w[field][-1] = False
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.api.history_proxy(w)


if __name__ == '__main__':
    unittest.main()
