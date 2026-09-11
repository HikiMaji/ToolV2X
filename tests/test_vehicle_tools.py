"""Tool boundaries: lazy remote access, full prediction context and wire validity."""
import copy
import json
import sys
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from tools.vehicle import VehicleTools, decode_response, history_window, make_evidence


def window(source='peer'):
    states = np.zeros((2, 11, 7), np.float32)
    states[:, :, 0] = [[2.] * 11, [20.] * 11]
    states[:, :, 3:6] = [4., 2., 1.5]
    return dict(source=source, scene='scene', g=10, track_ids=np.array([7, 9]), states=states,
                valid=np.ones((2, 11), bool), scores=np.ones((2, 11), np.float32),
                time_seconds=np.arange(-10, 1) / 10.)


def fixture_prediction(w):
    # Test-only stand-in at the expensive model boundary; never used by a runner.
    means = np.repeat(w['states'][:, None, -1:, :2], 6, axis=1)
    means = np.repeat(means, 50, axis=2)
    means[:, :, :, 1] = len(w['track_ids'])
    return dict(track_ids=w['track_ids'], states=w['states'][:, -1], means=means,
                scores=np.full((len(means), 6), 1 / 6, np.float32), model_used=np.ones(len(means), bool))


class VehicleToolTests(unittest.TestCase):
    def test_no_remote_read_before_query_and_roi_does_not_remove_model_context(self):
        reads, model_contexts = [], []
        def load():
            reads.append('peer')
            return window()
        def predict(w):
            model_contexts.append(len(w['track_ids']))
            return fixture_prediction(w)
        tools = VehicleTools(load, predict, 'scene', 10, provider='peer')
        self.assertEqual(reads, [])
        response = tools.query('F', [0., -5., 5., 5.])
        packet = decode_response(response['wire'], 'scene', 10)
        self.assertEqual(model_contexts, [2])
        self.assertEqual(len(packet['objects']), 1)
        self.assertEqual(packet['objects'][0]['track_id'], 7)
        self.assertEqual(packet['objects'][0]['forecast'][0][0][1], 2.)
        self.assertEqual(response['cost']['response_bytes'], len(response['wire']))
        self.assertEqual(packet['forecast_times'], [.5, 1., 1.5, 2., 2.5, 3.])

    def test_full_p_history_round_trips_for_same_information_control(self):
        w = window()
        tools = VehicleTools(lambda: w, fixture_prediction, 'scene', 10, provider='peer')
        response = tools.query('P')
        p = decode_response(response['wire'], 'scene', 10)
        reconstructed = history_window(p)
        for key in ('states', 'valid', 'scores', 'track_ids', 'time_seconds'):
            np.testing.assert_allclose(reconstructed[key], w[key])
        f = decode_response(tools.query('F')['wire'], 'scene', 10)
        e = make_evidence(window('ego'), fixture_prediction(window('ego')), [p, f], fixture_prediction)
        self.assertEqual(len(e['objects']), 6)
        self.assertEqual({o['source'] for o in e['objects']}, {'ego', 'peer:P_local', 'peer:F'})
        self.assertTrue(all(len(o['forecast']) == 6 for o in e['objects']))

    def test_receiver_rejects_future_mismatched_and_extra_payload_fields(self):
        response = VehicleTools(lambda: window(), fixture_prediction, 'scene', 10, provider='peer').query('P')
        with self.assertRaises(ValueError):
            decode_response(response['wire'], 'scene', 9)
        p = json.loads(response['wire'])
        p['objects'][0]['gt_future'] = [99.]
        with self.assertRaises(ValueError):
            decode_response(json.dumps(p).encode(), 'scene', 10)
        p = json.loads(response['wire'])
        p['objects'][0]['history_times'][-1] = .1
        with self.assertRaises(ValueError):
            decode_response(json.dumps(p).encode(), 'scene', 10)

    def test_provider_checks_query_and_keeps_empty_distinct_from_observed_free_space(self):
        tools = VehicleTools(lambda: window(), fixture_prediction, 'scene', 10, provider='peer')
        with self.assertRaises(ValueError):
            tools.query('P', [5, 0, 0, 1])
        packet = decode_response(tools.query('P', [50, 50, 60, 60])['wire'], 'scene', 10)
        self.assertEqual(packet['status'], 'no_observed_targets')
        self.assertEqual(packet['coverage'], 'not_established')

    def test_evidence_rejects_different_record_even_with_same_frame_number(self):
        response = VehicleTools(lambda: window(), fixture_prediction, 'scene', 10, provider='peer').query('P')
        packet = json.loads(response['wire'])
        packet['scene'] = 'different_recording'
        with self.assertRaises(ValueError):
            make_evidence(window('ego'), fixture_prediction(window('ego')), [packet], fixture_prediction)

    def test_provider_never_caches_a_window_from_another_record_or_time(self):
        for field, value in (('scene', 'different_recording'), ('g', 11)):
            bad = dict(window(), **{field: value})
            tools = VehicleTools(lambda: bad, fixture_prediction, 'scene', 10, provider='peer')
            for _ in range(2):
                with self.assertRaises(ValueError):
                    tools.query('F')

    def test_cost_separates_new_model_fallback_and_roi_returned_targets(self):
        def predict(w):
            result = fixture_prediction(w)
            result['model_used'] = np.array([True, False])
            return result

        tools = VehicleTools(lambda: window(), predict, 'scene', 10, provider='peer')
        # Target counts must not depend on the timer resolving a short operation.
        with patch('tools.vehicle.perf_counter', return_value=5.):
            first = tools.query('F', [0., -5., 5., 5.])
            cached = tools.query('F')
            perception = tools.query('P')
        self.assertEqual(first['cost']['model_targets_computed'], 1)
        self.assertEqual(first['cost']['fallback_targets_computed'], 1)
        self.assertEqual(first['cost']['returned_targets'], 1)
        self.assertFalse(first['cost']['within_decision_forecast_cache_hit'])
        self.assertEqual(cached['cost']['model_targets_computed'], 0)
        self.assertEqual(cached['cost']['fallback_targets_computed'], 0)
        self.assertEqual(cached['cost']['returned_targets'], 2)
        self.assertTrue(cached['cost']['within_decision_forecast_cache_hit'])
        self.assertEqual(perception['cost']['model_targets_computed'], 0)
        self.assertEqual(perception['cost']['fallback_targets_computed'], 0)
        self.assertFalse(perception['cost']['within_decision_forecast_cache_hit'])


if __name__ == '__main__':
    unittest.main()
