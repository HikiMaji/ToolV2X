"""P/F services use source-local causal track states and an injected predictor.

history_valid marks available tracker output, including prediction-maintained
states on missed detections; it is not a per-frame detection-match mask.
"""
import json
from time import perf_counter
import numpy as np
from probe.kinematic_tools import encode, history_packet
from planning.inputs import validate_evidence

VERSION = 'toolv2x_pf_v1'
TIMES = [.5, 1., 1.5, 2., 2.5, 3.]
FUTURE_INDICES = [4, 9, 14, 19, 24, 29]
COMMON = {'track_id', 'box', 'score'}
P_FIELDS = COMMON | {'history', 'history_valid', 'history_scores', 'history_times'}
F_FIELDS = COMMON | {'forecast', 'forecast_scores', 'model_used'}
PACKET_FIELDS = {'version', 'tool', 'provider', 'scene', 'g', 'coordinate_frame', 'roi',
                 'status', 'coverage', 'objects', 'forecast_times'}


def check_roi(roi):
    if roi is None:
        return
    a = np.asarray(roi, dtype=float)
    if a.shape != (4,) or not np.isfinite(a).all() or not (a[0] < a[2] and a[1] < a[3]):
        raise ValueError('ROI must be finite [xmin, ymin, xmax, ymax]')


def in_roi(box, roi):
    return roi is None or (roi[0] <= box[0] <= roi[2] and roi[1] <= box[1] <= roi[3])


def prediction_objects(window, prediction):
    history_packet(window)
    count = len(window['track_ids'])
    if (not np.array_equal(window['track_ids'], prediction['track_ids']) or
            np.shape(prediction['means']) != (count, 6, 50, 2) or
            np.shape(prediction['scores']) != (count, 6) or
            np.shape(prediction['model_used']) != (count,) or
            not np.allclose(prediction['states'], window['states'][:, -1])):
        raise ValueError('predictor target/time alignment mismatch')
    return [dict(track_id=int(track_id), box=window['states'][i, -1].tolist(),
        score=float(window['scores'][i, -1]),
        forecast=prediction['means'][i][:, FUTURE_INDICES, :].tolist(),
        forecast_scores=prediction['scores'][i].tolist(), model_used=bool(prediction['model_used'][i]))
        for i, track_id in enumerate(window['track_ids'])]


class VehicleTools:
    def __init__(self, window_loader, predictor, scene, g, provider='no_fusion_cav1'):
        self._load = window_loader
        self._predict = predictor
        self.scene, self.g, self.provider = scene, int(g), provider
        self._window = None
        self._forecast = None

    def query(self, tool, roi=None):
        if tool not in ('P', 'F'):
            raise ValueError('unknown tool')
        check_roi(roi)
        roi = None if roi is None else np.asarray(roi, dtype=float).tolist()
        request = dict(version=VERSION, tool=tool, provider=self.provider, scene=self.scene, g=self.g, roi=roi)
        begin = perf_counter()
        if self._window is None:
            window = self._load()
            history_packet(window)
            if window.get('scene') != self.scene or window['source'] != self.provider or window['g'] != self.g:
                raise ValueError('provider window scene/source/time mismatch')
            self._window = window
        w = self._window
        model_seconds = 0.
        model_targets, fallback_targets = 0, 0
        cache_hit = self._forecast is not None
        if tool == 'P':
            objects = [dict(track_id=int(track_id), box=w['states'][i, -1].tolist(),
                score=float(w['scores'][i, -1]), history=w['states'][i].tolist(),
                history_valid=w['valid'][i].tolist(), history_scores=w['scores'][i].tolist(),
                history_times=w['time_seconds'].tolist()) for i, track_id in enumerate(w['track_ids'])]
        else:
            if self._forecast is None:
                start = perf_counter()
                # Full-frame prediction preserves every target's original neighbour context.
                # ponytail: predicts all current targets once; query-target batching can remove
                # unused output computation without changing context in a later measured update.
                self._forecast = self._predict(w)
                model_seconds = perf_counter() - start
            objects = prediction_objects(w, self._forecast)
            if not cache_hit:
                model_targets = sum(obj['model_used'] for obj in objects)
                fallback_targets = len(objects) - model_targets
        objects = [obj for obj in objects if in_roi(obj['box'], roi)]
        packet = dict(request, coordinate_frame='ego_at_t', objects=objects,
            status='ok' if objects else 'no_observed_targets', coverage='not_established',
            forecast_times=TIMES if tool == 'F' else [])
        wire = encode(packet)
        decode_response(wire, self.scene, self.g)
        return dict(request=request, wire=wire, cost=dict(request_bytes=len(encode(request)),
            response_bytes=len(wire), service_seconds=perf_counter() - begin,
            model_seconds=model_seconds, model_targets_computed=model_targets,
            fallback_targets_computed=fallback_targets, returned_targets=len(objects),
            within_decision_forecast_cache_hit=cache_hit if tool == 'F' else False))


def decode_response(wire, scene, g):
    p = json.loads(wire)
    if (set(p) != PACKET_FIELDS or p['version'] != VERSION or p['tool'] not in ('P', 'F') or
            p['scene'] != scene or p['g'] != g or isinstance(p['g'], bool) or
            not isinstance(p['g'], int) or p['coordinate_frame'] != 'ego_at_t' or
            not isinstance(p['provider'], str) or not p['provider'] or
            p['coverage'] != 'not_established' or not isinstance(p['objects'], list)):
        raise ValueError('invalid source/time/packet contract')
    check_roi(p['roi'])
    if p['status'] != ('ok' if p['objects'] else 'no_observed_targets'):
        raise ValueError('inconsistent response status')
    if p['forecast_times'] != (TIMES if p['tool'] == 'F' else []):
        raise ValueError('invalid forecast time axis')
    ids = []
    for obj in p['objects']:
        if set(obj) != (P_FIELDS if p['tool'] == 'P' else F_FIELDS):
            raise ValueError('unexpected response object fields')
        box = np.asarray(obj['box'], dtype=float)
        if (box.shape != (7,) or (box[3:6] <= 0).any() or not 0 <= obj['score'] <= 1 or
                not in_roi(box, p['roi']) or isinstance(obj['track_id'], bool) or not isinstance(obj['track_id'], int)):
            raise ValueError('invalid target anchor or identity')
        ids.append(obj['track_id'])
        if p['tool'] == 'P':
            history = np.asarray(obj['history'])
            valid = np.asarray(obj['history_valid'])
            scores = np.asarray(obj['history_scores'])
            times = np.asarray(obj['history_times'])
            if (history.shape != (11, 7) or valid.shape != (11,) or valid.dtype != bool or
                    not valid[-1] or scores.shape != (11,) or (scores < 0).any() or (scores > 1).any() or
                    times.shape != (11,) or not np.allclose(times, np.arange(-10, 1) / 10.) or
                    not np.allclose(history[-1], box) or not np.isclose(scores[-1], obj['score'])):
                raise ValueError('invalid causal track-state history')
        else:
            scores = np.asarray(obj['forecast_scores'])
            if (np.shape(obj['forecast']) != (6, 6, 2) or scores.shape != (6,) or
                    (scores < 0).any() or scores.sum() > 1.0001 or not isinstance(obj['model_used'], bool)):
                raise ValueError('invalid multimodal forecast')
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate source-local targets')
    encode(p)
    return p


def history_window(packet):
    p = decode_response(encode(packet), packet['scene'], packet['g'])
    if p['tool'] != 'P':
        raise ValueError('track-state history requires P')
    objects, n = p['objects'], len(p['objects'])
    w = dict(source=p['provider'], scene=p['scene'], g=p['g'], track_ids=np.array([o['track_id'] for o in objects], dtype=np.int64),
        states=np.array([o['history'] for o in objects], dtype=np.float32).reshape(n, 11, 7),
        valid=np.array([o['history_valid'] for o in objects], dtype=bool).reshape(n, 11),
        scores=np.array([o['history_scores'] for o in objects], dtype=np.float32).reshape(n, 11),
        time_seconds=np.arange(-10, 1) / 10.)
    history_packet(w)
    return w


def make_evidence(local_window, local_prediction, packets, predictor, predict_p=True):
    if not isinstance(predict_p, bool):
        raise ValueError('P processing must explicitly enable or disable local prediction')
    objects = []
    for obj in prediction_objects(local_window, local_prediction):
        objects.append(dict(obj, source='ego', forecast_times=TIMES))
    queries, peer_anchors = [], {}
    for p in packets:
        p = decode_response(encode(p), local_window['scene'], local_window['g'])
        if p['provider'] == local_window['source']:
            raise ValueError('remote response must come from a different source')
        if p['tool'] == 'P' and predict_p:
            w = history_window(p)
            predicted = prediction_objects(w, predictor(w))
            observed = {o['track_id']: o for o in p['objects']}
            for obj in predicted:
                original = observed[obj['track_id']]
                obj.update(history=original['history'], history_valid=original['history_valid'],
                           history_times=original['history_times'])
        elif p['tool'] == 'P':
            predicted = [{k: v for k, v in obj.items() if k != 'history_scores'} for obj in p['objects']]
        else:
            predicted = p['objects']
        for obj in predicted:
            key = (p['provider'], obj['track_id'])
            if key in peer_anchors and not np.allclose(peer_anchors[key], obj['box']):
                raise ValueError('P/F anchor conflict at identical source/time')
            peer_anchors[key] = obj['box']
            source = p['provider'] + ((':P_local' if predict_p else ':P') if p['tool'] == 'P' else ':F')
            objects.append(dict(obj, source=source, **({'forecast_times': TIMES} if 'forecast' in obj else {})))
        queries.append(dict(tool=p['tool'], roi=p['roi'], response_bytes=len(encode(p)), status=p['status']))
    relations = []
    for ego in objects:
        if ego['source'] != 'ego':
            continue
        for (provider, peer_id), peer_box in peer_anchors.items():
            distance = float(np.linalg.norm(np.asarray(ego['box'][:2]) - np.asarray(peer_box[:2])))
            ratios = np.asarray(peer_box[3:6]) / np.asarray(ego['box'][3:6])
            if distance <= 3. and (ratios >= .5).all() and (ratios <= 2.).all():
                relations.append(dict(ego_id=ego['track_id'], peer_id=peer_id, distance_m=distance,
                    status='current_geometry_candidate_not_confirmed_identity'))
    result = dict(as_of_g=int(local_window['g']), coordinate_frame='ego_at_t', objects=objects,
                  relations=relations, queries=queries)
    validate_evidence(result)
    return result
