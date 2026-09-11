"""Run real CMP P/F and original V2V-GoT components on a held-out decision frame."""
import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path
from time import perf_counter
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from common import v2v4real_meta as M
from prediction import cmp_adapter as C
from planning.inputs import load_ego_features, load_ego_motion, make_prompt
from planning.v2vgot import CHECKPOINT, UPSTREAM, V2VGoTPlanner, load_projector, project_ego_features, fit_evidence
from tools.vehicle import VehicleTools, decode_response, make_evidence


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def load_window(scene, local_frame, source, paths, archive_loader=None):
    # Validation scenes come from the physical train archive; split_manifest defines holdout.
    path = ROOT / 'outputs/causal_windows_v1/train' / source / (scene + '.pkl')
    if archive_loader is None:
        with open(path, 'rb') as handle:
            artifact = pickle.load(handle)
    else:
        artifact = archive_loader(path)
    meta = artifact['meta']
    if (meta['gt_access'] is not False or meta['coordinate_frame'] != 'ego_at_t' or
            meta['scene'] != scene or meta['split'] != 'train' or meta['yaw_unit'] != 'radian' or meta['dt_seconds'] != .1):
        raise ValueError('unexpected causal window provenance')
    selected = dict(artifact['windows'][local_frame], scene=scene)
    if selected['source'] != source:
        raise ValueError('window source mismatch')
    paths.append(dict(path=str(path), selected_local_frame=local_frame, selected_global_frame=int(selected['g']),
                      source=source, contains_other_windows=True, model_receives_selected_window_only=True))
    return selected


def snapshot_code(out):
    for name in ('src', 'tests'):
        shutil.copytree(ROOT / name, out / 'code_snapshot' / name, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(ROOT / 'vendor/cmp_mtr', out / 'code_snapshot/vendor/cmp_mtr',
                    ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('llava/model/llava_arch.py', 'llava/model/builder.py',
                 'llava/model/multimodal_projector/builder.py', 'llava/model/language_model/llava_llama.py',
                 'llava/constants.py', 'llava/conversation.py', 'llava/mm_utils.py'):
        destination = out / 'code_snapshot/v2vgot_original' / name
        destination.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy2(UPSTREAM / name, destination)


def select_decision(role, scene_index, local_frame):
    if role not in ('train', 'validation') or scene_index < 0:
        raise ValueError('only predefined train/validation frames may be prepared')
    manifest = json.loads((ROOT / 'outputs/protocol_audit/split_manifest.json').read_text())
    if scene_index >= len(manifest[role + '_scenes']):
        raise ValueError('scene index outside requested role')
    scene = manifest[role + '_scenes'][scene_index]
    allowed = json.loads((ROOT / ('outputs/protocol_audit/' + role + '_frames.json')).read_text())
    if [scene, local_frame] not in allowed:
        raise ValueError('frame is not a pre-existing decision frame for this role')
    return scene


def prepare(out, validation_index, local_frame, actions, evidence_format='json', research_split='validation'):
    from transformers import AutoTokenizer
    scene = select_decision(research_split, validation_index, local_frame)
    paths = []
    ego = load_window(scene, local_frame, 'no_fusion', paths)
    g = int(ego['g'])
    features = load_ego_features(M.V2VGOT_ROOT, 'train', g)
    motion, motion_paths = load_ego_motion(M.V2VGOT_ROOT, 'train', g)
    np.savez(str(out / 'ego_features.npz'), **{key: value for key, value in features.items() if isinstance(value, np.ndarray)})
    projector, provenance = load_projector()
    begin = perf_counter()
    tokens = project_ego_features(projector, features)
    projection = dict(provenance, shape=list(tokens.shape), finite=bool(torch.isfinite(tokens).all()),
        seconds=perf_counter() - begin, input_frames=features['frame_indices'],
        feature_read_paths=features['read_paths'], norm=float(tokens.norm()),
        real_dataset_input=True, language_model_executed=False)
    if not projection['finite']:
        raise ValueError('nonfinite original projector output')
    np.save(str(out / 'ego_point_tokens.npy'), tokens.cpu().numpy())
    save_json(out / 'projection.json', projection)
    del projector, tokens
    model, model_info = C.load_model()
    save_json(out / 'cmp_model_loading.json', model_info)
    calls = []
    (out / 'forecasts').mkdir()

    def predictor(window):
        begin = perf_counter()
        prediction = C.predict(model, window)
        path = out / 'forecasts' / ('%03d_%s.npz' % (len(calls), window['source']))
        np.savez(str(path), **prediction)
        calls.append(dict(source=window['source'], g=int(window['g']), targets=len(window['track_ids']),
            original_model_targets=int(prediction['model_used'].sum()),
            seconds=perf_counter() - begin, path=str(path)))
        return prediction

    local_prediction = predictor(ego)
    tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT), use_fast=False, local_files_only=True)
    results, all_evidence = {}, {}
    for action in actions:
        directory = out / action
        directory.mkdir()
        reads_before, calls_before = len(paths), len(calls)
        service = VehicleTools(lambda: load_window(scene, local_frame, 'no_fusion_cav1', paths),
                               predictor, scene, g)
        responses, packet_costs = [], []
        for tool in (() if action == 'Ego' else tuple(action)):
            response = service.query(tool)
            (directory / (tool + '_response.json')).write_bytes(response['wire'])
            save_json(directory / (tool + '_request.json'), response['request'])
            responses.append(decode_response(response['wire'], scene, g))
            packet_costs.append(dict(tool=tool, **response['cost']))
        evidence = make_evidence(ego, local_prediction, responses, predictor)
        view, selection = fit_evidence(tokenizer, motion, evidence, projection['shape'][1], evidence_format=evidence_format)
        save_json(directory / 'evidence_full.json', evidence)
        save_json(directory / 'evidence_used.json', view)
        save_json(directory / 'context_selection.json', selection)
        (directory / 'q8_prompt.txt').write_text(make_prompt('Q8', motion, view, evidence_format=evidence_format) + '\n')
        save_json(directory / 'tool_costs.json', packet_costs)
        remote_reads = paths[reads_before:]
        if action == 'Ego' and (remote_reads or packet_costs):
            raise AssertionError('Ego branch accessed remote evidence')
        retained_remote = sum(obj['source'] != 'ego' for obj in view['objects'])
        results[action] = dict(evidence_objects=len(evidence['objects']), remote_archive_reads=len(remote_reads),
            predictor_calls=len(calls) - calls_before, selected_remote_records=retained_remote,
            context_selection=selection, request_bytes=sum(c['request_bytes'] for c in packet_costs),
            response_bytes=sum(c['response_bytes'] for c in packet_costs), q9_executed=False)
        all_evidence[action] = evidence
        print(json.dumps(dict(action=action, objects=len(evidence['objects']),
              retained=len(view['objects']), retained_remote=retained_remote,
              response_bytes=results[action]['response_bytes'])), flush=True)
    equivalence = {'checked': False}
    if 'PF' in all_evidence:
        objs = all_evidence['PF']['objects']
        left = {o['track_id']: o for o in objs if o['source'].endswith(':P_local')}
        right = {o['track_id']: o for o in objs if o['source'].endswith(':F')}
        if left.keys() != right.keys():
            raise AssertionError('full-context target mismatch')
        difference = max([float(np.max(np.abs(np.array(left[i]['forecast']) - np.array(right[i]['forecast'])))) for i in left] or [0.])
        score_difference = max([float(np.max(np.abs(np.array(left[i]['forecast_scores']) - np.array(right[i]['forecast_scores'])))) for i in left] or [0.])
        equivalence = dict(checked=True, target_count=len(left), max_waypoint_difference=difference,
                           max_score_difference=score_difference, equal_within_1e_5=max(difference, score_difference) <= 1e-5)
        if not equivalence['equal_within_1e_5']:
            raise AssertionError('same-information local/remote prediction mismatch')
    save_json(out / 'same_information.json', equivalence)
    save_json(out / 'input_access.json', dict(ego_feature_paths=features['read_paths'], ego_motion_paths=motion_paths,
        window_archives=paths, online_gt_fields=False, archive_selection='causal selected window only'))
    save_json(out / 'predictor_calls.json', calls)
    metadata = dict(scene=scene, local_frame=local_frame, g=g, physical_split='train', research_split=research_split,
        ego_motion=motion, actions=results, evidence_format=evidence_format, inputs_prepared=True,
        actual_original_cmp_executed=any(call['original_model_targets'] > 0 for call in calls),
        original_cmp_target_evaluations=sum(call['original_model_targets'] for call in calls),
        actual_original_projector_executed=True, language_model_executed=False,
        training=False, quality_evaluation=False, closed_loop=False, full_framework_complete=False)
    save_json(out / 'connection.json', metadata)
    return features, metadata


def infer(out, features, metadata, decoding='q8_q9'):
    planner = V2VGoTPlanner(evidence_format=metadata['evidence_format'],
                           adapter_directory=out / 'llava-toolv2x-lora-ego')
    save_json(out / 'v2vgot_model_loading.json', planner.provenance)
    metadata['language_model_executed'] = False
    metadata['decoding'] = decoding
    for action in metadata['actions']:
        evidence = json.loads((out / action / 'evidence_full.json').read_text())
        result = planner.plan(features, metadata['ego_motion'], evidence, decoding=decoding)
        save_json(out / action / 'plan.json', result)
        metadata['language_model_executed'] |= result['language_model_executed']
        metadata['actions'][action]['plan_status'] = result['status']
        metadata['actions'][action]['q9_executed'] = result['q9_executed']
        save_json(out / 'connection.json', metadata)
        print(json.dumps(dict(action=action, status=result['status'], q8=result['q8_raw'],
                             q9=result.get('q9_raw'))), flush=True)
    metadata['all_actions_parsed'] = all(row.get('plan_status') == 'parsed' for row in metadata['actions'].values())
    metadata['scope'] = 'fixed-action original-model connection; unadapted driving checkpoint, no task-quality claim'
    save_json(out / 'connection.json', metadata)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    parser.add_argument('--mode', choices=('prepare', 'infer'), default='prepare')
    parser.add_argument('--evidence-format', choices=('json', 'compact'), default='json')
    parser.add_argument('--decoding', choices=('q8_q9', 'direct'), default='q8_q9')
    parser.add_argument('--research-split', choices=('train', 'validation'), default='validation')
    parser.add_argument('--scene-index', '--validation-index', dest='validation_index', type=int, default=0)
    parser.add_argument('--local-frame', type=int, default=10)
    parser.add_argument('--actions', nargs='+', choices=('Ego', 'P', 'F', 'PF'), default=['Ego', 'P', 'F', 'PF'])
    args = parser.parse_args()
    if len(args.actions) != len(set(args.actions)):
        parser.error('actions must be unique')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    snapshot_code(args.out)
    try:
        features, metadata = prepare(args.out, args.validation_index, args.local_frame, args.actions,
                                     args.evidence_format, args.research_split)
        if args.mode == 'infer':
            infer(args.out, features, metadata, args.decoding)
        save_json(args.out / 'execution.json', dict(status='completed', mode=args.mode,
            language_model_executed=metadata['language_model_executed'],
            all_actions_parsed=metadata.get('all_actions_parsed'), full_framework_complete=False))
    except Exception as exc:
        save_json(args.out / 'execution.json', dict(status='failed', mode=args.mode,
            error_type=type(exc).__name__, error=str(exc), full_framework_complete=False))
        raise


if __name__ == '__main__':
    main()
