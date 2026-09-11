"""Offline-only scene/label audit and gallery. Never imported by the online runner."""
import argparse
from collections import Counter, defaultdict
import html
import json
from pathlib import Path
import pickle
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from common import v2v4real_meta as M
from planning.inputs import parse_q8, parse_q9
from audit_training_evidence import ACTIONS, coverage, save


def read(path):
    return json.loads(Path(path).read_text())


def distribution(values):
    a = np.asarray(values, dtype=float)
    return dict(n=len(a), minimum=float(a.min()), median=float(np.median(a)),
                mean=float(a.mean()), maximum=float(a.max())) if len(a) else dict(n=0)


def check_window(window, frames, g, pose):
    """Reproject existing world tracks with solve, without calling make_window."""
    ids = window['track_ids'].tolist()
    if not np.allclose(window['time_seconds'], np.arange(-10, 1) / 10, atol=1e-7, rtol=0):
        raise AssertionError('history time axis mismatch')
    errors = dict(xyz_m=0., yaw_rad=0., dimension_m=0., score=0.)
    inverse_basis = np.linalg.solve(pose, np.eye(4))
    yaw_shift = np.arctan2(inverse_basis[1, 0], inverse_basis[0, 0])
    for h, past in enumerate(range(g - 10, g + 1)):
        lookup = {int(r[0]): r for r in frames[past]}
        for i, track in enumerate(ids):
            if bool(window['valid'][i, h]) != (track in lookup):
                raise AssertionError('history presence mismatch')
            if track not in lookup:
                continue
            original = np.asarray(lookup[track], dtype=float)
            expected = np.linalg.solve(pose, np.r_[original[1:4], 1.])[:3]
            state = window['states'][i, h].astype(float)
            errors['xyz_m'] = max(errors['xyz_m'], float(np.max(np.abs(state[:3] - expected))))
            yaw_error = (state[6] - original[7] - yaw_shift + np.pi) % (2 * np.pi) - np.pi
            errors['yaw_rad'] = max(errors['yaw_rad'], abs(float(yaw_error)))
            errors['dimension_m'] = max(errors['dimension_m'], float(np.max(np.abs(state[3:6] - original[4:7]))))
            errors['score'] = max(errors['score'], abs(float(window['scores'][i, h]) - original[8]))
    errors['within_tolerance'] = (errors['xyz_m'] <= 1e-3 and errors['yaw_rad'] <= 5e-6 and
                                errors['dimension_m'] <= 1e-6 and errors['score'] <= 1e-6)
    return errors


def motion_geometry(points):
    deltas = np.diff(np.vstack([np.zeros(2), points]), axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    headings = np.unwrap(np.arctan2(deltas[:, 1], deltas[:, 0]))
    raw = float(np.degrees(headings[-1] - headings[0]))
    reliable = bool(lengths.sum() >= 1 and lengths.min() >= .1)
    return dict(path_length_3s_m=float(lengths.sum()), path_heading_reliable=reliable,
                raw_path_heading_change_deg=raw, heading_change_deg=raw if reliable else None,
                future_segment_speeds_mps=(lengths * 2).tolist())


def independent_label(g, label):
    """Recompute poses and native class semantics independently of ego_label."""
    pose = np.asarray(M.load_pose('train', g), dtype=float)
    end = M.seq_of(g, 'train')[3]
    points, valid = [], []
    pose_headings = [np.arctan2(pose[1, 0], pose[0, 0])]
    for offset in (5, 10, 15, 20, 25, 30):
        available = g + offset < end and Path(M.pose_dir('train'), '%04d_lidar_pose.npy' % (g + offset)).is_file()
        valid.append(available)
        future = np.asarray(M.load_pose('train', g + offset), dtype=float) if available else None
        if future is not None:
            pose_headings.append(np.arctan2(future[1, 0], future[0, 0]))
        points.append(np.linalg.solve(pose, future[:, 3])[:2].tolist() if available else None)
    if valid != label['valid']:
        raise AssertionError('offline label availability mismatch')
    if not all(valid):
        return dict(valid_points=sum(valid), all_valid=False, independently_recomputed_waypoints=points)
    points = np.array(points)
    error = float(np.max(np.abs(points - np.asarray(label['waypoints']))))
    # The native mean of six consecutive displacements telescopes to endpoint / 6.
    distance = float(np.linalg.norm(points[-1]) / 6)
    angle = float(np.degrees(np.arctan2(points[-1, 1], points[-1, 0])))
    speed = next((name for threshold, name in ((8, 'fast'), (4, 'moderate'), (2, 'slow'), (.1, 'very slow'))
                  if distance > threshold), 'stop')
    steering = ('straight' if distance < .5 or -5 <= angle <= 5 else
                'left' if angle <= -15 else 'slightly left' if angle < -5 else
                'right' if angle >= 15 else 'slightly right')
    parsed = parse_q8(label['target_q8'])
    rounding_error = float(np.max(np.abs(np.asarray(parse_q9(label['target_q9'])) - points)))
    return dict(valid_points=6, all_valid=True, coordinate_error_m=error,
        q9_rounding_error_m=rounding_error, native_classes_match=parsed == dict(speed=speed, steering=steering),
        speed=speed, steering=steering, endpoint_angle_deg=angle,
        pose_heading_change_deg=float(np.degrees(np.unwrap(pose_headings)[-1] - pose_headings[0])),
        displacement_3s_m=float(np.linalg.norm(points[-1])),
        independently_recomputed_waypoints=points.tolist(), **motion_geometry(points))


def scan_pose_continuity(out, rows):
    """Read-only diagnostic of published train pose steps, never a cleaning rule."""
    manifest = read(ROOT / 'outputs/protocol_audit/split_manifest.json')
    candidates = read(out / 'manifest.json')['candidates']
    jumps, steps = [], []
    for scene in manifest['train_scenes']:
        seq = manifest['source_scenes']['train'].index(scene)
        _, start, end = M.seq_ranges('train')[seq]
        previous = np.asarray(M.load_pose('train', start), dtype=float)
        for g in range(start + 1, end):
            current = np.asarray(M.load_pose('train', g), dtype=float)
            distance = float(np.linalg.norm(current[:3, 3] - previous[:3, 3]))
            delta = np.arctan2(current[1, 0], current[0, 0]) - np.arctan2(previous[1, 0], previous[0, 0])
            yaw = float(np.degrees((delta + np.pi) % (2 * np.pi) - np.pi))
            step = dict(scene=scene, from_g=g - 1, to_g=g, translation_m=distance, yaw_change_deg=yaw)
            steps.append(step)
            if distance > 20 or abs(yaw) > 20:
                jumps.append(step)
            previous = current
    affected = []
    for row in candidates:
        history = [j['to_g'] for j in jumps if j['scene'] == row['scene'] and row['g'] - 10 < j['to_g'] <= row['g']]
        future = [j['to_g'] for j in jumps if j['scene'] == row['scene'] and row['g'] < j['to_g'] <= row['g'] + 30]
        if history or future:
            affected.append(dict(sample_id=row['sample_id'], g=row['g'], history_jump_frames=history,
                                 future_3s_jump_frames=future))
    ids = {r['sample_id'] for r in rows}
    case = next(r for r in rows if r['g'] == 2762)
    seq = M.seq_of(case['g'], 'train')[0]
    cache_case = []
    for source, short in (('no_fusion', 'ego'), ('no_fusion_cav1', 'peer')):
        with (ROOT / 'outputs/tracks/train' / (source + '_world.pkl')).open('rb') as handle:
            tracks = pickle.load(handle)
        for g in range(2759, 2765):
            directory = Path(M.config_dir(source, 'train'))
            scores = np.load(str(directory / ('%04d_pred_score.npy' % g)))
            cache_case.append(dict(source=short, g=g, raw_detections=len(scores),
                detections_at_least_0_2=int((scores >= .2).sum()), tracks=len(tracks['seqs'][seq][g])))
    result = dict(scope='research train only; diagnostic, no online changes or automatic filtering',
        nominal_dt_seconds=.1, actual_sensor_timestamps_verified=False,
        threshold_translation_m=20, threshold_yaw_deg=20, adjacent_steps=len(steps),
        max_translation_m=max(r['translation_m'] for r in steps), jumps=jumps,
        affected_train_frames=affected, affected_train_count=len(affected), train_candidates=len(candidates),
        affected_selected=[r for r in affected if r['sample_id'] in ids],
        selected_count=64, g2762_source_counts=cache_case,
        interpretation='pose/cache discontinuity; exact cause and tracker association not replayed')
    save(out / 'offline/pose_continuity.json', result)
    return result


def near_origin_diagnostic(out, rows):
    entries = []
    for row in rows:
        for action in ('P', 'F', 'PF'):
            frame = out / row['directory'] / action
            full, used = read(frame / 'evidence_full.json'), read(frame / 'evidence_used.json')
            peers = [o for o in full['objects'] if o['source'] != 'ego']
            near = {o['track_id'] for o in peers if np.linalg.norm(o['box'][:2]) < 1.5}
            retained = {o['track_id'] for o in used['objects'] if o['source'] != 'ego'}
            associated = {r['peer_id'] for r in full['relations']}
            outside_roi_unassociated = {o['track_id'] for o in peers if o['track_id'] not in near | associated
                and 0 <= o['box'][0] <= 70 and -20 <= o['box'][1] <= 20}
            entries.append(dict(audit_index=row['audit_index'], sample_id=row['sample_id'], action=action,
                near_origin_peer_ids=sorted(near), retained_near_origin_peer_ids=sorted(near & retained),
                retained_near_origin_records=sum(o['source'] != 'ego' and o['track_id'] in near for o in used['objects']),
                roi_unassociated_outside_origin_total=len(outside_roi_unassociated),
                roi_unassociated_outside_origin_retained=len(outside_roi_unassociated & retained)))
    result = dict(radius_m=1.5, identity_confirmed=False, online_evidence_modified=False,
        scope='geometric sensitivity check; near origin may be ego observation, not confirmed identity or uselessness',
        entries=entries, summaries={a: dict(
            frames_with_retained_near_origin=sum(bool(e['retained_near_origin_peer_ids']) for e in entries if e['action'] == a),
            retained_near_origin_records=sum(e['retained_near_origin_records'] for e in entries if e['action'] == a),
            roi_unassociated_outside_origin_total=sum(e['roi_unassociated_outside_origin_total'] for e in entries if e['action'] == a),
            roi_unassociated_outside_origin_retained=sum(e['roi_unassociated_outside_origin_retained'] for e in entries if e['action'] == a))
            for a in ('P', 'F', 'PF')})
    save(out / 'offline/near_origin.json', result)
    return result


def scene_audit(out):
    manifest = read(out / 'manifest.json')
    if read(out / 'prepare_status.json')['completed'] != 64:
        raise RuntimeError('scene audit needs all 64 prepared frames; failures remain in denominator')
    offline = out / 'offline'
    offline.mkdir(exist_ok=False)
    shutil.copy2(__file__, offline / 'report_source.py')
    labels = {r['sample_id']: r for r in map(json.loads,
        (ROOT / 'outputs/adaptation_data_v1/offline_labels/train.jsonl').read_text().splitlines())}
    worlds = {}
    for source in ('no_fusion', 'no_fusion_cav1'):
        with (ROOT / 'outputs/tracks/train' / (source + '_world.pkl')).open('rb') as handle:
            worlds[source] = pickle.load(handle)
        if (worlds[source]['meta']['frame'] != 'world' or worlds[source]['meta']['split'] != 'train'
                or worlds[source]['meta']['config'] != source):
            raise AssertionError('world-track provenance mismatch')
    rows = []
    for selected in manifest['frames']:
        frame = out / selected['directory']
        connection = read(frame / 'connection.json')
        access = read(frame / 'input_access.json')
        projection = read(frame / 'projection.json')
        g = selected['g']
        pose = np.asarray(M.load_pose('train', g), dtype=float)
        seq, _, start, _ = M.seq_of(g, 'train')
        if (g - 10 < start or access['online_gt_fields'] is not False or
                connection['actions']['Ego']['remote_archive_reads'] != 0 or
                any(int(Path(p).name.split('_')[0]) not in (g, g - 1)
                    for p in access['ego_feature_paths'] + access['ego_motion_paths'])):
            raise AssertionError('online time/source access mismatch')
        if projection['shape'][1] != 540 or projection['input_frames'] != [g, g - 1] or not projection['finite']:
            raise AssertionError('unexpected actual point feature input')
        audits, history = {}, {}
        for short, source in (('ego', 'no_fusion'), ('peer', 'no_fusion_cav1')):
            with np.load(str(frame / (short + '_window.npz'))) as window:
                audits[short] = check_window(window, worlds[source]['seqs'][seq], g, pose)
                history[short] = dict(targets=len(window['track_ids']),
                    valid_states=int(window['valid'].sum()), possible_states=int(window['valid'].size),
                    short_history_targets=int((window['valid'].sum(axis=1) < 2).sum()))
        previous = np.asarray(M.load_pose('train', g - 1), dtype=float)
        speed = float(np.linalg.norm(pose[:2, 3] - previous[:2, 3]) * 10)
        heading = np.arctan2(pose[1, 0], pose[0, 0]) - np.arctan2(previous[1, 0], previous[0, 0])
        yaw_rate = float(((heading + np.pi) % (2 * np.pi) - np.pi) * 10)
        motion_errors = dict(speed_mps=abs(speed - connection['ego_motion']['speed_mps']),
                             yaw_rate_rps=abs(yaw_rate - connection['ego_motion']['yaw_rate_rps']))
        counts, modes = {}, {}
        for action in ACTIONS:
            full = read(frame / action / 'evidence_full.json')
            counts[action] = {}
            for format_name, name in (('compact', 'evidence_used.json'), ('json', 'evidence_used_json.json')):
                used = read(frame / action / name)
                counts[action][format_name] = coverage(full, used)
                expected = read(frame / 'audit_capacity.json')[action][format_name]
                if expected != counts[action][format_name]:
                    raise AssertionError('capacity recomputation mismatch')
                native = [o for o in used['objects'] if o['model_used']]
                unique_counts = [len(np.unique(np.asarray(o['forecast']).reshape(6, -1), axis=0)) for o in native]
                modes[action + '_' + format_name] = dict(native_records=len(native),
                    native_six_distinct=sum(n == 6 for n in unique_counts),
                    fallback_records=len(used['objects']) - len(native))
            # Original tool-cost ledger counts actual bytes, independent of prompt rounding.
            for packet in read(frame / action / 'tool_costs.json'):
                wire_bytes = len((frame / action / (packet['tool'] + '_response.json')).read_bytes())
                if wire_bytes != packet['response_bytes']:
                    raise AssertionError('wire byte mismatch')
        feature_detections = [len(np.load(p)) for p in access['ego_feature_paths'] if p.endswith('_detection_box_score.npy')]
        rows.append(dict(selected, motion=connection['ego_motion'], history=history,
            coordinate_checks=audits, motion_errors=motion_errors,
            feature_detections=feature_detections, feature_summary=read(frame / 'feature_summary.json'),
            labels=independent_label(g, labels[selected['sample_id']]),
            capacity=counts, retained_modes=modes,
            tool_bytes={a: connection['actions'][a]['response_bytes'] for a in ACTIONS},
            same_information=read(frame / 'audit_array_equivalence.json')))
    save(offline / 'frames.json', rows)
    classes = [parse_q8(r['target_q8']) for r in labels.values() if r['target_q8'] is not None]
    summary = dict(frames=len(rows), recordings=len({r['recording'] for r in rows}),
        full_train_label_classes={key: dict(Counter(r[key] for r in classes)) for key in ('speed', 'steering')},
        sample_label_classes={key: dict(Counter(r['labels'].get(key, 'invalid') for r in rows)) for key in ('speed', 'steering')},
        current_speed_mps=distribution([r['motion']['speed_mps'] for r in rows]),
        source_targets={k: distribution([r['history'][k]['targets'] for r in rows]) for k in ('ego', 'peer')},
        coordinate_checks_passed=sum(all(c['within_tolerance'] for c in r['coordinate_checks'].values()) for r in rows),
        max_history_xyz_error_m=max(c['xyz_m'] for r in rows for c in r['coordinate_checks'].values()),
        max_history_yaw_error_rad=max(c['yaw_rad'] for r in rows for c in r['coordinate_checks'].values()),
        max_motion_speed_error_mps=max(r['motion_errors']['speed_mps'] for r in rows),
        max_motion_yaw_error_rps=max(r['motion_errors']['yaw_rate_rps'] for r in rows),
        all_six_label_frames=sum(r['labels']['all_valid'] for r in rows),
        native_classes_match=sum(r['labels'].get('native_classes_match', False) for r in rows),
        unreliable_path_heading_frames=sum(not r['labels'].get('path_heading_reliable', False) for r in rows),
        max_label_coordinate_error_m=max(r['labels'].get('coordinate_error_m', 0) for r in rows),
        feature_box_truncation_frames=sum(any(r['feature_summary']['truncated_boxes']) for r in rows),
        capacity={}, retained_modes={}, history={})
    for k in ('ego', 'peer'):
        summary['history'][k] = {field: sum(r['history'][k][field] for r in rows) for field in rows[0]['history'][k]}
    for action in ACTIONS:
        summary['capacity'][action] = {}
        for fmt in ('compact', 'json'):
            entries = [r['capacity'][action][fmt] for r in rows]
            summary['capacity'][action][fmt] = dict(
                sums={key: sum(e[key] for e in entries) for key in entries[0]},
                distributions={key: distribution([e[key] for e in entries]) for key in entries[0]},
                frames_with_peers=sum(e['remote_unique_total'] > 0 for e in entries),
                frames_with_peers_but_none_retained=sum(e['remote_unique_total'] > 0 and e['remote_unique_retained'] == 0 for e in entries),
                frames_with_roi_candidates=sum(e['roi_unassociated_total'] > 0 for e in entries),
                frames_with_roi_candidates_but_none_retained=sum(e['roi_unassociated_total'] > 0 and e['roi_unassociated_retained'] == 0 for e in entries))
            key = action + '_' + fmt
            summary['retained_modes'][key] = {k: sum(r['retained_modes'][key][k] for r in rows) for k in rows[0]['retained_modes'][key]}
    save(offline / 'scene_summary.json', summary)
    scan_pose_continuity(out, rows)
    near_origin_diagnostic(out, rows)
    make_figures(out, rows)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def make_figures(out, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Polygon, Rectangle
    from matplotlib.lines import Line2D
    font = Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
    if font.is_file():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    chosen = {}
    for group in sorted({r['recording'] for r in rows}):
        group_rows = [r for r in rows if r['recording'] == group]
        row = min(group_rows, key=lambda r: abs(r['history']['ego']['targets'] - np.median([x['history']['ego']['targets'] for x in group_rows])))
        chosen[row['audit_index']] = '该录制组的中等自车密度样本'
    diagnostics = [
        ('自车与邻车目标最多', max(rows, key=lambda r: sum(r['history'][k]['targets'] for k in ('ego', 'peer')))),
        ('F 丢弃的前方未关联候选最多', max(rows, key=lambda r: r['capacity']['F']['compact']['roi_unassociated_total'] - r['capacity']['F']['compact']['roi_unassociated_retained'])),
        ('有足够位移且离线姿态航向变化最大', max((r for r in rows if r['labels'].get('path_heading_reliable')),
                                                key=lambda r: abs(r['labels']['pose_heading_change_deg']))),
        ('当前跟踪输出最少；需结合原始检测检查', min(rows, key=lambda r: sum(r['history'][k]['targets'] for k in ('ego', 'peer'))))]
    for reason, row in diagnostics:
        chosen.setdefault(row['audit_index'], reason)
    figures = out / 'offline/figures'
    figures.mkdir()
    cases = []
    for index, reason in chosen.items():
        row = next(r for r in rows if r['audit_index'] == index)
        frame = out / row['directory']
        full = read(frame / 'F/evidence_full.json')
        points = np.array(row['labels']['independently_recomputed_waypoints'])
        xy = np.array([o['box'][:2] for o in full['objects']] + [[0, 0]] + points.tolist())
        xlim = (min(-15., float(xy[:, 0].min()) - 5), max(75., float(xy[:, 0].max()) + 5))
        ylim = (min(-25., float(xy[:, 1].min()) - 5), max(25., float(xy[:, 1].max()) + 5))
        fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
        for ax, action in zip(axes.flat, ('all', 'P', 'F', 'PF')):
            used = full if action == 'all' else read(frame / action / 'evidence_used.json')
            for obj in full['objects']:
                x, y, _, length, width, _, yaw = obj['box']
                peer = obj['source'] != 'ego'
                present = action == 'all' or any(o['track_id'] == obj['track_id'] and (o['source'] != 'ego') == peer for o in used['objects'])
                color = ('#dc7619' if peer else '#2879b5') if present else '#d0d4d8'
                corners = np.array([[length, width], [length, -width], [-length, -width], [-length, width]]) / 2
                rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
                ax.add_patch(Polygon(corners @ rotation.T + [x, y], fill=False, edgecolor=color,
                                     linewidth=1.4 if present else .6))
                if present:
                    prefix = 'R' if peer else 'E'
                    ax.text(x, y, prefix + str(obj['track_id']), fontsize=6, color=color)
            if action != 'all':
                for obj in used['objects']:
                    color = '#dc7619' if obj['source'] != 'ego' else '#2879b5'
                    # Capability records remain separate; plotted modes are not confidence regions.
                    for mode in obj['forecast']:
                        path = np.array(mode)
                        ax.plot(path[:, 0], path[:, 1], color=color, alpha=.22, linewidth=.6)
                    if 'history' in obj:
                        path = np.array(obj['history'])[np.array(obj['history_valid'], dtype=bool)]
                        ax.plot(path[:, 0], path[:, 1], color=color, linestyle=':', linewidth=1)
            ax.add_patch(Rectangle((0, -20), 70, 40, fill=False, edgecolor='#718096', linestyle='--', linewidth=.7))
            ax.scatter([0], [0], marker='>', s=90, color='black', zorder=5)
            path = np.vstack([np.zeros(2), points])
            ax.plot(path[:, 0], path[:, 1], 'o-', color='#278342', markersize=3, linewidth=1.7)
            if action == 'all':
                title = '完整当前跟踪：自车 %d，邻车 %d' % (row['history']['ego']['targets'], row['history']['peer']['targets'])
            else:
                c = row['capacity'][action]['compact']
                title = '%s 入模文字：%d 条记录，%d 个邻车目标' % (action, c['objects_retained'], c['remote_unique_retained'])
            ax.set(title=title, xlim=xlim, ylim=ylim, xlabel='x / 米（向前为正）', ylabel='y / 米（向右为正）')
            ax.set_aspect('equal', adjustable='box')
            ax.grid(alpha=.15)
        fig.suptitle('%02d | %s | g=%d\n%s；框来自检测/因果跟踪，绿色未来仅供离线核查' %
                     (index, row['recording'].replace('testoutput_CAV_data_', ''), row['g'], reason), fontsize=12)
        handles = [Line2D([0], [0], color=c, label=l) for c, l in
                   (('#2879b5', '自车来源'), ('#dc7619', '邻车来源'), ('#d0d4d8', '未保留的文字目标'), ('#278342', '离线自车未来'))]
        axes[0, 0].legend(handles=handles, fontsize=7, loc='upper left')
        name = '%02d_g%d.png' % (index, row['g'])
        fig.savefig(figures / name, dpi=140)
        plt.close(fig)
        cases.append(dict(audit_index=index, reason=reason, figure='figures/' + name))
    save(out / 'offline/cases.json', cases)


def final_report(out):
    from transformers import AutoTokenizer
    from planning.v2vgot import CHECKPOINT, prompt_tokens
    rows = read(out / 'offline/frames.json')
    near = near_origin_diagnostic(out, rows)
    continuity = read(out / 'offline/pose_continuity.json')
    labels = {r['sample_id']: r for r in map(json.loads,
        (ROOT / 'outputs/adaptation_data_v1/offline_labels/train.jsonl').read_text().splitlines())}
    tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT), use_fast=False, local_files_only=True)
    summary, answer_rows = {}, []
    for init in ('got', 'llm'):
        directory = out / ('q8_' + init)
        status = read(directory / 'status.json')
        if status['attempted'] != 256:
            raise RuntimeError('all 256 attempts per initialization are required')
        results = []
        for row in rows:
            label = labels[row['sample_id']]
            for action in ACTIONS:
                result = read(directory / ('%02d_%s.json' % (row['audit_index'], action)))
                if result['sample_id'] != row['sample_id'] or result['action'] != action or result['q9_executed']:
                    raise AssertionError('generation provenance mismatch')
                parsed = None
                if result['q8_executed']:
                    try:
                        parsed = parse_q8(result['q8_raw'])
                    except ValueError:
                        pass
                if (parsed is not None) != (result['status'] == 'parsed_q8'):
                    raise AssertionError('independent Q8 parse disagreement')
                if parsed is not None:
                    count = len(prompt_tokens(tokenizer, result['q9_prompt'])) - 1 + 540
                    if count != result['q9_input_tokens']:
                        raise AssertionError('actual-parent Q9 budget mismatch')
                    target_count = len(tokenizer(label['target_q9'], add_special_tokens=False).input_ids) + 1
                    result['q9_supervised_tokens'] = count + target_count
                    result['q9_supervision_fits'] = count + target_count <= 4096
                    result['matches_offline_q8_label'] = parsed == parse_q8(label['target_q8'])
                results.append(result)
        summary[init] = dict(expected=256, executed=sum(r['q8_executed'] for r in results),
            parsed=sum(r['status'] == 'parsed_q8' for r in results),
            execution_failures=sum(r['status'] == 'execution_failed' for r in results),
            actual_parent_overflows=sum(r.get('q9_budget_fits') is False for r in results),
            q9_supervision_overflows=sum(r.get('q9_supervision_fits') is False for r in results),
            per_action={action: dict(parsed=sum(r['status'] == 'parsed_q8' for r in results if r['action'] == action),
                                    attempted=64) for action in ACTIONS},
            generated_classes={key: dict(Counter(r['parsed_q8'][key] for r in results if r['status'] == 'parsed_q8'))
                               for key in ('speed', 'steering')},
            joint_class_counts=dict(Counter(r['parsed_q8']['speed'] + ' / ' + r['parsed_q8']['steering']
                                           for r in results if r['status'] == 'parsed_q8')),
            matches_offline_label=sum(r.get('matches_offline_q8_label', False) for r in results),
            frames_with_different_parsed_actions=sum(len({tuple(sorted(r['parsed_q8'].items()))
                for r in results if r['sample_id'] == row['sample_id'] and r['status'] == 'parsed_q8'}) > 1 for row in rows),
            generation_seconds=sum(r.get('q8_cost', {}).get('seconds', 0) for r in results),
            q9_prompt_tokens=distribution([r['q9_input_tokens'] for r in results if 'q9_input_tokens' in r]))
        answer_rows.extend(results)
    save(out / 'offline/q8_summary.json', summary)
    save(out / 'offline/q8_label_checks.json', answer_rows)
    cases = read(out / 'offline/cases.json')
    parts = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>ToolV2X 64 帧核查</title>',
             '<style>body{font:17px/1.7 sans-serif;max-width:1250px;margin:40px auto;padding:0 20px;color:#172636}img{width:100%}table{border-collapse:collapse}td,th{padding:6px 12px;border:1px solid #ccd4dc}pre{white-space:pre-wrap;background:#eef2f6;padding:12px}section{margin:50px 0}small{color:#536273}</style>',
             '<h1>这一轮究竟在验证什么</h1><p>自车保留当前和上一帧的点特征。P 请求邻车历史，自车用 MTR 重算预测；F 请求邻车用同一 MTR 算好的预测；PF 同时拿到两种记录。下面比较完整信息与真正放进驾驶提示的文字证据。</p>',
             '<p>每组固定八帧，共 64 帧。图中的框来自检测/因果跟踪，不是真实相机图片。灰色表示文字记录未进入提示；自车点特征仍在。绿色是离线观察到的自车未来，在线工具和驾驶生成均没有读取它。虚线框为固定前方区域。</p>',
             '<p>R 编号是邻车内部轨迹号，E 编号是自车内部轨迹号，相同数字不表示同一目标。几何未关联也不等于已证明遮挡。彩色细线是所保留的六模态预测，P 的虚点线为可用历史；图外预测线按当前场景视野裁切，完整数值仍保留在证据文件中。</p>',
             '<p>GoT Q8 可解析 %d/256；V2V-LLM %d/256。本轮不执行 Q9 轨迹生成和参数更新。可解析只表示能作为下一问的父回答。</p>' % (summary['got']['parsed'], summary['llm']['parsed'])]
    parts += ['<p><strong>本轮发现：</strong>训练片段内 %d 个相邻 pose 间隔中，%d 处超过诊断阈值（20米或20度），影响 %d/3027 个候选帧的历史或3秒未来，固定抽样中涉及 %d/64 帧。阈值只定位异常，不是自动清洗标准。</p>' %
              (continuity['adjacent_steps'], len(continuity['jumps']), continuity['affected_train_count'], len(continuity['affected_selected'])),
              '<p>另有 %d/64 帧的 F 文字证据保留了距自车原点1.5米内的邻车来源目标，可能包含邻车对自车的观测；尚未确认身份。因此“邻车来源目标数”不能直接解释成新增外部车辆数，也不能据此断定这些信息无用。</p>' % near['summaries']['F']['frames_with_retained_near_origin'],
              '<p>低位移时路径方向会受定位抖动支配。本图集的转向例要求3秒累计位移≥1米、每半秒段≥0.1米，并用未来pose的航向变化选择。此条件只用于选图，64帧统计和监督标签未改。</p>']
    parts.append('<p><strong>格式通过也不等于理解了场景：</strong>GoT 有 %d/256 次回答“fast / straight”，而样本中包含停车和转弯。四动作之间出现不同可解析类别的帧只有 %d/64。这里使用的是尚未适配 ToolV2X 驾驶输入的发布初始化，不能据此判断最终协作收益。</p>' %
                 (summary['got']['joint_class_counts'].get('fast / straight', 0), summary['got']['frames_with_different_parsed_actions']))
    for case in cases:
        row = next(r for r in rows if r['audit_index'] == case['audit_index'])
        parts += ['<section><h2>%02d：%s</h2>' % (row['audit_index'], html.escape(case['reason'])),
                  '<p>%s，当前速度 %.2f m/s；离线标签 %s / %s。</p>' %
                  (html.escape(row['sample_id']), row['motion']['speed_mps'], row['labels'].get('speed'), row['labels'].get('steering')),
                  '<img src="%s" alt="当前信息与 P/F/PF 文字证据对比">' % case['figure'],
                  '<table><tr><th>动作</th><th>完整→保留记录</th><th>邻车唯一目标</th><th>前方未关联候选</th></tr>']
        for action in ACTIONS:
            c = row['capacity'][action]['compact']
            parts.append('<tr><td>%s</td><td>%d → %d</td><td>%d → %d</td><td>%d → %d</td></tr>' %
                (action, c['objects_total'], c['objects_retained'], c['remote_unique_total'], c['remote_unique_retained'],
                 c['roi_unassociated_total'], c['roi_unassociated_retained']))
        parts.append('</table><p><small>PF 可为同一邻车目标保留两种记录；记录数与唯一轨迹数分开计算。</small></p>')
        if row['g'] == 2762:
            parts.append('<p><strong>这不是自然空场景：</strong>该帧自车和邻车的原始检测各有11个目标，但跟踪输出均为空；此前g2760→2761的pose平移67.34米，名义间隔0.1秒。缓存与跟踪确认规则支持“跳变后重新确认”的解释，尚未重放内部匹配，也未查明跳变来源。该帧三秒净位移仅0.023米，原路径角度626度为低位移诊断失真，不能当作真实转向。</p>')
        for init in ('got', 'llm'):
            answer = next(a for a in answer_rows if a['audit_index'] == row['audit_index'] and a['action'] == 'PF' and a['initialization'] == init)
            parts.append('<details><summary>%s 的 PF 原始 Q8：%s</summary><pre>%s</pre></details>' %
                         (init, answer['status'], html.escape(answer.get('q8_raw', answer.get('error', '')))))
        parts.append('</section>')
    parts.append('</html>')
    (out / 'offline/gallery.html').write_text('\n'.join(parts))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--phase', choices=('scene', 'final'), required=True)
    args = parser.parse_args()
    if args.phase == 'scene':
        scene_audit(args.out)
    else:
        final_report(args.out)
