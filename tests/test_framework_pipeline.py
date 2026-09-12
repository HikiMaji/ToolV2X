"""Check exported task supervision and failure-aware whole-episode evaluation."""
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


class PipelineTests(unittest.TestCase):
    def test_preparation_resume_reuses_only_complete_frames_and_rejects_drift(self):
        from planning.run_framework import completed_prefix
        from planning.inputs import make_prompt
        policies = ['Ego', 'P']
        motion = dict(speed_mps=1.,yaw_rate_rps=0.)
        rows = [dict(sample_id='s:%d'%g, scene='s', role='train', g=g, ego_motion=motion,
                     feature_read_paths=['/features/%d'%g], motion_read_paths=['/motion/%d'%g]) for g in (10,11)]
        config = dict(source_data='/data', per_recording=0, policies=policies, p_processing='local_mtr',
                      input_layout='source_blocks_v1', mtr_checkpoint='/model', context_limit=4096, peer_reserve=1536)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'config.json').write_text(json.dumps(config))
            (root/'selected_index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            (root/'progress.json').write_text(json.dumps(dict(frames=1,tasks=2,status='running')))
            feature = root/'features.npz'
            feature.write_bytes(b'existing feature placeholder')
            evidence = dict(as_of_g=10, coordinate_frame='ego_at_t', objects=[], queries=[])
            prompt = make_prompt('Trajectory',motion,evidence,evidence_format='compact')
            records=[]
            for policy in policies:
                path=root/(policy+'.json')
                record=dict(rows[0], policy=policy, path=str(path))
                task=dict(rows[0],policy=policy,status='prepared',feature_path=str(feature),
                    episode=dict(status='tools_completed',p_processing='local_mtr'),
                    prepared=dict(input_layout='source_blocks_v1',evidence_used=evidence,remote_evidence_used=None,q9_prompt=prompt))
                path.write_text(json.dumps(task))
                records.append(record)
            # A write interrupted midway through the next frame must not invalidate the complete prefix.
            (root/'tasks.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records)+'{"unfinished":')
            actual, metadata = completed_prefix(root, rows, config)
            self.assertEqual(actual, records)
            self.assertEqual(metadata['reused_frames'],1)
            self.assertEqual(metadata['unreused_task_lines'],1)
            with self.assertRaisesRegex(ValueError,'configuration'):
                completed_prefix(root,rows,dict(config,peer_reserve=0))
            with self.assertRaisesRegex(ValueError,'index'):
                completed_prefix(root,list(reversed(rows)),config)
            original=json.loads((root/'P.json').read_text())
            altered=json.loads((root/'P.json').read_text())
            altered['ego_motion']['speed_mps'] = 3.
            altered['prepared']['q9_prompt'] = make_prompt('Trajectory',altered['ego_motion'],evidence,evidence_format='compact')
            (root/'P.json').write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError,'causal input'):
                completed_prefix(root,rows,config)
            (root/'P.json').write_text(json.dumps(original))
            altered=json.loads((root/'P.json').read_text())
            altered['prepared']['q9_prompt'] += ' modified'
            (root/'P.json').write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError,'prompt'):
                completed_prefix(root,rows,config)

    def test_incomplete_generation_cannot_be_published_as_completed(self):
        from evaluation.framework import validate_generation_run
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [dict(sample_id='s0', role='validation', scene='r0', g=1,
                            policy=p, path='/'+p+'/task.json') for p in ('Ego', 'F')]
            (root/'selected_tasks.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            (root/'config.json').write_text(json.dumps(dict(tasks=2)))
            generated = [dict(records[0], generation_path='/generation.json')]
            (root/'generations.jsonl').write_text(json.dumps(generated[0])+'\n')
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                validate_generation_run(root)
            (root/'completion.json').write_text(json.dumps(dict(status='completed', attempts=2)))
            with self.assertRaisesRegex(ValueError, 'coverage'):
                validate_generation_run(root)
            generated.append(dict(records[1], generation_path='/generation_F.json'))
            (root/'generations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in generated))
            self.assertEqual(validate_generation_run(root), generated)
            generated[1]['path'] = '/edited_task.json'
            (root/'generations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in generated))
            with self.assertRaisesRegex(ValueError, 'identity'):
                validate_generation_run(root)

    def test_sampling_counts_physical_recordings_across_scene_segments(self):
        from planning.run_framework import select_rows
        stem = 'testoutput_CAV_data_2022-03-15-10-09-50'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'online_index').mkdir()
            rows = [dict(sample_id=stem+'_%d:10'%segment, role='train', physical_split='train',
                         scene=stem+'_%d'%segment, g=segment*100+10) for segment in range(3)]
            (root/'online_index/train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            (root/'online_index/validation.jsonl').write_text('')
            self.assertEqual(len(select_rows(root, per_recording=2)), 2)

    def test_offline_export_preserves_prompt_and_rejects_role_or_frame_mismatch(self):
        path = ROOT / 'scripts/prepare_framework_training.py'
        self.assertTrue(path.is_file(), 'episode-to-supervision exporter missing')
        spec = importlib.util.spec_from_file_location('framework_export_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from planning.inputs import make_prompt
        evidence = dict(as_of_g=10, coordinate_frame='ego_at_t', objects=[], queries=[])
        motion = dict(speed_mps=4., yaw_rate_rps=0.)
        prompt = make_prompt('Trajectory', motion, evidence, evidence_format='compact')
        prepared = dict(input_layout='source_blocks_v1', decoding='direct', q8_executed=False,
                        evidence_used=evidence, remote_evidence_used=None,
                        evidence_selection=dict(evidence_format='compact'), q9_prompt=prompt)
        task = dict(sample_id='scene:10', scene='scene', g=10, role='train', policy='Ego',
                    feature_path='/features.npz', ego_motion=motion, status='prepared', prepared=prepared)
        label = dict(sample_id='scene:10', g=10, role='train', target_q9='six points', valid=[True]*6)
        result = module.training_row(task, label, '/task.json')
        self.assertEqual(result['prompt'], prompt)
        self.assertEqual(result['input_layout'], 'source_blocks_v1')
        for invalid in (dict(label, g=11), dict(label, role='validation'), dict(label, sample_id='other:10')):
            with self.assertRaises(ValueError):
                module.training_row(task, invalid, '/task.json')

    def test_training_order_balances_actions_and_resume_does_not_replay_rows(self):
        self.assertTrue((ROOT / 'src/planning/train_driver.py').is_file(), 'formal training loop missing')
        module = importlib.import_module('planning.train_driver')
        rows = [dict(sample_id='scene:%d' % frame, action=action)
                for frame in range(3) for action in ('Ego', 'P', 'F', 'PF', 'rule')]
        order = module.epoch_order(rows, seed=20, epoch=0)
        self.assertEqual(sorted(order), list(range(15)))
        self.assertEqual(order, module.epoch_order(rows, seed=20, epoch=0))
        # Every frame supplies every condition before the next frame is consumed.
        for begin in range(0, 15, 5):
            chosen = [rows[i] for i in order[begin:begin+5]]
            self.assertEqual(len({r['sample_id'] for r in chosen}), 1)
            self.assertEqual({r['action'] for r in chosen}, {'Ego', 'P', 'F', 'PF', 'rule'})
        self.assertEqual(order[8:], module.epoch_order(rows, 20, 0, start=8))

    def test_metric_summary_preserves_invalid_attempts_and_pairs_only_valid_answers(self):
        self.assertTrue((ROOT / 'src/evaluation/framework.py').is_file(), 'complete episode evaluator missing')
        module = importlib.import_module('evaluation.framework')
        rows = [dict(sample_id=sample, scene='recording', policy=policy, valid=valid,
                     ADE3=ade, FDE3=ade, calls=int(policy != 'Ego'), request_bytes=10*int(policy != 'Ego'),
                     response_bytes=100*int(policy != 'Ego'), stop_reason='stopped')
                for sample, policy, valid, ade in [('s0','Ego',True,2.), ('s0','F',True,1.),
                    ('s1','Ego',True,3.), ('s1','F',False,None)]]
        result = module.summarize_episodes(rows)
        self.assertEqual(result['policies']['F']['attempts'], 2)
        self.assertEqual(result['policies']['F']['failures'], 1)
        self.assertEqual(result['paired_vs_ego']['F']['both_valid'], 1)
        self.assertEqual(result['paired_vs_ego']['F']['mean_ADE3_difference'], -1.)
        self.assertEqual(result['paired_vs_ego']['F']['ego_valid_peer_failed'], 1)
        self.assertEqual(result['policies']['Ego']['mean_response_bytes'], 0.)


if __name__ == '__main__':
    unittest.main()
