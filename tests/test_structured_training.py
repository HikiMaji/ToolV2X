"""Tiny CPU training contracts using authentic numeric runtime task archives."""
import copy
import importlib.util
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import test_structured_episode as episode_fixtures
SCENE = episode_fixtures.SCENE
from test_structured_driver import tiny_spec
from planning.method_episode import run_task_episode
from planning.structured_driver import collate_structured_inputs


class StructuredTrainingTests(unittest.TestCase):
    def training(self):
        self.assertIsNotNone(importlib.util.find_spec('planning.train_structured_driver'),
                             'numeric training module is missing')
        from planning import train_structured_driver
        return train_structured_driver

    def archive(self, root, *, failed=False, scene=SCENE, role='train', control='feedback',
                runtime_history=False):
        fixture = episode_fixtures.StructuredEpisodeTests(); fixture.setUp()
        with patch.object(episode_fixtures, 'SCENE', scene):
            if control == 'one_shot':
                bundle=dict(continuation_policy_id='diagnostic',candidate_sources=['initial','slower'],
                    slowdown_scale=.5,max_candidates=2,wrapper_reserve_bytes=4096,extra_generation='self_refinement')
                args=fixture.inputs(control,bundle=bundle)
                from planning.method_controls import diagnostic_bundle_continuation
                args['service'].register_bundle_policy('diagnostic',diagnostic_bundle_continuation)
                args['policy']=lambda state:dict(tool='P',mode='current',reason='bundle')
            else:
                args = fixture.inputs(control)
        history_paths = []
        if runtime_history:
            times = np.arange(-10, 1, dtype=np.float32) / 10.
            args['features'].update(ego_pose_history=np.column_stack((2. * times, times, np.zeros(11))),
                ego_pose_history_valid=np.ones(11, dtype=bool), ego_pose_history_times=times)
            history_paths = ['/synthetic/ego/%04d_lidar_pose.npy' % frame for frame in range(11)]
        args['sample_id'] = scene + ':10'
        if failed == 'invalid_plan':
            with torch.no_grad():
                args['driver'].model.output_head[-1].weight.zero_()
                args['driver'].model.output_head[-1].bias.fill_(1000.)
        elif failed:
            def fail(features, prepared):
                raise ValueError('synthetic untrained failure')
            args['driver'].plan_prepared = fail
        ep = run_task_episode(**args)
        with patch.object(episode_fixtures, 'SCENE', scene):
            task = fixture.task(ep)
        task['row'].update(sample_id=args['sample_id'], role=role, physical_split='train')
        task['row']['local_frame'] = 10
        np.savez(root / 'ego_features.npz', **args['features'])
        task['inputs'].update(feature_path='ego_features.npz', artifact_root=str(root),
                              ego_history_read_paths=history_paths)
        (root / 'task.json').write_text(json.dumps(task))
        (root / 'tasks.jsonl').write_text(json.dumps(dict(sample_id=args['sample_id'], path='task.json'))+'\n')
        label = dict(sample_id=args['sample_id'], role=role, g=10,
                     waypoints=[[i+.123456789, .25] for i in range(6)], valid=[True]*6,
                     times_seconds=[.5,1.,1.5,2.,2.5,3.], label_read_paths=['offline_pose'],
                     scope='offline observed ego-trajectory imitation labels, not optimal planning labels')
        return task, label

    def config(self):
        return dict(version='toolv2x_structured_training_v1', driver_spec=tiny_spec().to_dict(),
                    seed=7, optimizer=dict(name='AdamW', lr=.0001, weight_decay=.01),
                    batch_size=2, epochs=2, refinement_depth=1, save_interval=1, device='cpu')

    def test_masked_smooth_l1_literal_gradient_and_missing_points(self):
        train = self.training()
        predicted = torch.tensor([[[2.,.5],[99.,99.]]], requires_grad=True)
        target = torch.tensor([[[0.,0.],[float('nan'),float('nan')]]])
        valid = torch.tensor([[True,False]])
        loss = train.masked_trajectory_loss(predicted,target,valid)
        self.assertEqual(loss.item(),.8125)
        loss.backward()
        torch.testing.assert_close(predicted.grad,torch.tensor([[[.5,.25],[0.,0.]]]))
        self.assertIsNone(train.masked_trajectory_loss(predicted,target,torch.zeros_like(valid)))

    def test_authentic_export_failed_rows_full_precision_and_tampering(self):
        train = self.training()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); task,label=self.archive(root)
            rows=train.prepare_training_rows(root/'tasks.jsonl',[label])
            self.assertEqual(len(rows),3)
            self.assertEqual([len(r['inputs']['prefix']) for r in rows],[1,2,3])
            self.assertEqual(rows[2]['supervision']['waypoints'],label['waypoints'])
            self.assertNotIn('supervision',rows[0]['inputs'])
            bad=copy.deepcopy(task);bad['episode']['plans'][0]['prepared']['tensor_inputs']['observations'][0][0][0][0]+=1
            with self.assertRaises(ValueError):train.prepare_training_rows([bad],[label])
            old=copy.deepcopy(task);old['episode']['execution_kind']='got'
            with self.assertRaises(ValueError):train.prepare_training_rows([old],[label])
            label['role']='validation'
            with self.assertRaises(ValueError):train.prepare_training_rows([task],[label])
            task,label=self.archive(root,failed=True)
            failed=train.prepare_training_rows(root/'tasks.jsonl',[label])
            self.assertEqual(len(failed),1)
            self.assertEqual(json.loads(Path(failed[0]['source_task']).read_text())['episode']['plans'][0]['status'],'driver_error')
            task,label=self.archive(root,failed='invalid_plan')
            rows=train.prepare_training_rows(root/'tasks.jsonl',[label])
            self.assertEqual(len(rows),1)
            self.assertIsNotNone(task['episode']['plans'][0]['output'])
            self.assertEqual(task['episode']['status'],'invalid_plan')

    def test_runtime_history_audit_exports_and_fits_from_numeric_arrays(self):
        train = self.training()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task, label = self.archive(root, runtime_history=True)
            prepared = task['episode']['plans'][0]['prepared']['ego_history_used']
            self.assertTrue(task['inputs']['ego_history_read_paths'])
            self.assertEqual(prepared['read_paths'], [])
            with np.load(root / 'ego_features.npz') as saved:
                np.testing.assert_allclose(saved['ego_pose_history'], prepared['states'])
                np.testing.assert_array_equal(saved['ego_pose_history_valid'], prepared['valid'])
                np.testing.assert_allclose(saved['ego_pose_history_times'], prepared['times'])
            rows = train.prepare_training_rows(root / 'tasks.jsonl', [label])
            checkpoint = train.fit(rows, root / 'fit', self.config(), stop_after_steps=1)
            snapshot = json.loads((root / 'fit' / 'inputs' / 'task_000000.json').read_text())
            self.assertEqual(snapshot['inputs']['ego_history_read_paths'],
                             task['inputs']['ego_history_read_paths'])
            self.assertEqual(torch.load(checkpoint, weights_only=False)['progress']['optimizer_steps'], 1)

    def test_export_requires_physical_train_for_both_research_roles(self):
        train=self.training()
        for role in ('train','validation'):
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);task,label=self.archive(root,role=role)
                for physical_split in ('test',None):
                    with self.subTest(role=role,physical_split=physical_split):
                        changed=copy.deepcopy(task)
                        if physical_split is None:changed['row'].pop('physical_split')
                        else:changed['row']['physical_split']=physical_split
                        (root/'task.json').write_text(json.dumps(changed))
                        with self.assertRaisesRegex(ValueError,'physical.*train'):
                            train.prepare_training_rows(root/'tasks.jsonl',[label])

    def test_saved_rows_cannot_bypass_physical_split_guard_during_fit(self):
        train=self.training()
        for role in ('train','validation'):
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);(root/'source').mkdir()
                task,label=self.archive(root/'source',role=role)
                rows=train.prepare_training_rows(root/'source'/'tasks.jsonl',[label])
                if role=='validation':
                    (root/'anchor').mkdir()
                    anchor,anchor_label=self.archive(root/'anchor',scene=SCENE.replace('09-50','09-51'))
                    rows+=train.prepare_training_rows([anchor],[anchor_label])
                saved=root/'prepared_rows.jsonl'
                saved.write_text(''.join(json.dumps(row)+'\n' for row in rows))
                for physical_split in ('test',None):
                    with self.subTest(role=role,physical_split=physical_split):
                        changed=copy.deepcopy(task)
                        if physical_split is None:changed['row'].pop('physical_split')
                        else:changed['row']['physical_split']=physical_split
                        (root/'source'/'task.json').write_text(json.dumps(changed))
                        with self.assertRaisesRegex(ValueError,'physical.*train'):
                            train.fit(saved,root/('rejected_'+str(physical_split)),self.config(),stop_after_steps=1)

    def test_recording_holdout_rejected_across_different_scenes(self):
        train=self.training()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'a').mkdir();(root/'b').mkdir()
            task,label=self.archive(root/'a')
            other,other_label=self.archive(root/'b',scene=SCENE[:-1]+'1',role='validation')
            with self.assertRaisesRegex(ValueError,'recording'):
                train.prepare_training_rows([task,other],[label,other_label])

    def test_validation_never_updates_weights_and_empty_labels_skip_optimizer(self):
        train=self.training()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'a').mkdir();(root/'b').mkdir()
            task,label=self.archive(root/'a')
            other,other_label=self.archive(root/'b',scene=SCENE.replace('09-50','09-51'),role='validation')
            only=train.prepare_training_rows([task],[label])
            together=train.prepare_training_rows([task,other],[label,other_label])
            config=self.config();config['epochs']=1
            a=torch.load(train.fit(only,root/'only',config),weights_only=False)
            b=torch.load(train.fit(together,root/'together',config),weights_only=False)
            for key in a['model_state']:self.assertTrue(torch.equal(a['model_state'][key],b['model_state'][key]))
            self.assertIsNotNone(json.loads((root/'together'/'report.json').read_text())['validation_loss'])
            empty=copy.deepcopy(label);empty['valid']=[False]*6;empty['waypoints']=[None]*6
            rows=train.prepare_training_rows([task],[empty])
            result=torch.load(train.fit(rows,root/'empty',config),weights_only=False)
            self.assertEqual(result['progress']['optimizer_steps'],0)
            self.assertEqual(result['model_version']['training']['status'],'initialized_untrained')
            self.assertEqual(result['optimizer_state']['state'],{})

    def test_invalid_prior_is_masked_and_checkpoint_tampering_rejected(self):
        train=self.training()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);_,label=self.archive(root)
            row=train.prepare_training_rows(root/'tasks.jsonl',[label])[-1]
            from planning.structured_driver import StructuredPlannerNetwork, load_structured_planner
            model=StructuredPlannerNetwork(tiny_spec())
            with torch.no_grad():model.output_head[-1].weight.zero_();model.output_head[-1].bias.fill_(100.)
            seen=[]
            hook=model.register_forward_pre_hook(lambda m,args:seen.append(args[0]['previous_plan_valid'].clone()))
            with np.load(root/'ego_features.npz') as saved:features={k:saved[k] for k in saved.files}
            loss,report=train.training_loss(model,features,row,refinement_depth=1)
            hook.remove()
            self.assertTrue(torch.isfinite(loss));self.assertEqual(report['invalid_prior_masks'],3)
            self.assertIn('plan exceeds configured admissibility bounds',report.get('invalid_prior_reasons',{}))
            self.assertTrue(all(not valid.any() for valid in seen))
            checkpoint=train.fit([row],root/'fit',self.config(),stop_after_steps=1)
            saved=torch.load(checkpoint,weights_only=False)
            changed=copy.deepcopy(saved);changed['model_state']['output_head.3.bias'][0]+=1
            torch.save(changed,root/'changed.pt')
            with self.assertRaises(ValueError):load_structured_planner(root/'changed.pt')
            changed=copy.deepcopy(saved);changed['progress']['row_position']=999
            torch.save(changed,checkpoint.parent/'changed_progress.pt')
            with self.assertRaises(ValueError):train.fit([row],root/'bad_resume',self.config(),resume=checkpoint.parent/'changed_progress.pt')
            changed=copy.deepcopy(saved)
            first=next(iter(changed['optimizer_state']['state'].values()))
            first['exp_avg'].add_(1)
            torch.save(changed,checkpoint.parent/'changed_optimizer.pt')
            with self.assertRaises(ValueError):train.fit([row],root/'bad_optimizer',self.config(),resume=checkpoint.parent/'changed_optimizer.pt')

    def test_seeded_initializer_reload_and_clean_help(self):
        train=self.training()
        from planning.structured_driver import load_structured_planner
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            train.initialize(root/'first.pt',tiny_spec().to_dict(),seed=11)
            train.initialize(root/'second.pt',tiny_spec().to_dict(),seed=11)
            a=load_structured_planner(root/'first.pt');b=load_structured_planner(root/'second.pt')
            self.assertIn('training_run_id',a.provenance['model_version']['training'])
            self.assertNotEqual(a.provenance['model_version']['training']['training_run_id'],
                                b.provenance['model_version']['training']['training_run_id'])
            self.assertEqual(a.provenance,load_structured_planner(root/'first.pt').provenance)
            self.assertEqual(a.provenance['model_version']['training']['status'],'initialized_untrained')
            for key,value in a.model.state_dict().items():self.assertTrue(torch.equal(value,b.model.state_dict()[key]))
            with self.assertRaises(FileExistsError):train.initialize(root/'first.pt',tiny_spec().to_dict(),seed=11)
        result=subprocess.run([sys.executable,'-c',"import sys; import planning.train_structured_driver; assert 'torch' not in sys.modules"],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        result=subprocess.run([sys.executable,'-m','planning.train_structured_driver','--help'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_resume_exact_state_data_binding_and_real_loader(self):
        train=self.training()
        from planning.structured_driver import load_structured_planner
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);task,label=self.archive(root)
            rows=train.prepare_training_rows(root/'tasks.jsonl',[label]);config=self.config()
            full=train.fit(rows,root/'full',config)
            part=train.fit(rows,root/'part',config,stop_after_steps=1)
            self.assertTrue((root/'part'/'checkpoint_000000.pt').is_file())
            resumed=train.fit(rows,root/'resumed',config,resume=part)
            a=torch.load(full,map_location='cpu',weights_only=False);b=torch.load(resumed,map_location='cpu',weights_only=False)
            self.assertEqual(a['progress'],b['progress'])
            self.assertNotEqual(a['model_version']['training']['training_run_id'],b['model_version']['training']['training_run_id'])
            self.assertEqual(b['model_version']['training']['training_run_id'],
                             torch.load(part,weights_only=False)['model_version']['training']['training_run_id'])
            self.assertTrue(torch.equal(a['rng']['torch'],b['rng']['torch']))
            self.assertEqual(a['rng']['python'],b['rng']['python'])
            np.testing.assert_array_equal(a['rng']['numpy'][1],b['rng']['numpy'][1])
            for key,value in a['model_state'].items():self.assertTrue(torch.equal(value,b['model_state'][key]),key)
            for key,state in a['optimizer_state']['state'].items():
                for name,value in state.items():
                    if isinstance(value,torch.Tensor):self.assertTrue(torch.equal(value,b['optimizer_state']['state'][key][name]))
                    else:self.assertEqual(value,b['optimizer_state']['state'][key][name])
            planner=load_structured_planner(full)
            with np.load(root/'ego_features.npz') as saved:features={k:saved[k] for k in saved.files}
            prepared=task['episode']['plans'][0]['prepared']
            output=planner.model(collate_structured_inputs([features],[prepared]))
            from planning.structured_driver import StructuredPlannerNetwork
            reference=StructuredPlannerNetwork(tiny_spec());reference.load_state_dict(a['model_state'])
            torch.testing.assert_close(output,reference(collate_structured_inputs([features],[prepared])),rtol=0,atol=0)
            changed=copy.deepcopy(config);changed['optimizer']['lr']=.1
            with self.assertRaises(ValueError):train.fit(rows,root/'bad_config',changed,resume=part)
            changed=copy.deepcopy(rows);changed[0]['supervision']['waypoints'][0][0]+=1
            with self.assertRaises(ValueError):train.fit(changed,root/'bad_rows',config,resume=part)
            with np.load(root/'ego_features.npz') as saved:arrays={k:saved[k] for k in saved.files}
            arrays['regression_map'][0,0,0,0,0,0]+=1;np.savez(root/'ego_features.npz',**arrays)
            with self.assertRaises(ValueError):train.fit(rows,root/'bad_features',config,resume=part)

    def test_resume_uses_content_after_input_directory_relocation(self):
        train=self.training()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'data').mkdir()
            _,label=self.archive(root/'data')
            rows=train.prepare_training_rows(root/'data'/'tasks.jsonl',[label])
            config=self.config();config['epochs']=1
            part=train.fit(rows,root/'part',config,stop_after_steps=1)
            shutil.copytree(root/'data',root/'moved')
            task=json.loads((root/'moved'/'task.json').read_text())
            task['inputs']['artifact_root']=str(root/'moved')
            (root/'moved'/'task.json').write_text(json.dumps(task))
            moved=train.prepare_training_rows(root/'moved'/'tasks.jsonl',[label])
            resumed=train.fit(moved,root/'resumed',config,resume=part)
            self.assertEqual(torch.load(resumed,weights_only=False)['model_version']['training']['training_run_id'],
                             torch.load(part,weights_only=False)['model_version']['training']['training_run_id'])

    def test_export_validates_actual_feature_mask_against_archived_numeric_tokens(self):
        train=self.training()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);_,label=self.archive(root)
            with np.load(root/'ego_features.npz') as saved:features={k:saved[k] for k in saved.files}
            features['active_agent_mask'][0,1,0]=False
            np.savez(root/'ego_features.npz',**features)
            with self.assertRaisesRegex(ValueError,'feature'):
                train.prepare_training_rows(root/'tasks.jsonl',[label])

    def test_bundle_training_counts_actual_primitive_evidence(self):
        train=self.training()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);_,label=self.archive(root,control='one_shot')
            rows=train.prepare_training_rows(root/'tasks.jsonl',[label])
            try:
                checkpoint=train.fit(rows,root/'fit',self.config(),stop_after_steps=1)
            except KeyError as exc:
                self.fail('actual bundle task schema is not handled: '+str(exc))
            self.assertEqual(torch.load(checkpoint,weights_only=False)['report']['evidence_stages'],{'Ego':1,'PF':2})

    def test_archived_exact_repeat_keeps_its_original_prior_slot(self):
        train=self.training()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);_,label=self.archive(root,control='exact_repeat')
            row=train.prepare_training_rows(root/'tasks.jsonl',[label])[-1]
            from planning.structured_driver import StructuredPlannerNetwork
            model=StructuredPlannerNetwork(tiny_spec())
            with torch.no_grad():model.output_head[-1].weight.mul_(.001);model.output_head[-1].bias.mul_(.001)
            seen=[]
            hook=model.register_forward_pre_hook(lambda m,args:seen.append(args[0]['previous_plan_valid'].clone()))
            with np.load(root/'ego_features.npz') as saved:features={k:saved[k] for k in saved.files}
            train.training_loss(model,features,row,refinement_depth=1)
            hook.remove()
            self.assertEqual([bool(mask.any()) for mask in seen],[False,False,False,True])

    def test_prefix_and_refinement_use_detached_predictions_never_labels(self):
        train=self.training()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);_,label=self.archive(root)
            rows=train.prepare_training_rows(root/'tasks.jsonl',[label])
            from planning.structured_driver import StructuredPlannerNetwork
            torch.manual_seed(3);model=StructuredPlannerNetwork(tiny_spec())
            # Small actual model residuals ensure valid causal priors for this test.
            with torch.no_grad():model.output_head[-1].weight.mul_(.001);model.output_head[-1].bias.mul_(.001)
            seen=[]
            hook=model.register_forward_hook(lambda m,args,out:seen.append((args[0],out.detach().clone())))
            with np.load(root/'ego_features.npz') as saved:features={k:saved[k] for k in saved.files}
            loss,report=train.training_loss(model,features,rows[-1],refinement_depth=1)
            hook.remove();loss.backward()
            self.assertEqual(len(seen),4)
            self.assertFalse(seen[0][0]['previous_plan_valid'].any())
            for i in range(1,4):
                torch.testing.assert_close(seen[i][0]['previous_plan'],seen[i-1][1])
                self.assertFalse(seen[i][0]['previous_plan'].requires_grad)
            self.assertGreater(seen[1][0]['observation_mask'].sum(),seen[0][0]['observation_mask'].sum())
            torch.testing.assert_close(seen[-1][0]['observations'],seen[-2][0]['observations'])
            self.assertEqual(report['supervised_forwards'],4)
            self.assertGreater(model.prior_encoder[0].weight.grad.abs().sum(),0)
