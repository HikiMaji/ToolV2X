"""T8 causal features and policy contracts; torch tests are explicitly separate."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest

from test_query_data import QueryFixture
from planning.method_controls import capture_control_prefix
from planning.method_episode import decision_state, run_task_episode


class ValueFixture(QueryFixture):
    def value_module(self):
        return importlib.import_module('planning.query_value')

    def visible(self, after=0):
        inputs=self.inputs()
        prefix=capture_control_prefix(after_calls=after,**inputs,
            policy=lambda _:dict(tool='P',mode='current',reason='fixture'))
        return decision_state(prefix['episode'])[0],inputs,prefix


class FeatureTests(ValueFixture):
    def test_features_are_ordered_causal_and_detached(self):
        q=self.value_module();state,_,_=self.visible(1);before=copy.deepcopy(state)
        f=q.state_features(state,q.feature_spec())
        self.assertEqual(len(f['values']),len(f['names']))
        self.assertEqual(len(f['names']),len(set(f['names'])))
        self.assertEqual(sum(n.startswith('plan.') for n in f['names']),24)
        self.assertEqual(state,before)
        self.assertEqual(f,q.state_features(copy.deepcopy(state),q.feature_spec()))
        self.assertGreater(f['values'][f['names'].index('acquired.history.count')],0)
        self.assertGreater(f['values'][f['names'].index('derived.forecast.count')],0)
        self.assertFalse(any('loss' in n or 'actual_response' in n for n in f['names']))

    def test_unknown_gt_unbought_and_candidate_outcomes_fail_closed(self):
        q=self.value_module();state,_,_=self.visible()
        for key in ('gt','unbought_forecast','candidate_actual_response_bytes','future_poses'):
            bad=copy.deepcopy(state);bad[key]=[999.]
            with self.subTest(key=key),self.assertRaises(ValueError):q.state_features(bad,q.feature_spec())
        for key in ('gt_ADE','unbought_forecast','candidate_actual_bytes'):
            bad=copy.deepcopy(state);bad['current_plan'][key]=999.
            with self.assertRaises(ValueError):q.state_features(bad,q.feature_spec())
        bad=copy.deepcopy(state);bad['action_feasibility'][0]['response_bytes']=123
        with self.assertRaises(ValueError):q.state_features(bad,q.feature_spec())
        bad=copy.deepcopy(state);bad['local_evidence'][0]['value']['gt']=99
        with self.assertRaises(ValueError):q.state_features(bad,q.feature_spec())

    def test_missing_old_and_motion_have_explicit_bits_not_imputed_observations(self):
        q=self.value_module();state,_,_=self.visible();state['ego_motion']['speed_mps']=None
        f=q.state_features(state,q.feature_spec());v=dict(zip(f['names'],f['values']))
        self.assertEqual(v['previous_plan.missing'],1.)
        self.assertEqual(v['motion.speed_mps.missing'],1.)
        self.assertEqual(v['motion.speed_mps'],0.)
        self.assertIn('motion.speed_mps',f['missing'])
        with self.assertRaises(ValueError):q.state_features(state,dict(q.feature_spec(),version='unknown'))

    def test_acquired_requires_receipt_and_admission_cannot_name_unknown_evidence(self):
        q=self.value_module();state,_,_=self.visible(1)
        bad=copy.deepcopy(state);bad['response_receipts']=[]
        with self.assertRaises(ValueError):q.state_features(bad,q.feature_spec())
        bad=copy.deepcopy(state);ref=copy.deepcopy(bad['acquired_fields'][0]['ref']);ref['track_handle']=999
        bad['admission_report']['admitted_field_refs'].append(ref)
        with self.assertRaises(ValueError):q.state_features(bad,q.feature_spec())

    def test_selection_stop_zero_ties_and_feasibility(self):
        q=self.value_module()
        self.assertEqual(q.choose_query({'P_current':-.2,'F_current':0.},['STOP','P_current','F_current']),'STOP')
        self.assertEqual(q.choose_query({'STOP':99.,'P_current':.3,'F_change':99.},['STOP','P_current']),'P_current')
        self.assertEqual(q.choose_query({'F_current':.3,'P_current':.3},['F_current','P_current','STOP']),'P_current')
        self.assertEqual(q.choose_query({'P_current':99.},['STOP']),'STOP')
        with self.assertRaises(ValueError):q.choose_query({'P_current':float('nan')},['STOP','P_current'])

    def test_initial_change_and_exhausted_calls_are_masked(self):
        q=self.value_module();state,_,_=self.visible()
        state['available_actions'].append(dict(tool='F',mode='change'))
        self.assertNotIn('F_change',q.feasible_queries(state))
        state['remaining_budget']['calls']=0
        self.assertEqual(q.feasible_queries(state),['STOP'])
        state['remaining_budget']['calls']=2;state['remaining_budget']['bytes']=0
        self.assertEqual(q.feasible_queries(state),['STOP'])


class TrainingFixture(ValueFixture):
    def training_data(self,root,control_name='feedback'):
        from planning.query_data import collect_branches, make_terminal_targets
        from test_query_data import utility
        runtime,rows,spec=self.collection(root/'seed',limit_changes=dict(max_request_bytes=1),control_name=control_name)
        spec['limits']['execution_spec']['max_request_bytes']=30000
        groups=['testoutput_CAV_data_2022-03-%02d-10-09-50'%day for day in (15,16,17)]
        spec['recording_folds']={g:'fold%d'%i for i,g in enumerate(groups)}
        expanded=[dict(rows[0],sample_id='s%d'%i,scene=g+'_0') for i,g in enumerate(groups)]
        branches=root/'branches';collect_branches(expanded,branches,runtime,spec)
        labels=root/'labels';labels.mkdir()
        (labels/'train.jsonl').write_text(''.join(json.dumps(dict(
            **{k:r[k] for k in ('sample_id','scene','role','g')},times_seconds=[.5,1.,1.5,2.,2.5,3.],
            valid=[True]*6,waypoints=[[float(i),0.] for i in range(1,7)]))+'\n' for r in expanded))
        terminal=root/'terminal';make_terminal_targets(branches,labels,utility(),terminal)
        binding=json.loads((terminal/'config.json').read_text())['binding']
        return branches,labels,terminal,groups,binding

    def config(self,root,groups,binding,**changes):
        from test_query_data import utility
        q=self.value_module()
        return q.training_config(checkpoint=str(root/'value.pt'),train_recordings=groups,binding=binding,
            utility_spec=utility(),steps=3,batch_size_per_stage=4,**changes)


class DatasetTests(TrainingFixture):
    def test_algebraically_consistent_terminal_loss_contamination_is_rejected_before_fit(self):
        from unittest.mock import patch
        q=self.value_module()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);_,_,targets,groups,binding=self.training_data(root)
            config=self.config(root,groups[:2],binding,held_out_fold='fold2')
            original=self.read_rows(targets/'targets.jsonl')
            for field in ('source_loss','terminal_loss'):
                rows=copy.deepcopy(original)
                row=next(r for r in rows if r['action']['tool']!='STOP' and r['physical_recording']==groups[0])
                row[field]+=1.
                row['target']=row['source_loss']-row['terminal_loss']-row['cost_penalty']
                (targets/'targets.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                with self.subTest(field=field),patch.object(q,'_fit',return_value=None),self.assertRaises(ValueError):
                    q.fit_terminal_policy(targets,groups[:2],config)
            self.assertFalse(Path(config['checkpoint']).exists())

    def test_first_target_quality_is_recomputed_for_the_actual_chosen_continuation(self):
        from planning.query_data import make_first_targets
        from test_query_data import Teacher,utility
        q=self.value_module()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);branches,labels,_,groups,binding=self.training_data(root)
            teachers={}
            for i,group in enumerate(groups):
                teacher=Teacher(binding,action=dict(tool='F',mode='current',reason='fixture'))
                teacher.provenance.update(held_out_fold='fold%d'%i,
                    training_recordings=[g for g in groups if g!=group],feature_spec=q.feature_spec())
                teachers['fold%d'%i]=teacher
            first=root/'first';make_first_targets(branches,labels,teachers,utility(),first)
            config=self.config(root,groups,binding)
            self.assertTrue(q.training_examples(first,'first',config)['examples'])
            rows=self.read_rows(first/'targets.jsonl')
            row=next(r for r in rows if r['action']['tool']=='P')
            self.assertEqual(row['continuation_action']['tool'],'F')
            row['terminal_loss']+=1.
            row['target']=row['source_loss']-row['terminal_loss']-row['cost_penalty']
            (first/'targets.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaises(ValueError):q.training_examples(first,'first',config)

    def test_quality_reload_keeps_weighted_fde_failure_and_stop_contracts(self):
        from test_query_data import TargetTests,utility
        from planning.query_data import make_terminal_targets
        q=self.value_module()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);branches,labels,_=TargetTests.label_data(self,root,invalid_stage=2)
            label=self.read_rows(labels/'train.jsonl')[0]
            label['waypoints']=[[float(i),0.] for i in range(1,7)]
            (labels/'train.jsonl').write_text(json.dumps(label)+'\n')
            u=utility();u['quality_weights']=dict(ADE3=.4,FDE3=.6)
            targets=root/'terminal';rows=make_terminal_targets(branches,labels,u,targets)
            binding=json.loads((targets/'config.json').read_text())['binding']
            groups=sorted({r['physical_recording'] for r in rows})
            config=self.config(root,groups,binding,held_out_fold='fold1');config['utility_spec']=u
            examples=q.training_examples(targets,'terminal',config)['examples']
            # Six lateral errors .1,...,.6 give ADE=.35 and FDE=.6: weighted loss=.5.
            for row in rows:self.assertAlmostEqual(row['source_loss'],.5)
            failed=next(r for r in rows if r['terminal_status']=='recorded_failure')
            self.assertEqual(failed['terminal_loss'],10.)
            self.assertIsNone(failed['terminal_plan_id'])
            self.assertTrue(any(e['state_id']==failed['state_id'] and e['action']==failed['action'] for e in examples))
            stop=next(r for r in rows if r['action']['tool']=='STOP')
            self.assertEqual(stop['target'],0.)
            self.assertTrue(all(v==0 for v in stop['incremental_cost'].values()))
            for status in ('recorded_failure','STOP'):
                bad=copy.deepcopy(rows)
                row=next(r for r in bad if (r['action']['tool']=='STOP' if status=='STOP'
                                          else r['terminal_status']==status))
                row['terminal_loss']+=1.
                if status!='STOP':row['target']-=1.
                (targets/'targets.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in bad))
                with self.subTest(status=status),self.assertRaises(ValueError):q.training_examples(targets,'terminal',config)

    def test_target_quality_requires_independent_matching_offline_labels(self):
        from test_query_data import TargetTests,utility
        from planning.query_data import make_terminal_targets
        q=self.value_module()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);branches,labels,_=TargetTests.label_data(self,root)
            targets=root/'terminal';rows=make_terminal_targets(branches,labels,utility(),targets)
            binding=json.loads((targets/'config.json').read_text())['binding']
            config=self.config(root,sorted({r['physical_recording'] for r in rows}),binding)
            self.assertTrue(q.training_examples(targets,'terminal',config)['examples'])
            path=targets/'labels.jsonl';original=self.read_rows(path)
            moved=copy.deepcopy(original);moved[0]['waypoints'][0][1]+=1.
            wrong=copy.deepcopy(original);wrong[0]['role']='validation'
            for content in (moved,wrong,original*2,[]):
                path.write_text(''.join(json.dumps(r)+'\n' for r in content))
                with self.subTest(content=len(content)),self.assertRaises(ValueError):
                    q.training_examples(targets,'terminal',config)
            path.unlink()
            with self.assertRaisesRegex(ValueError,'independent offline'):
                q.training_examples(targets,'terminal',config)

    def test_training_rows_join_actual_online_states_and_keep_recording_holdout(self):
        q=self.value_module();self.assertTrue(hasattr(q,'training_examples'),'T8 target adapter missing')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);_,_,targets,groups,binding=self.training_data(root)
            config=self.config(root,groups[:2],binding,held_out_fold='fold2')
            table=q.training_examples(targets,'terminal',config)
            self.assertEqual({r['physical_recording'] for r in table['examples']},set(groups[:2]))
            self.assertTrue(all(r['stage']=='terminal' for r in table['examples']))
            self.assertTrue(all(r['action']['tool']!='STOP' for r in table['examples']))
            self.assertEqual(table['coverage']['available_targets'],30)
            wrong=copy.deepcopy(config);wrong['train_recordings']=groups
            with self.assertRaises(ValueError):q.training_examples(targets,'terminal',wrong)
            wrong=copy.deepcopy(config);wrong['binding']['limits']['receiver_spec']['peer_reserve']-=1
            with self.assertRaises(ValueError):q.training_examples(targets,'terminal',wrong)

    def test_label_values_never_change_feature_vectors_and_missing_rows_rejected(self):
        q=self.value_module();self.assertTrue(hasattr(q,'training_examples'),'T8 target adapter missing')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);branches,labels,targets,groups,binding=self.training_data(root);config=self.config(root,groups[:2],binding)
            before=q.training_examples(targets,'terminal',config)
            # Change independent synthetic labels and regenerate offline targets;
            # hand-editing losses is invalid supervision, not a feature-isolation test.
            from planning.query_data import make_terminal_targets
            from test_query_data import utility
            changed=self.read_rows(labels/'train.jsonl')
            for label in changed:label['waypoints']=[[float(i)+2.,0.] for i in range(1,7)]
            (labels/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in changed))
            updated=root/'updated';make_terminal_targets(branches,labels,utility(),updated)
            after=q.training_examples(updated,'terminal',config)
            self.assertEqual([r['features'] for r in before['examples']],[r['features'] for r in after['examples']])
            rows=self.read_rows(targets/'targets.jsonl')
            self.assertNotEqual([r['source_loss'] for r in rows],
                                [r['source_loss'] for r in self.read_rows(updated/'targets.jsonl')])
            (targets/'targets.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows[:-1]))
            with self.assertRaises(ValueError):q.training_examples(targets,'terminal',config)


class ModelTests(TrainingFixture):
    """Run explicitly with torch installed and CUDA_VISIBLE_DEVICES=''."""
    def test_teacher_save_reload_and_exact_resume(self):
        q=self.value_module();self.assertTrue(hasattr(q,'fit_terminal_policy'),'T8 fitter missing')
        import torch
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);branches,_,targets,groups,binding=self.training_data(root)
            config=self.config(root,groups[:2],binding,held_out_fold='fold2')
            config['steps']=8
            policy=q.fit_terminal_policy(targets,groups[:2],config)
            state=self.read_rows(branches/'online_states.jsonl')[-1]['state']
            loaded=q.load_query_policy(config['checkpoint'],expected_binding=binding,policy_kind='terminal')
            self.assertEqual(policy.values(state),loaded.values(state))
            self.assertEqual(loaded.provenance['training_recordings'],groups[:2])
            self.assertEqual(loaded.provenance['held_out_fold'],'fold2')
            self.assertEqual(loaded.training['step'],8)
            short=dict(config,steps=3,checkpoint=str(root/'short.pt'))
            q.fit_terminal_policy(targets,groups[:2],short)
            resumed=dict(config,checkpoint=str(root/'resumed.pt'),resume_from=short['checkpoint'])
            restored=q.fit_terminal_policy(targets,groups[:2],resumed)
            self.assertEqual(restored.values(state),loaded.values(state))
            self.assertEqual(restored.training['loss_history'],loaded.training['loss_history'])
            self.assertTrue(all(p.device.type=='cpu' for p in restored.model.parameters()))
            wrong=copy.deepcopy(binding);wrong['predictor']['model_version']['revision']='drift'
            with self.assertRaises(ValueError):q.load_query_policy(config['checkpoint'],expected_binding=wrong)
            with self.assertRaises(ValueError):q.load_query_policy(config['checkpoint'],policy_kind='bundle_terminal')
            self.assertEqual(loaded.training['normalization_recordings'],groups[:2])

    def test_fitted_teachers_feed_t7_and_shared_batches_keep_both_stages(self):
        q=self.value_module();self.assertTrue(hasattr(q,'fit_shared_policy'),'T8 shared fitter missing')
        from planning.query_data import make_first_targets
        from test_query_data import utility
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);branches,labels,terminal,groups,binding=self.training_data(root)
            teachers={}
            for i,g in enumerate(groups):
                train=[x for x in groups if x!=g]
                config=self.config(root,train,binding,held_out_fold='fold%d'%i)
                config.update(checkpoint=str(root/('teacher%d.pt'%i)),policy_id='teacher%d_v1'%i)
                teachers['fold%d'%i]=q.fit_terminal_policy(terminal,train,config)
            first=root/'first';make_first_targets(branches,labels,teachers,utility(),first)
            config=self.config(root,groups,binding)
            shared=q.fit_shared_policy(first,terminal,config)
            self.assertEqual(shared.training['stage_draws'],dict(first=12,terminal=12))
            self.assertEqual(shared.kind,'shared')
            self.assertEqual(shared.training['actual_training_recordings'],groups)
            outer=self.config(root,groups[:2],binding,held_out_fold='fold2')
            outer['checkpoint']=str(root/'invalid.pt')
            with self.assertRaisesRegex(ValueError,'teacher.*training|outer'):
                q.fit_shared_policy(first,terminal,outer)
            self.assertFalse(Path(outer['checkpoint']).exists())
            state,inputs,_=self.visible()
            # Contract fixture scene differs; semantic model versions match.
            ep=run_task_episode(**inputs,policy=shared)
            self.assertEqual(ep['status'],'completed',ep.get('error'))
            self.assertEqual(ep['value_policy']['policy_id'],config['policy_id'])

    def test_synthetic_regression_learns_and_checkpoint_is_written_before_updates(self):
        q=self.value_module();self.assertTrue(hasattr(q,'fit_terminal_policy'),'T8 fitter missing')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);_,_,targets,groups,binding=self.training_data(root);config=self.config(root,groups[:2],binding)
            config['held_out_fold']='fold2';config['steps']=100
            policy=q.fit_terminal_policy(targets,groups[:2],config)
            self.assertLess(sum(policy.training['loss_history'][-10:]),sum(policy.training['loss_history'][:10]))
            self.assertTrue(Path(config['checkpoint']).exists())
            with self.assertRaises(FileExistsError):q.fit_terminal_policy(targets,groups[:2],config)


class BundleFeatureTests(ValueFixture):
    def bundle_visible(self):
        from planning.method_controls import episode_bundle,control_spec
        from tools.control_bundle import bundle_actions
        from tools.vehicle import decode_task_response
        from test_vehicle_tools import window
        from test_task_spec import provenance
        from tools.vehicle import VehicleTools
        inputs=self.inputs();scene='testoutput_CAV_data_2022-03-15-10-09-50_0'
        inputs['local_window']['scene']=scene
        peer=window();peer['scene']=scene
        inputs['service']=VehicleTools(lambda:peer,self.predictor,scene,10,'peer',task_provenance=provenance())
        prefix=capture_control_prefix(after_calls=0,**inputs,policy=lambda _:dict(tool='STOP',mode=None,reason='fixture'))
        state=decision_state(prefix['episode'])[0]
        bundle=dict(continuation_policy_id='bundle_value_v1',candidate_sources=['initial','slower','constant_motion'],
            slowdown_scale=.5,max_candidates=3,wrapper_reserve_bytes=2048,extra_generation=None)
        spec=control_spec('one_shot',bundle=bundle)
        envelope=episode_bundle(state,spec,'P');first=inputs['service'].query_task(envelope['first_request'])
        packet=decode_task_response(first['wire'],first['request'])
        return dict(envelope=envelope,first_response=packet,available_actions=bundle_actions(envelope)),inputs,spec

    def test_candidate_features_use_only_sent_paths_and_real_first_response(self):
        q=self.value_module();self.assertTrue(hasattr(q,'bundle_features'),'T8 bundle feature adapter missing')
        state,_,_=self.bundle_visible()
        a=dict(tool='F',mode='current',candidate_id='initial');b=dict(a,candidate_id='slower')
        left=q.bundle_features(state,a,q.feature_spec());right=q.bundle_features(state,b,q.feature_spec())
        self.assertNotEqual(left['values'],right['values'])
        normal,_,_=self.visible();self.assertEqual(left['names'],q.state_features(normal,q.feature_spec())['names'])
        self.assertEqual(dict(zip(left['names'],left['values']))['admission.missing'],1.)
        bad=copy.deepcopy(state);bad['actual_revised_plan']=[[99.,0.]]*6
        with self.assertRaises(ValueError):q.bundle_features(bad,a,q.feature_spec())
        with self.assertRaises(ValueError):q.bundle_features(state,dict(a,candidate_id='unseen'),q.feature_spec())
        bad=copy.deepcopy(state);bad['envelope']['public_summary']['gt_ADE']=0.
        with self.assertRaises(ValueError):q.bundle_features(bad,a,q.feature_spec())
        self.assertFalse(any('derived.forecast'==name for name in left['names']))


class BundleModelTests(ValueFixture):
    bundle_visible=BundleFeatureTests.bundle_visible
    def test_separate_conditional_fit_load_and_real_provider_registration(self):
        q=self.value_module();self.assertTrue(hasattr(q,'fit_bundle_continuation'),'T8 conditional fitter missing')
        from planning.method_controls import register_value_continuation
        from planning.query_data import _semantic_binding
        from test_query_data import utility
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);visible,inputs,control=self.bundle_visible()
            binding=_semantic_binding(dict(driver=inputs['driver'].provenance,predictor=inputs['predictor'].descriptor),
                dict(limits=inputs['limits'],local_provenance=inputs['local_provenance'],control=control))
            groups=['testoutput_CAV_data_2022-03-15-10-09-50','testoutput_CAV_data_2022-03-16-10-09-50']
            # Explicit synthetic caller-supplied supervision. No real branch or
            # real-quality claim; T7 alternating targets must not be accepted.
            rows=[]
            for action in visible['available_actions']:
                rows.append(dict(state_id='bundle0',physical_recording=groups[0],fold='fold0',state=copy.deepcopy(visible),
                    action=action,stop_loss=1.,terminal_loss=1. if action['tool']=='STOP' else .5,
                    incremental_cost=dict(request_bytes=0.,response_bytes=0.,total_compute_seconds=0.),
                    target=0. if action['tool']=='STOP' else .5))
            table=dict(version='toolv2x_bundle_targets_v1',kind='bundle_terminal',binding=binding,utility_spec=utility(),
                recording_folds={groups[0]:'fold0',groups[1]:'fold1'},rows=rows,
                supervision=dict(version='toolv2x_bundle_supervision_v1',origin='synthetic_contract',
                    stop_reference='first_return_then_final_driver',cost_scope='outer_wire_and_complete_terminal_compute'))
            config=q.training_config(checkpoint=str(root/'bundle.pt'),train_recordings=groups[:1],binding=binding,
                utility_spec=utility(),steps=5,policy_id='bundle_value_v1')
            # The public fitter requires research-role-bearing raw archives.
            # This explicitly synthetic optimizer contract exercises private helpers.
            fitted=q._fit([q._bundle_examples(table,config)],'bundle_terminal',config)
            loaded=q.load_query_policy(config['checkpoint'],policy_kind='bundle_terminal',expected_binding=binding)
            self.assertEqual(loaded(visible),fitted(visible))
            self.assertIn({k:v for k,v in loaded(visible).items() if k!='reason'},visible['available_actions'])
            changed=copy.deepcopy(visible);changed['envelope']['limits']['budget_mode']='episode_aggregate'
            with self.assertRaisesRegex(ValueError,'budget/candidate'):loaded(changed)
            # The original fixture has already purchased P; use a new provider episode.
            from tools.vehicle import VehicleTools
            from test_task_spec import provenance
            from test_vehicle_tools import window
            peer=window();peer['scene']=visible['envelope']['first_request']['scene']
            fresh=VehicleTools(lambda:peer,self.predictor,peer['scene'],10,'peer',task_provenance=provenance())
            from tools.task_spec import FrozenPredictor
            wrong_predictor=FrozenPredictor(self.predictor,dict(name='different_predictor',revision='v2'),{})
            wrong_service=VehicleTools(lambda:peer,wrong_predictor,peer['scene'],10,'peer',task_provenance=provenance())
            with self.assertRaisesRegex(ValueError,'actual provider'):register_value_continuation(wrong_service,config['checkpoint'],binding)
            self.assertIsNone(wrong_service._task_window)
            register_value_continuation(fresh,config['checkpoint'],binding)
            with self.assertRaisesRegex(ValueError,'budget/candidate'):fresh.query_bundle(changed['envelope'])
            self.assertIsNone(fresh._task_window);self.assertEqual(fresh.task_records,[])
            result=fresh.query_bundle(visible['envelope'])
            self.assertEqual(result['cost']['rpc_rounds'],1)
            self.assertLessEqual(result['cost']['capability_calls'],2)
            wrong=copy.deepcopy(table);wrong['kind']='terminal'
            with self.assertRaises(ValueError):q._bundle_examples(wrong,config)
            wrong=copy.deepcopy(table);wrong['rows'].pop()
            with self.assertRaises(ValueError):q._bundle_examples(wrong,config)


class TrainingAuditTests(unittest.TestCase):
    def test_controls_require_equal_capacity_groups_updates_and_supervision(self):
        from planning import query_value as q
        self.assertTrue(hasattr(q,'compare_training_contracts'),'training fairness audit missing')
        base=dict(feature_spec=q.feature_spec(),hidden_sizes=[64,64],action_spec=dict(version='a'),
            train_recordings=['recording_a'],optimizer=dict(learning_rate=.001,seed=0),updates=20,
            stage_draws=dict(first=80,terminal=80),supervision_counts=dict(first=3,terminal=8),
            candidates_by_recording=dict(recording_a=13),receiver_query_spec=dict(bytes=100),
            utility_spec=dict(failure_loss=10.),base_binding=dict(predictor='fixture_v1'))
        self.assertTrue(q.compare_training_contracts(base,copy.deepcopy(base))['matched'])
        for key,new in (('hidden_sizes',[32,32]),('train_recordings',['recording_b']),('updates',21),
                ('supervision_counts',dict(first=3,terminal=4)),('stage_draws',dict(first=80,terminal=40)),
                ('utility_spec',dict(failure_loss=999.)),('base_binding',dict(predictor='drift'))):
            other=copy.deepcopy(base);other[key]=new
            result=q.compare_training_contracts(base,other)
            self.assertFalse(result['matched']);self.assertIn(key,result['differences'])


class RecoveryTests(TrainingFixture):
    def test_interruption_has_initial_checkpoint_and_rejects_changed_targets_on_resume(self):
        from unittest.mock import patch
        q=self.value_module();import torch
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);_,_,targets,groups,binding=self.training_data(root)
            config=self.config(root,groups[:2],binding,held_out_fold='fold2')
            with patch.object(torch.optim.Adam,'step',side_effect=RuntimeError('fixture interruption')):
                with self.assertRaisesRegex(RuntimeError,'fixture interruption'):q.fit_terminal_policy(targets,groups[:2],config)
            loaded=q.load_query_policy(config['checkpoint']);self.assertEqual(loaded.training['step'],0)
            self.assertEqual(loaded.training['loss_history'],[])
            rows=self.read_rows(targets/'targets.jsonl')
            for r in rows:
                if r['physical_recording']==groups[0] and r['action']['tool']!='STOP':r['source_loss']+=1.;r['target']+=1.
            (targets/'targets.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError,'actual archived trajectories'):
                q.fit_terminal_policy(targets,groups[:2],dict(config,resume_from=config['checkpoint']))

    def test_teacher_normalization_is_only_the_used_training_features(self):
        import numpy as np
        q=self.value_module()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);_,_,targets,groups,binding=self.training_data(root)
            config=self.config(root,groups[:1],binding,held_out_fold='fold2')
            table=q.training_examples(targets,'terminal',config)
            policy=q.fit_terminal_policy(targets,groups[:1],config)
            expected=np.asarray([r['features'] for r in table['examples']]).mean(axis=0).tolist()
            self.assertEqual(policy.normalization['mean'],expected)
            self.assertEqual(policy.training['normalization_recordings'],groups[:1])

    def test_loaded_policy_rejects_runtime_drift_before_driver_or_peer_execution(self):
        from planning.method_controls import make_control_policy
        q=self.value_module()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);_,_,targets,groups,binding=self.training_data(root)
            config=self.config(root,groups[:2],binding,held_out_fold='fold2')
            policy=q.fit_terminal_policy(targets,groups[:2],config)
            inputs=self.inputs();inputs['limits']['receiver_spec']['peer_reserve']-=1
            wrapped=make_control_policy(inputs['control_spec'],policy)
            with self.assertRaisesRegex(ValueError,'binding drift'):run_task_episode(**inputs,policy=wrapped)
            self.assertEqual(inputs['driver'].inputs,[])
            self.assertIsNone(inputs['service']._task_window)


class ReviewRegressionTests(TrainingFixture):
    def test_distinct_fits_have_distinct_frozen_teacher_sources(self):
        q=self.value_module()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);_,_,targets,groups,binding=self.training_data(root)
            c=self.config(root,groups[:2],binding,held_out_fold='fold2')
            a=q.fit_terminal_policy(targets,groups[:2],c)
            b=q.fit_terminal_policy(targets,groups[:2],dict(c,seed=1,checkpoint=str(root/'different.pt')))
            self.assertNotEqual(a.provenance,b.provenance)
            self.assertEqual(a.provenance,q.load_query_policy(c['checkpoint']).provenance)
            self.assertEqual(a.provenance['frozen_source']['step'],c['steps'])
            self.assertEqual(a.provenance['frozen_source']['normalization'],a.normalization)
            self.assertFalse(q.compare_training_contracts(a.training_contract,b.training_contract)['matched'])
            same=copy.deepcopy(a.training_contract);same['utility_spec']['failure_loss']+=1.
            self.assertFalse(q.compare_training_contracts(a.training_contract,same)['matched'])

    def test_first_target_adapter_rejects_changed_continuation_terminal_and_costs(self):
        from planning.query_data import make_first_targets
        from test_query_data import Teacher,utility
        q=self.value_module()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);branches,labels,terminal,groups,binding=self.training_data(root)
            teachers={}
            for i,group in enumerate(groups):
                teacher=Teacher(binding,action=dict(tool='F',mode='current',reason='fixture'))
                teacher.provenance.update(held_out_fold='fold%d'%i,training_recordings=[g for g in groups if g!=group],
                    feature_spec=q.feature_spec())
                teachers['fold%d'%i]=teacher
            first=root/'first';make_first_targets(branches,labels,teachers,utility(),first)
            c=self.config(root,groups,binding);original=self.read_rows(first/'targets.jsonl')
            q.training_examples(first,'first',c)
            for changes in (dict(continuation_action=dict(tool='I',mode='current',reason='bad')),
                    dict(terminal_branch_id='missing'),dict(terminal_plan_id='missing'),
                    dict(terminal_status='missing'),dict(incremental_cost=None),dict(cost_penalty=999.,target=-999.)):
                rows=copy.deepcopy(original);row=next(r for r in rows if r['action']['tool']=='P');row.update(changes)
                (first/'targets.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                with self.subTest(changes=changes),self.assertRaises(ValueError):q.training_examples(first,'first',c)


class MetadataBindingTests(ValueFixture):
    def test_executor_binding_hook_only_receives_detached_json_metadata(self):
        seen=[];case=self
        class Policy:
            def validate_runtime(self,**metadata):
                case.assertEqual(set(metadata),{'driver_provenance','predictor_descriptor','limits','local_provenance','control_spec'})
                json.dumps(metadata)
                seen.append(copy.deepcopy(metadata))
                metadata['limits']['max_calls']=99
            def __call__(self,state):return dict(tool='STOP',mode=None,reason='metadata_contract')
        inputs=self.inputs();expected=copy.deepcopy(inputs['limits'])
        episode=run_task_episode(**inputs,policy=Policy())
        self.assertEqual(episode['limits'],expected)
        self.assertEqual(set(seen[0]),{'driver_provenance','predictor_descriptor','limits','local_provenance','control_spec'})
        self.assertIsNone(inputs['service']._task_window)

    def test_bundle_binding_hook_has_no_predictor_object_and_cannot_change_sent_data(self):
        from test_control_bundle import envelope
        from test_task_spec import provenance
        seen=[];case=self
        class Policy:
            def validate_provider(self,descriptor,metadata):
                case.assertIsInstance(descriptor,dict)
                json.dumps(dict(descriptor=descriptor,metadata=metadata))
                seen.append(copy.deepcopy(descriptor))
                metadata['prediction']['revision']='changed'
            def validate_envelope(self,envelope):envelope['limits']['max_calls']=99
            def __call__(self,state):return dict(tool='STOP',mode=None,candidate_id=None,reason='metadata_contract')
        service=self.service();sent=envelope(max_calls=1)
        service.register_bundle_policy(sent['continuation_policy_id'],Policy())
        result=service.query_bundle(sent)
        self.assertEqual(result['cost']['capability_calls'],1)
        self.assertEqual(result['request'],sent)
        self.assertEqual(service._task_provenance,provenance())
        self.assertEqual(len(seen),2)


if __name__=='__main__':unittest.main()
