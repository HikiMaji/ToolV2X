"""Numeric integration on a real tiny CPU planner and paid synthetic P/F service."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from test_evidence_ledger import LedgerFixture
from test_method_episode import limits, feedback
from test_structured_driver import tiny_spec, features
from test_task_spec import provenance
from test_vehicle_tools import window
from planning.structured_driver import StructuredPlanner
from planning.method_episode import run_task_episode, validate_limits
from planning.method_controls import control_spec, capture_control_prefix, make_control_policy
from evaluation.framework import evaluate_method_task
from evaluation.planning import trajectory_metrics

SCENE = 'testoutput_CAV_data_2022-03-15-10-09-50_0'


class StructuredEpisodeTests(LedgerFixture):
    def inputs(self, name='feedback', p_processing='observations_only', **options):
        torch.manual_seed(12)
        driver = StructuredPlanner(tiny_spec(p_processing=p_processing))
        # Keep random untrained outputs comfortably inside the physical contract.
        with torch.no_grad():
            driver.model.output_head[-1].weight.mul_(.01)
            driver.model.output_head[-1].bias.mul_(.01)
        config = limits(version='toolv2x_interaction_v2', receiver_spec=driver.spec.to_dict(),
                        driver_version={k: driver.provenance['model_version'][k] for k in ('name','revision')})
        ego = window('ego'); ego['scene'] = SCENE
        peer = window(); peer['scene'] = SCENE
        from tools.vehicle import VehicleTools
        self.reads = []
        def peer_window():
            self.reads.append('peer')
            return peer
        service = VehicleTools(peer_window, self.predictor, SCENE, 10, 'peer', task_provenance=provenance())
        feature = {k:v.numpy() if isinstance(v, torch.Tensor) else v for k,v in features().items()}
        spec = control_spec(name, **options)
        return dict(local_window=ego, local_prediction=self.predictor(ego), motion=dict(speed_mps=2., yaw_rate_rps=0.),
            features=feature, service=service, predictor=self.predictor, driver=driver,
            policy=make_control_policy(spec, feedback), limits=config, local_provenance=provenance(),
            sample_id='sample', control_spec=spec)

    def task(self, ep):
        return dict(row=dict(sample_id='sample', scene=SCENE, g=10, role='train'), episode=ep,
                    status='completed' if ep['status']=='completed' else 'failed', inputs=dict(local_prediction_seconds=.1))

    def test_numeric_feedback_uses_paid_evidence_actual_prior_and_cost(self):
        args = self.inputs()
        try:
            validate_limits(args['limits'])
        except ValueError as exc:
            self.fail('numeric limits must admit the full structured specification: '+str(exc))
        initial = []
        def progress(ep):
            if ep['events'][-1]['kind']=='driver_completed' and len(ep['plans'])==1:
                initial.append(list(self.reads))
        ep = run_task_episode(**args, on_progress=progress)
        self.assertEqual(ep['status'], 'completed', ep.get('error'))
        self.assertEqual(initial, [[]])
        self.assertEqual(len(ep['plans']), 3)
        self.assertEqual([r['tool'] for r in ep['requests']], ['P','F'])
        self.assertEqual(ep['execution_kind'], 'structured_numeric')
        for i,p in enumerate(ep['plans']):
            self.assertFalse({'q9_raw','q9_executed','q8_executed'} & set(p['output']))
            self.assertEqual(p['prepared']['previous_plan'], None if i==0 else ep['plans'][i-1]['output']['waypoints'])
            self.assertEqual(p['prepared']['previous_parent_refs'], [] if i==0 else ep['plans'][i-1]['output']['parent_refs'])
        self.assertIsNone(ep['decisions'][0]['state']['current_plan']['raw'])
        self.assertEqual(ep['ledger_snapshots'][0]['version'], 'toolv2x_evidence_ledger_v2')
        self.assertEqual(ep['ledger_snapshots'][1]['derived_fields'], [])
        result = evaluate_method_task(self.task(ep), None)
        self.assertTrue(result['task_success'], result)
        self.assertTrue(result['cost_complete'], result)
        self.assertEqual(result['output_tokens'], 0)
        self.assertEqual(result['input_tokens'], 0)
        self.assertEqual(result['numeric_output_points'], 18)
        self.assertGreater(result['numeric_token_count'], 0)
        json.dumps(ep, allow_nan=False)

    def test_exact_repeat_freezes_prior_and_refinement_changes_only_prior(self):
        for name in ('exact_repeat','self_refinement'):
            ep = run_task_episode(**self.inputs(name))
            self.assertEqual(ep['status'], 'completed', ep.get('error'))
            self.assertEqual(len(ep['plans']), 3)
            self.assertEqual(ep['requests'], [])
            first = ep['plans'][0]['prepared']
            for i in (1,2):
                prepared = ep['plans'][i]['prepared']
                if name=='exact_repeat':
                    self.assertEqual(prepared, first)
                else:
                    self.assertEqual(prepared['previous_plan'], ep['plans'][i-1]['output']['waypoints'])
                    for key in first:
                        if key not in ('previous_plan','previous_parent_refs'):
                            self.assertEqual(prepared[key], first[key])
                    self.assertNotEqual(ep['plans'][i]['output']['waypoints'], ep['plans'][i-1]['output']['waypoints'])

    def test_driver_kind_mismatch_rejected_before_private_read(self):
        args = self.inputs(); args['limits'] = limits()
        with self.assertRaises(ValueError):
            run_task_episode(**args)
        self.assertEqual(self.reads, [])

    def test_prefix_metrics_use_all_points_and_preserve_endpoints(self):
        label = dict(valid=[True]*6, times_seconds=[.5,1.,1.5,2.,2.5,3.], waypoints=[[0.,0.]]*6)
        result = trajectory_metrics([[float(i),0.] for i in range(1,7)], label, None)
        self.assertEqual(result['L2_1'], 2.)
        self.assertEqual(result.get('got_prefix_L2_1s'), 1.5)
        self.assertEqual(result['got_prefix_L2_2s'], 2.5)
        self.assertEqual(result['got_prefix_L2_3s'], 3.5)
        self.assertEqual(result['got_prefix_L2_avg'], 2.5)
        label['valid'][0]=False
        result = trajectory_metrics([[float(i),0.] for i in range(1,7)], label, None)
        self.assertEqual(result['L2_1'], 2.)
        self.assertTrue(all(result[k] is None for k in result if k.startswith('got_prefix_')))

    def test_shared_paid_prefix_controls_and_one_shot_bundle(self):
        args = self.inputs('evidence_refinement')
        prefix = capture_control_prefix(**args)
        for name in ('feedback','frozen_feedback','current_old','current_union','evidence_refinement','exact_repeat','self_refinement'):
            spec = control_spec(name, repeat_after_calls=1)
            policy = lambda s: dict(tool='F', mode='old' if name=='current_old' else 'union' if name=='current_union' else 'current', reason='test')
            ep = run_task_episode(**dict(args, control_spec=spec, policy=make_control_policy(spec, policy),
                service=args['service'].fork_task(), branch_id=name), prefix=prefix)
            self.assertEqual(ep['status'], 'completed', ep.get('error'))
            self.assertEqual(ep['plans'][:2], prefix['episode']['plans'])
            self.assertEqual(len(ep['plans']), 3)
            self.assertTrue(evaluate_method_task(self.task(ep), None)['task_success'])
        bundle = dict(continuation_policy_id='diagnostic', candidate_sources=['initial','slower'],
            slowdown_scale=.5, max_candidates=2, wrapper_reserve_bytes=4096, extra_generation='self_refinement')
        args = self.inputs('one_shot', bundle=bundle)
        from planning.method_controls import diagnostic_bundle_continuation
        args['service'].register_bundle_policy('diagnostic', diagnostic_bundle_continuation)
        args['policy'] = lambda s: dict(tool='P', mode='current', reason='bundle')
        ep = run_task_episode(**args)
        self.assertEqual(ep['status'], 'completed', ep.get('error'))
        self.assertEqual(len(ep['requests']), 1)
        self.assertEqual(len(ep['plans']), 3)
        cost = evaluate_method_task(self.task(ep), None)
        self.assertEqual(cost['rpc_rounds'], 1)
        self.assertEqual(cost['calls'], 2)
        self.assertTrue(cost['task_success'], cost)
        self.assertTrue(cost['cost_complete'], cost)

    def test_archived_numeric_payload_and_model_identity_are_bound(self):
        from planning.query_data import _check_episode, _semantic_binding
        args = self.inputs()
        ep = run_task_episode(**args)
        spec = dict(limits=args['limits'], control=args['control_spec'], local_provenance=provenance())
        runtime = dict(driver=args['driver'].provenance, predictor=self.predictor.descriptor)
        binding = _semantic_binding(runtime, spec)
        self.assertEqual(binding['driver']['settings'].get('model_version'), runtime['driver']['model_version'])
        config = dict(binding=binding, runtime_binding=runtime)
        _check_episode(self.task(ep), config)
        # A internally valid replacement Z still has to agree with the actual source ledger.
        changed = copy.deepcopy(args['local_window'])
        changed['states'][...,0] += .3
        ledger = self.e.new_ledger(changed, self.predictor(changed), predictor=self.predictor,
            local_provenance=provenance(), p_processing='observations_only', include_local_history=True)
        prepared = args['driver'].prepare_input(args['features'],args['motion'],ledger,args['limits']['receiver_spec'])
        bad = copy.deepcopy(ep)
        bad['plans'][0]['prepared']=prepared
        bad['plans'][0]['output']['prepared_input']=copy.deepcopy(prepared)
        with self.assertRaises(ValueError):
            _check_episode(self.task(bad), config)
        self.assertEqual(evaluate_method_task(self.task(bad), None)['artifact_status'], 'invalid_artifact')
        for field in ('parent_refs','driver_cost'):
            bad=copy.deepcopy(ep)
            if field=='parent_refs': bad['plans'][0]['output'][field]=[]
            else: bad['plans'][0]['output'][field]['numeric_token_count'] += 1
            with self.assertRaises(ValueError):
                _check_episode(self.task(bad), config)
        for model_change in ('status','optimizer_steps'):
            changed=copy.deepcopy(runtime)
            changed['driver']['model_version']['training'][model_change]='trained' if model_change=='status' else 1
            self.assertNotEqual(_semantic_binding(changed,spec), binding)
        legacy=dict(driver=dict(model_class='old_got',decoding='direct'), predictor=self.predictor.descriptor)
        with self.assertRaises(ValueError):
            _semantic_binding(legacy, spec)

    def test_failed_driver_attempt_keeps_prepared_and_paid_cost(self):
        args=self.inputs()
        actual=args['driver'].plan_prepared
        calls=[]
        def fail(features,prepared):
            calls.append(1)
            if len(calls)==2: raise RuntimeError('synthetic inference interruption')
            return actual(features,prepared)
        args['driver'].plan_prepared=fail
        ep=run_task_episode(**args)
        self.assertEqual(ep['status'],'driver_error')
        self.assertEqual(len(ep['plans']),2)
        self.assertIsNone(ep['plans'][-1]['output'])
        self.assertEqual(ep['plans'][-1]['prepared']['previous_plan'],ep['plans'][0]['output']['waypoints'])
        self.assertGreater(ep['cost']['response_bytes'],0)
        result=evaluate_method_task(self.task(ep),None)
        self.assertEqual(result['artifact_status'],'recorded_failure')
        self.assertFalse(result['cost_complete'])
        self.assertEqual(result['driver_calls'],2)

    def test_readonly_feature_conversion_does_not_share_mutable_storage(self):
        from planning.structured_driver import _numeric_tensor, _bool_tensor
        for array, convert in ((np.array([1.],np.float32),_numeric_tensor), (np.array([True]),_bool_tensor)):
            array.setflags(write=False)
            tensor=convert(array,(1,),'cpu','fixture')
            tensor[0]=0
            self.assertEqual(array[0],1)

    def test_numeric_branch_archive_and_state_features(self):
        from planning.query_data import collect_branches, _branch_archive
        from planning.query_value import state_features, feature_spec
        from common.audit_protocol import recording
        args=self.inputs()
        rows=[dict(sample_id='sample',scene=SCENE,g=10,role='train',physical_split='train',local_frame=10,
            ego_motion=args['motion'],feature_read_paths=[],motion_read_paths=[])]
        def load(row,directory):
            return {**{k:args[k] for k in ('local_window','local_prediction','motion','features','service')},
                    'metadata':dict(local_prediction_seconds=.1)}
        runtime=SimpleNamespace(driver=args['driver'],predictor=self.predictor,load_inputs=load,provenance=dict(kind='synthetic_cpu'))
        spec=dict(version='toolv2x_query_collection_v1',limits=args['limits'],local_provenance=provenance(),
                  control=args['control_spec'],recording_folds={recording(SCENE):'fold0'})
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'branches'
            collect_branches(rows,root,runtime,spec)
            archive=_branch_archive(root)
            self.assertGreater(len(archive[2]),1)
            for record in archive[2]:
                values=state_features(record['state'],feature_spec())
                self.assertTrue(np.isfinite(values['values']).all())
            self.assertEqual(json.loads((root/'progress.json').read_text())['status'],'completed')

    def test_paid_receipt_must_match_real_wire_and_all_stages_keep_binding(self):
        from planning.query_data import _check_episode, _semantic_binding
        args=self.inputs();ep=run_task_episode(**args)
        spec=dict(limits=args['limits'],control=args['control_spec'],local_provenance=provenance())
        runtime=dict(driver=args['driver'].provenance,predictor=self.predictor.descriptor)
        config=dict(binding=_semantic_binding(runtime,spec),runtime_binding=runtime)
        for change in ('receipt_wire','response_wire','cost_event','nested_cost','numeric_kind'):
            bad=copy.deepcopy(ep)
            if change=='receipt_wire':
                bad['ledger_snapshots'][1]['receipts'][0]['wire_text']='{}'
            elif change=='response_wire':
                bad['responses'][0]['wire_hex']=b'{}'.hex()
            elif change=='cost_event':
                next(e for e in bad['cost_events'] if e['kind']=='driver')['generation_cost']['seconds'] += 1.
            elif change=='nested_cost':
                seconds=bad['plans'][0]['driver_attempt_seconds']+1.
                bad['plans'][0]['output']['driver_cost']['seconds']=seconds
                next(e for e in bad['cost_events'] if e['kind']=='driver')['generation_cost']['seconds']=seconds
            else:
                bad['driver_provenance']['driver_kind']='got'
            with self.subTest(change=change), self.assertRaises(ValueError):
                _check_episode(self.task(bad),config)

    def test_explicit_ego_and_legacy_fixed_controls_share_numeric_driver(self):
        for name, options in [('ego_max_context',{})]+[('legacy_v2',dict(baseline=b)) for b in ('Ego','P','F','PF','rule')]:
            args=self.inputs(name,**options)
            ep=run_task_episode(**args)
            self.assertEqual(ep['status'],'completed',ep.get('error'))
            self.assertTrue(evaluate_method_task(self.task(ep),None)['task_success'])
            self.assertLessEqual(len(ep['plans']),3)
            if name=='ego_max_context' or options.get('baseline')=='Ego': self.assertEqual(ep['requests'],[])

    def test_numeric_bundle_archive_can_spend_common_refinement_budget(self):
        from planning.bundle_data import collect_bundle_branches, validate_bundle_archive, bundle_collection_spec
        from common.audit_protocol import recording
        bundle=dict(continuation_policy_id='diagnostic',candidate_sources=['initial'],slowdown_scale=.5,
                    max_candidates=1,wrapper_reserve_bytes=4096,extra_generation='self_refinement')
        args=self.inputs('one_shot',bundle=bundle)
        rows=[dict(sample_id='sample',scene=SCENE,g=10,role='train',physical_split='train',local_frame=10,
            ego_motion=args['motion'],feature_read_paths=[],motion_read_paths=[])]
        runtime=SimpleNamespace(driver=args['driver'],predictor=self.predictor,provenance=dict(kind='synthetic_cpu'),
            load_inputs=lambda row,directory:dict(**{k:args[k] for k in ('local_window','local_prediction','motion','features','service')},
                metadata=dict(local_prediction_seconds=.1)))
        spec=dict(version='toolv2x_bundle_collection_v1',limits=args['limits'],local_provenance=provenance(),
                  control=args['control_spec'],recording_folds={recording(SCENE):'fold0'})
        try: bundle_collection_spec(spec)
        except ValueError as exc: self.fail('numeric one-shot archive must allow shared refinements: '+str(exc))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'bundle'
            collect_bundle_branches(rows,root,runtime,spec)
            _,_,branches=validate_bundle_archive(root)
            completed=[b for b in branches if b['status']=='completed']
            self.assertTrue(completed)
            for b in completed:
                task=json.loads((root/b['path']).read_text())
                self.assertEqual(len(task['episode']['plans']),3)
                self.assertEqual(len(task['episode']['requests']),1)
                self.assertTrue(evaluate_method_task(task,None)['task_success'])

    def test_real_checkpoint_loader_rejects_old_got_query_binding(self):
        from planning.query_data import _semantic_binding
        from planning.query_value import training_config, load_query_policy
        from test_query_data import utility
        from common.audit_protocol import recording
        args=self.inputs()
        numeric=_semantic_binding(dict(driver=args['driver'].provenance,predictor=self.predictor.descriptor),
            dict(limits=args['limits'],control=args['control_spec'],local_provenance=provenance()))
        legacy=_semantic_binding(dict(driver=dict(model_class='got',decoding='direct'),predictor=self.predictor.descriptor),
            dict(limits=limits(),control=args['control_spec'],local_provenance=provenance()))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'synthetic_query.pt'
            config=training_config(checkpoint=path,train_recordings=[recording(SCENE)],binding=legacy,utility_spec=utility())
            torch.save(dict(version='toolv2x_query_checkpoint_v1',config=config,kind='shared'),path)
            with self.assertRaisesRegex(ValueError,'binding drift'):
                load_query_policy(path,expected_binding=numeric)

    def test_numeric_invalid_answers_and_costs_preserve_failed_attempt(self):
        for failure in ('bounds','shape','cost'):
            with self.subTest(failure=failure):
                args=self.inputs()
                original=args['driver'].plan_prepared
                def invalid(features,prepared):
                    output=original(features,prepared)
                    if failure=='bounds': output['waypoints']=[[10000.,0.]]*6
                    elif failure=='shape': output['waypoints']=[[0.,0.]]
                    else: output['driver_cost']['seconds']='unknown'
                    return output
                args['driver'].plan_prepared=invalid
                ep=run_task_episode(**args)
                self.assertEqual(ep['status'],'invalid_plan')
                self.assertEqual(len(ep['plans']),1)
                self.assertEqual(ep['requests'],[])
                self.assertIsNotNone(ep['plans'][0]['output'])
                if failure!='cost':
                    self.assertEqual(evaluate_method_task(self.task(ep),None)['artifact_status'],'recorded_failure')
                else:
                    self.assertFalse(ep['cost']['complete'])

    def test_local_history_arrays_and_explicit_paid_p_local_mode(self):
        args=self.inputs(p_processing='local_mtr')
        times=np.linspace(-1.,0.,11)
        args['features'].update(ego_pose_history=np.column_stack((2.*times,np.zeros((11,2)))),
            ego_pose_history_valid=np.ones(11,dtype=bool),ego_pose_history_times=times)
        args['policy']=lambda s: dict(tool='P' if s['previous_plan'] is None else 'STOP',
            mode='current' if s['previous_plan'] is None else None,reason='test_p_local')
        ep=run_task_episode(**args)
        self.assertEqual(ep['status'],'completed',ep.get('error'))
        self.assertTrue(ep['ledger_snapshots'][-1]['derived_fields'])
        for plan in ep['plans']:
            self.assertEqual(plan['prepared']['ego_history_used']['states'],args['features']['ego_pose_history'].tolist())
        result=evaluate_method_task(self.task(ep),None)
        self.assertTrue(result['task_success'],result)
        self.assertGreater(result['receiver_model_seconds'],0.)

    def test_live_numeric_prefix_rejects_coordinated_prepared_replacement(self):
        args=self.inputs()
        prefix=capture_control_prefix(**args,after_calls=0)
        changed=copy.deepcopy(args['local_window']);changed['states'][...,0]+=.2
        ledger=self.e.new_ledger(changed,self.predictor(changed),predictor=self.predictor,local_provenance=provenance(),
            p_processing='observations_only',include_local_history=True)
        prepared=args['driver'].prepare_input(args['features'],args['motion'],ledger,args['limits']['receiver_spec'])
        prefix['episode']['plans'][0]['prepared']=prepared
        prefix['episode']['plans'][0]['output']['prepared_input']=copy.deepcopy(prepared)
        with self.assertRaises(ValueError):
            run_task_episode(**args,prefix=prefix)
        self.assertEqual(self.reads,[])
