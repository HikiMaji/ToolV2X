"""T7 contracts: actual services/driver executor, synthetic external models only."""
import copy
import importlib
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from itertools import count

import numpy as np

from test_evidence_ledger import LedgerFixture
from test_method_episode import EvidenceDriver, limits
from test_task_spec import provenance
from test_vehicle_tools import window
from planning.method_controls import capture_control_prefix, control_spec
from planning.method_episode import run_task_episode
from common.audit_protocol import recording
from types import SimpleNamespace


class QueryFixture(LedgerFixture):
    def module(self):
        self.assertTrue((Path(__file__).resolve().parents[1]/'src/planning/query_data.py').exists(),'T7 collection missing')
        return importlib.import_module('planning.query_data')

    def collection(self, root, *, invalid_stage=None, limit_changes=None, audit_reads=False, driver_metadata=None, control_name='feedback'):
        scene='testoutput_CAV_data_2022-03-15-10-09-50_0'
        rows=[dict(sample_id='sample',scene=scene,g=10,role='train',physical_split='train',local_frame=10,
            ego_motion=dict(speed_mps=2.,yaw_rate_rps=0.),feature_read_paths=[],motion_read_paths=[])]
        driver=EvidenceDriver([],invalid_stage=invalid_stage)
        if driver_metadata:driver.provenance=dict(driver.provenance,**driver_metadata)
        reads=['ego']
        def load(row,directory):
            result=self.inputs()
            result['local_window'].update(scene=row['scene'],g=row['g'])
            result['local_prediction']=self.predictor(result['local_window'])
            peer=window();peer.update(scene=row['scene'],g=row['g'])
            from tools.vehicle import VehicleTools
            def peer_window():
                if audit_reads:reads.append('peer_'+str(len(reads)))
                return peer
            result['service']=VehicleTools(peer_window,self.predictor,row['scene'],row['g'],'peer',task_provenance=provenance())
            for key in ('driver','predictor','limits','local_provenance','sample_id','control_spec'):result.pop(key)
            result['metadata']=dict(local_prediction_seconds=.25)
            if audit_reads:result['metadata']['window_reads']=reads
            return result
        runtime=SimpleNamespace(driver=driver,predictor=self.predictor,load_inputs=load,provenance=dict(kind='synthetic_contract'))
        config=dict(version='toolv2x_query_collection_v1',limits=limits(),local_provenance=provenance(),
            control=control_spec(control_name),recording_folds={recording(scene):'fold0',
                'testoutput_CAV_data_2022-03-16-10-00-00':'fold1'})
        if limit_changes:config['limits']['execution_spec'].update(limit_changes)
        ticks=count()
        with patch('planning.method_episode.perf_counter',side_effect=lambda:float(next(ticks))):
            self.module().collect_branches(rows,root,runtime,config)
        return runtime,rows,config

    def read_rows(self,path):
        return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]

    def inputs(self):
        ego=window('ego')
        return dict(local_window=ego,local_prediction=self.predictor(ego),
            motion=dict(speed_mps=2.,yaw_rate_rps=0.),features=dict(active_agent_mask=np.array([[[True],[False]]])),
            service=self.service(),predictor=self.predictor,driver=EvidenceDriver([]),limits=limits(),
            local_provenance=provenance(),sample_id='sample',token_counter=lambda s:len(s)//4,
            control_spec=control_spec('feedback'))


class BranchTests(QueryFixture):
    def test_semantic_binding_ignores_paths_load_time_and_run_local_predictor_id(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            self.collection(root/'one',limit_changes=dict(max_request_bytes=1),
                driver_metadata=dict(checkpoint='/first/checkpoint',loading_seconds=1.))
            self.setUp()
            self.collection(root/'two',limit_changes=dict(max_request_bytes=1),
                driver_metadata=dict(checkpoint='/second/checkpoint',loading_seconds=9.))
            one=json.loads((root/'one/config.json').read_text());two=json.loads((root/'two/config.json').read_text())
            self.assertEqual(one['binding'],two['binding'])
            self.assertNotEqual(one['runtime_binding'],two['runtime_binding'])

    def test_branch_input_audit_keeps_own_reads_and_no_sibling_reads(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'branches';self.collection(root,audit_reads=True)
            states=self.read_rows(root/'online_states.jsonl');branches=self.read_rows(root/'branches.jsonl')
            for state in states:
                expected=['ego'] if state['depth']==0 else ['ego','peer_1' if state['state_id'].endswith('P_current') else 'peer_2']
                prefix=json.loads((root/state['prefix_path']).read_text())
                self.assertEqual(prefix['inputs']['window_reads'],expected)
                if state['depth']==1:
                    for branch in branches:
                        if branch['state_id']==state['state_id']:
                            task=json.loads((root/branch['path']).read_text())
                            self.assertEqual(task['inputs']['window_reads'],expected)

    def test_online_index_rejects_labels_and_recording_split_conflicts_before_runtime(self):
        q=self.module()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);_,rows,spec=self.collection(root/'base')
            for bad in (dict(rows[0],future=[[1,2]]),dict(rows[0],sample_id='other',role='validation',
                    scene=rows[0]['scene'].replace('_0','_1'))):
                supplied=[bad] if 'future' in bad else [rows[0],bad]
                with self.assertRaises(ValueError):q.collect_branches(supplied,root/'bad',None,spec)
            spec['recording_folds']={rows[0]['scene']:'fold0'}
            with self.assertRaises(ValueError):q.collect_branches(rows,root/'bad',None,spec)

    def test_collection_does_not_read_labels_and_changed_gt_does_not_change_queries(self):
        import io
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);labels=root/'offline_labels';labels.mkdir()
            original=io.open;traces=[]
            for i in (0,1):
                (labels/'train.jsonl').write_text('different future '+str(i))
                def guarded(path,*args,**kw):
                    if 'offline_labels' in str(path):raise AssertionError('collection read future labels')
                    return original(path,*args,**kw)
                with patch('io.open',guarded):
                    runtime,_,_=self.collection(root/('run'+str(i)))
                traces.append([p['q9_prompt'] for p in runtime.driver.inputs])
            self.assertEqual(traces[0],traces[1])

    def test_cli_collect_method_uses_the_original_runtime_adapter_without_labels(self):
        from planning import run_framework as runner
        self.assertTrue(hasattr(runner,'collect_method'),'collect-method CLI missing')
        with tempfile.TemporaryDirectory() as temp:
            from test_method_episode import InteractionRunnerTests
            helper=InteractionRunnerTests();helper.setUp()
            root=Path(temp);data,rows,spec_path=helper.make_data(root)
            spec=json.loads(spec_path.read_text())
            spec.update(version='toolv2x_query_collection_v1',control=control_spec('feedback'),
                recording_folds={recording(rows[0]['scene']):'fold0'})
            spec_path.write_text(json.dumps(spec))
            runtime=helper.runtime(rows)
            load=runtime.load_inputs
            def inputs(row,directory):
                value=load(row,directory);value['token_counter']=lambda s:len(s)//4
                value['metadata']['local_prediction_seconds']=.1
                return value
            runtime.load_inputs=inputs
            with patch.object(runner,'_load_interaction_runtime',return_value=runtime) as loader:
                runner.collect_method(data,root/'out',Path('/fixture/driver'),spec_path,role='validation',per_recording=1)
            self.assertEqual(loader.call_count,1)
            self.assertEqual(len(runtime.driver.inputs),11)
            self.assertEqual(json.loads((root/'out/progress.json').read_text())['completed_samples'],1)

    def test_full_tree_has_eleven_driver_calls_shared_prefixes_and_no_sibling_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'branches';runtime,rows,config=self.collection(root)
            branches=self.read_rows(root/'branches.jsonl');states=self.read_rows(root/'online_states.jsonl')
            self.assertEqual(len(runtime.driver.inputs),11)
            self.assertEqual(len(states),3)
            self.assertEqual(len(branches),13)
            self.assertEqual({s['physical_recording'] for s in states},{recording(rows[0]['scene'])})
            for state in states:
                siblings=[b for b in branches if b['state_id']==state['state_id']]
                prefix=json.loads((root/state['prefix_path']).read_text())['episode']
                for b in siblings:
                    task=json.loads((root/b['path']).read_text());ep=task['episode']
                    self.assertEqual(ep['plans'][:len(prefix['plans'])],prefix['plans'])
                    self.assertEqual(ep['responses'][:len(prefix['responses'])],prefix['responses'])
                    if state['depth']==1 and prefix['requests'][0]['tool']=='P' and b['action']['tool']=='F':
                        event=next(e for e in ep['cost_events'] if e['kind']=='service' and e['stage']==1)
                        self.assertFalse(event['service_cost']['within_decision_forecast_cache_hit'])
            progress=json.loads((root/'progress.json').read_text())
            self.assertEqual(progress['physical_driver_attempts'],11)
            self.assertEqual(progress['status'],'completed')

    def test_failed_first_branch_has_no_children_and_preserves_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'branches';runtime,_,_=self.collection(root,invalid_stage=1)
            branches=self.read_rows(root/'branches.jsonl')
            failed=next(b for b in branches if b['status']=='failed')
            self.assertIsNone(failed['next_state_id'])
            task=json.loads((root/failed['path']).read_text())
            self.assertEqual(task['episode']['status'],'invalid_plan')
            self.assertIsNone(task['episode']['final_plan_id'])
            self.assertEqual(len(runtime.driver.inputs),7)

    def test_infeasible_actions_are_preserved_without_paid_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'branches';runtime,_,_=self.collection(root,limit_changes=dict(max_request_bytes=1))
            branches=self.read_rows(root/'branches.jsonl')
            self.assertEqual(len(runtime.driver.inputs),1)
            self.assertEqual(len(branches),3)
            blocked=[b for b in branches if not b['feasible']]
            self.assertEqual(len(blocked),2)
            self.assertTrue(all(b['status']=='infeasible' and b['reason']=='request_byte_limit' and b['path'] is None for b in blocked))

    def test_initial_prefix_is_shared_without_regeneration_or_peer_reads(self):
        self.assertIn('after_calls',inspect.signature(capture_control_prefix).parameters,'initial prefix API missing')
        inputs=self.inputs()
        prefix=capture_control_prefix(after_calls=0,**inputs,
            policy=lambda s:dict(tool='STOP',mode=None,reason='initial'))
        self.assertEqual(len(inputs['driver'].inputs),1)
        self.assertIsNone(inputs['service']._task_window)
        for tool in ('STOP','P','F'):
            branch=run_task_episode(**dict(inputs,service=inputs['service'].fork_task()),prefix=prefix,
                policy=lambda s,t=tool:dict(tool=t if not s['response_receipts'] else 'STOP',
                    mode='current' if t!='STOP' and not s['response_receipts'] else None,reason='branch'))
            self.assertEqual(branch['status'],'completed',branch.get('error'))
            self.assertEqual(branch['plans'][0],prefix['episode']['plans'][0])
        self.assertEqual(len(inputs['driver'].inputs),3)
        self.assertEqual(inputs['service'].task_records,[])
        self.assertIsNone(inputs['service']._task_forecast)

    def test_service_fork_keeps_paid_prefix_but_isolates_sibling_forecast_and_receipts(self):
        service=self.service()
        self.assertTrue(hasattr(service,'fork_task'),'branch service API missing')
        service.query_task(self.request())
        left,right=service.fork_task(),service.fork_task()
        self.assertEqual(left.task_records,right.task_records)
        left.query_task(self.request(tool='F',request_id='q1'))
        self.assertIsNotNone(left._task_forecast)
        self.assertIsNone(right._task_forecast)
        self.assertIsNone(service._task_forecast)
        self.assertEqual(len(right.task_records),1)
        self.assertEqual(len(right._task_receipts),1)
        right.query_task(self.request(tool='F',request_id='q1'))
        self.assertFalse(right.task_records[-1]['cost']['within_decision_forecast_cache_hit'])
        self.assertNotEqual(left.task_records[-1]['response']['receipt_id'],right.task_records[-1]['response']['receipt_id'])


def utility():
    return dict(version='toolv2x_query_utility_v1',label_coverage='all_six',
        quality_weights=dict(ADE3=1.,FDE3=0.),failure_loss=10.,
        cost_weights=dict(request_bytes=0.,response_bytes=0.,total_compute_seconds=.01))


class Teacher:
    def __init__(self,binding,action=None):
        self.provenance=dict(version='toolv2x_continuation_v1',teacher_id='fixture_teacher',held_out_fold='fold0',
            training_recordings=['testoutput_CAV_data_2022-03-16-10-00-00'],binding=copy.deepcopy(binding),
            utility_spec=utility(),feature_spec=dict(version='fixture_visible_v1'),
            model_version=dict(name='frozen_fixture',revision='v1'))
        self.seen=[]
        self.action=action or dict(tool='F',mode='current',reason='frozen_choice')

    def __call__(self,state):
        self.seen.append(copy.deepcopy(state))
        state['current_plan']['waypoints'][0][0]=9876.
        return copy.deepcopy(self.action)


class TargetTests(QueryFixture):
    def test_terminal_supervision_preserves_both_improvement_and_harm(self):
        class SlopeDriver(EvidenceDriver):
            def plan_prepared(self,features,prepared):
                stage=len(self.inputs)
                output=super().plan_prepared(features,prepared)
                if stage in (2,3):
                    slope=0. if stage==2 else .2
                    output['waypoints']=[[float(i),slope*i] for i in range(1,7)]
                    output['q9_raw']='The suggested trajectory is: '+repr(output['waypoints'])
                return output
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with patch(__name__+'.EvidenceDriver',SlopeDriver):
                branches,labels,_=self.label_data(root)
            label=json.loads((labels/'train.jsonl').read_text());label['waypoints']=[[float(i),0.] for i in range(1,7)]
            (labels/'train.jsonl').write_text(json.dumps(label)+'\n')
            result=self.module().make_terminal_targets(branches,labels,utility(),root/'targets')
            rows={r['action']['tool']:r for r in result if r['state_id']=='s000000__P_current' and r['action']['mode']=='current'}
            self.assertAlmostEqual(rows['P']['target'],.27)
            self.assertAlmostEqual(rows['F']['target'],-.43)

    def test_control_grammars_keep_frozen_mask_and_bind_old_union_requests(self):
        q=self.module()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for name,calls in (('frozen_feedback',7),('current_old',11),('current_union',11)):
                current=root/name;current.mkdir()
                branches,labels,_=self.label_data(current,control_name=name)
                records=self.read_rows(branches/'branches.jsonl')
                self.assertEqual(json.loads((branches/'progress.json').read_text())['physical_driver_attempts'],calls)
                rows=q.make_terminal_targets(branches,labels,utility(),current/'targets')
                self.assertTrue(rows)
                modes={r['action']['mode'] for r in rows}
                self.assertNotIn('change',modes)
                if name!='frozen_feedback':self.assertIn('old' if name=='current_old' else 'union',modes)

    def test_copied_gt_field_in_saved_state_is_rejected_before_teacher(self):
        q=self.module()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);branches,labels,teacher=self.label_data(root)
            states=self.read_rows(branches/'online_states.jsonl');entries=self.read_rows(branches/'branches.jsonl')
            state=next(s for s in states if s['depth']==1)
            state['state']['candidate_gt_ADE3']=[0.,100.]
            stop=next(b for b in entries if b['state_id']==state['state_id'] and b['action']['tool']=='STOP')
            path=branches/stop['path'];task=json.loads(path.read_text())
            task['episode']['decisions'][-1]['state']=copy.deepcopy(state['state'])
            path.write_text(json.dumps(task))
            (branches/'online_states.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in states))
            with self.assertRaises(ValueError):q.make_first_targets(branches,labels,{'fold0':teacher},utility(),root/'bad')
            self.assertEqual(teacher.seen,[])

    def test_initial_failure_keeps_sample_denominator_without_inventing_targets(self):
        q=self.module()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);branches,labels,teacher=self.label_data(root,invalid_stage=0)
            result=q.make_first_targets(branches,labels,{'fold0':teacher},utility(),root/'targets')
            self.assertEqual(result,[])
            coverage=json.loads((root/'targets/coverage.json').read_text())
            self.assertEqual(coverage['expected_source_samples'],1)
            self.assertEqual(len(coverage['unlabelled_initial_samples']),1)
            self.assertEqual(teacher.seen,[])
            branch=self.read_rows(branches/'branches.jsonl')[0]
            path=branches/branch['path'];task=json.loads(path.read_text());task['status']='completed';path.write_text(json.dumps(task))
            with self.assertRaises(ValueError):q.make_first_targets(branches,labels,{'fold0':teacher},utility(),root/'bad')

    def test_missing_first_continuation_tree_cannot_be_relabelled_as_stop(self):
        q=self.module()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);branches,labels,teacher=self.label_data(root)
            entries=self.read_rows(branches/'branches.jsonl');states=self.read_rows(branches/'online_states.jsonl')
            parent=next(b for b in entries if b['next_state_id'] is not None)
            child_id=parent['next_state_id'];parent['next_state_id']=None
            entries=[b for b in entries if b['state_id']!=child_id]
            states=[s for s in states if s['state_id']!=child_id]
            for name,rows in [('branches.jsonl',entries),('online_states.jsonl',states)]:
                (branches/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaises(ValueError):q.make_terminal_targets(branches,labels,utility(),root/'bad')
            with self.assertRaises(ValueError):q.make_first_targets(branches,labels,{'fold0':teacher},utility(),root/'bad')

    def test_missing_weights_and_old_or_missing_costs_are_rejected(self):
        q=self.module()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);branches,labels,_=self.label_data(root)
            bad=utility();del bad['cost_weights']['total_compute_seconds']
            with self.assertRaises(ValueError):q.make_terminal_targets(branches,labels,bad,root/'bad')
            for bad in (dict(utility(),failure_loss=True),dict(utility(),failure_loss=float('inf'))):
                with self.assertRaises(ValueError):q.make_terminal_targets(branches,labels,bad,root/'bad')
            branch=next(b for b in self.read_rows(branches/'branches.jsonl') if b['state_id'] and b['state_id'].count('__')==1 and b['action']['tool']=='F')
            path=branches/branch['path'];task=json.loads(path.read_text())
            event=next(e for e in task['episode']['cost_events'] if e['kind']=='control' and e['stage']==1)
            del event['timing_version'];path.write_text(json.dumps(task))
            with self.assertRaises(ValueError):q.make_terminal_targets(branches,labels,utility(),root/'bad')

    def test_missing_z_and_duplicate_or_relabelled_branches_are_rejected(self):
        q=self.module()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);branches,labels,_=self.label_data(root)
            index=branches/'branches.jsonl';original=index.read_text()
            index.write_text(original+original.splitlines()[0]+'\n')
            with self.assertRaises(ValueError):q.make_terminal_targets(branches,labels,utility(),root/'bad')
            index.write_text(original)
            branch=next(b for b in self.read_rows(index) if b['state_id'] and b['state_id'].count('__')==1 and b['action']['tool']=='F')
            path=branches/branch['path'];task=json.loads(path.read_text());saved=copy.deepcopy(task)
            del task['episode']['plans'][-1]['prepared']['remote_evidence_used'];path.write_text(json.dumps(task))
            with self.assertRaises(ValueError):q.make_terminal_targets(branches,labels,utility(),root/'bad')
            path.write_text(json.dumps(saved))
            entries=self.read_rows(index)
            other=next(b for b in entries if b['state_id']==branch['state_id'] and b['action']==dict(tool='P',mode=branch['action']['mode']))
            for b in entries:
                if b['branch_id']==branch['branch_id']:b['path']=other['path']
            index.write_text(''.join(json.dumps(b)+'\n' for b in entries))
            with self.assertRaises(ValueError):q.make_terminal_targets(branches,labels,utility(),root/'bad')

    def test_teacher_recording_overlap_binding_drift_and_illegal_action_are_rejected(self):
        q=self.module()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);branches,labels,teacher=self.label_data(root)
            original=copy.deepcopy(teacher.provenance)
            for update in (dict(training_recordings=['testoutput_CAV_data_2022-03-15-10-09-50']),
                    dict(training_recordings=['testoutput_CAV_data_2022-03-15-10-09-50_1']),
                    dict(training_recordings=[]),dict(binding={}),dict(utility_spec=dict(utility(),failure_loss=20.))):
                teacher.provenance=dict(original,**update)
                with self.assertRaises(ValueError):q.make_first_targets(branches,labels,{'fold0':teacher},utility(),root/'bad')
            teacher.provenance=original;teacher.action=dict(tool='I',mode='current',reason='unknown')
            with self.assertRaises(ValueError):q.make_first_targets(branches,labels,{'fold0':teacher},utility(),root/'bad')

    def test_teacher_selects_before_gt_is_read_and_stop_only_charges_first_request(self):
        import io
        q=self.module()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);branches,labels,teacher=self.label_data(root)
            teacher.action=dict(tool='STOP',mode=None,reason='frozen_stop')
            original=io.open;traces=[];targets=[]
            for i in (0,1):
                if i:
                    label=json.loads((labels/'train.jsonl').read_text());label['waypoints']=[[float(n),0.] for n in range(1,7)]
                    (labels/'train.jsonl').write_text(json.dumps(label)+'\n')
                teacher.seen=[]
                def guarded(path,*a,**kw):
                    if str(labels) in str(path):self.assertEqual(len(teacher.seen),2,'GT read before continuation decisions')
                    return original(path,*a,**kw)
                with patch('io.open',guarded):
                    result=q.make_first_targets(branches,labels,{'fold0':teacher},utility(),root/('target'+str(i)))
                traces.append(teacher.seen);targets.append(result)
                for row in result:
                    if row['action']['tool']!='STOP':self.assertAlmostEqual(row['incremental_cost']['total_compute_seconds'],8.)
            self.assertEqual(traces[0],traces[1])
            self.assertNotEqual(targets[0][1]['target'],targets[1][1]['target'])

    def label_data(self,root,**kwargs):
        branches=root/'branches';runtime,rows,config=self.collection(branches,**kwargs)
        labels=root/'labels';labels.mkdir()
        label={k:rows[0][k] for k in ('sample_id','scene','role','g')}
        label.update(times_seconds=[.5,1.,1.5,2.,2.5,3.],valid=[True]*6,
            waypoints=[[float(i),i*.1] for i in range(1,7)])
        (labels/'train.jsonl').write_text(json.dumps(label)+'\n')
        binding=json.loads((branches/'config.json').read_text())['binding']
        return branches,labels,Teacher(binding)

    def test_terminal_targets_use_incremental_cost_and_stop_zero(self):
        q=self.module();self.assertTrue(hasattr(q,'make_terminal_targets'),'terminal supervision API missing')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);branches,labels,_=self.label_data(root)
            q.make_terminal_targets(branches,labels,utility(),root/'targets')
            rows=self.read_rows(root/'targets/targets.jsonl')
            self.assertEqual(len(rows),10)
            for row in rows:
                if row['action']['tool']=='STOP':self.assertEqual(row['target'],0.)
                else:
                    self.assertAlmostEqual(row['incremental_cost']['total_compute_seconds'],8.)
                    self.assertAlmostEqual(row['target'],-.08)

    def test_first_targets_follow_frozen_teacher_even_when_stop_is_better(self):
        q=self.module();self.assertTrue(hasattr(q,'make_first_targets'),'first supervision API missing')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);branches,labels,teacher=self.label_data(root)
            q.make_first_targets(branches,labels,{'fold0':teacher},utility(),root/'targets')
            rows=self.read_rows(root/'targets/targets.jsonl')
            self.assertEqual(len(rows),3)
            for row in rows:
                if row['action']['tool']=='STOP':self.assertEqual(row['target'],0.)
                else:
                    self.assertEqual(row['continuation_action']['tool'],'F')
                    self.assertAlmostEqual(row['incremental_cost']['total_compute_seconds'],14.)
                    self.assertAlmostEqual(row['target'],.21)
            self.assertEqual(len(teacher.seen),2)
            self.assertTrue(all(len(s['response_receipts'])==1 for s in teacher.seen))
            self.assertTrue(all('targets' not in s and 'branches' not in s and 'labels' not in s for s in teacher.seen))
            states=self.read_rows(branches/'online_states.jsonl')
            self.assertTrue(all(s['state']['current_plan']['waypoints'][0][0]!=9876. for s in states))

    def test_first_failure_uses_failure_loss_without_teacher_or_fake_children(self):
        q=self.module();self.assertTrue(hasattr(q,'make_first_targets'),'first supervision API missing')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);branches,labels,teacher=self.label_data(root,invalid_stage=1)
            q.make_first_targets(branches,labels,{'fold0':teacher},utility(),root/'targets')
            rows=self.read_rows(root/'targets/targets.jsonl')
            failed=next(r for r in rows if r['terminal_status']=='recorded_failure')
            self.assertEqual(failed['terminal_loss'],10.)
            self.assertIsNone(failed['terminal_plan_id'])
            self.assertIsNone(failed['continuation_action'])
            self.assertEqual(len(teacher.seen),1)



if __name__=='__main__':
    unittest.main()
