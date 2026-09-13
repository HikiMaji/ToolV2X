"""T9 offline collection contracts; all external model computations are fixtures."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import test_control_bundle as bundle_fixture
envelope,stop=bundle_fixture.envelope,bundle_fixture.stop
from test_query_data import QueryFixture
from test_method_episode import EvidenceDriver, limits
from test_task_spec import provenance
from test_vehicle_tools import window
from common.audit_protocol import recording
from planning.method_controls import control_spec
from probe.kinematic_tools import encode


class BundlePrefixTests(unittest.TestCase):
    setUp=bundle_fixture.ControlBundleTests.setUp
    register=bundle_fixture.ControlBundleTests.register
    def test_actual_first_return_is_shared_but_each_suffix_is_executed_and_charged(self):
        module=importlib.import_module('tools.control_bundle')
        self.assertTrue(hasattr(module,'capture_bundle_prefix'))
        env=envelope();self.register(stop)
        prefix=module.capture_bundle_prefix(self.service,env)
        self.assertEqual(len(self.prediction_windows),0)  # P never invokes MTR.
        self.assertEqual(len(self.service.task_records),1)
        self.assertEqual(prefix.record['status'],'prefix_ready')
        outputs=[]
        for action in (dict(tool='STOP',mode=None,candidate_id=None),
                       dict(tool='F',mode='current',candidate_id='tau0'),
                       dict(tool='F',mode='current',candidate_id='tau0')):
            branch=prefix.fork(lambda state,a=action:dict(a,reason='offline_enumeration'))
            self.assertEqual(branch.task_records,[])  # No online preview before dispatch.
            result=branch.query_bundle(env);outputs.append(result)
            self.assertEqual(result['request_wire'],encode(env))
            self.assertEqual(result['cost']['rpc_rounds'],1)
            self.assertEqual(result['cost']['capability_calls'],1 if action['tool']=='STOP' else 2)
            self.assertEqual(result['collection_reuse']['prefix_service_seconds'],prefix.record['cost']['service_seconds'])
            self.assertEqual(result['primitive_responses'][0]['wire'],bytes.fromhex(prefix.record['primitive_responses'][0]['wire_hex']))
            self.assertEqual(len(result['wire']),result['cost']['response_bytes'])
        self.assertEqual(len(self.prediction_windows),2)
        self.assertNotEqual(outputs[1]['primitive_responses'][1]['wire'],outputs[2]['primitive_responses'][1]['wire'])
        wrong=copy.deepcopy(env);wrong['candidates'][1]['waypoints']=[[1.,0.]]*6
        with self.assertRaises(ValueError):prefix.fork(stop).query_bundle(wrong)
        with self.assertRaises(ValueError):module.capture_bundle_prefix(self.service,env)


    def test_fork_deployment_cost_charges_setup_once_like_a_fresh_bundle(self):
        from tools import control_bundle as module
        from tools.vehicle import VehicleTools
        env=envelope();self.register(stop)
        original=module.validate_bundle_envelope;clock=[0.]
        def validation(value):
            clock[0]+=20.  # Controlled expensive setup, never a model-speed claim.
            return original(value)
        with patch.object(module,'validate_bundle_envelope',side_effect=validation),patch.object(module,'perf_counter',side_effect=lambda:clock[0]):
            prefix=module.capture_bundle_prefix(self.service,env)
            result=prefix.fork(stop).query_bundle(env)
            fresh=VehicleTools(window,self.predictor,'scene',10,'peer',task_provenance=provenance())
            fresh.register_bundle_policy(env['continuation_policy_id'],stop)
            reference=fresh.query_bundle(env)
        reuse=result['collection_reuse']
        self.assertEqual(reuse['prefix_service_seconds']+reuse['suffix_service_seconds'],reference['cost']['service_seconds'])
        self.assertEqual(reuse['suffix_service_seconds'],result['cost']['service_seconds'])


class BundleDataTests(QueryFixture):
    def bundle_collection(self, root, *, invalid_stage=None, cap=None):
        q=importlib.import_module('planning.bundle_data')
        scene='testoutput_CAV_data_2022-03-15-10-09-50_0'
        rows=[dict(sample_id='sample',scene=scene,g=10,role='train',physical_split='train',local_frame=10,
            ego_motion=dict(speed_mps=2.,yaw_rate_rps=0.),feature_read_paths=[],motion_read_paths=[])]
        driver=EvidenceDriver([],invalid_stage=invalid_stage)
        reads=['ego']
        def load(row,directory):
            from tools.vehicle import VehicleTools
            result=self.inputs();result['local_window'].update(scene=row['scene'],g=row['g'])
            result['local_prediction']=self.predictor(result['local_window'])
            peer=window();peer.update(scene=row['scene'],g=row['g'])
            def peer_window():reads.append('peer_'+str(len(reads)));return peer
            result['service']=VehicleTools(peer_window,self.predictor,row['scene'],row['g'],'peer',task_provenance=provenance())
            for key in ('driver','predictor','limits','local_provenance','sample_id','control_spec'):result.pop(key)
            result['metadata']=dict(local_prediction_seconds=.25,window_reads=reads)
            return result
        runtime=SimpleNamespace(driver=driver,predictor=self.predictor,load_inputs=load,provenance=dict(kind='synthetic_contract'))
        spec=dict(version='toolv2x_bundle_collection_v1',limits=limits(),local_provenance=provenance(),
            recording_folds={recording(scene):'fold0'},control=control_spec('one_shot',driver_calls=2,
            bundle=dict(continuation_policy_id='fixture_bundle_v1',candidate_sources=['initial','slower'],slowdown_scale=.5,
                max_candidates=2,wrapper_reserve_bytes=2048,extra_generation=None)))
        if cap:spec['limits']['execution_spec']['max_request_bytes']=cap
        q.collect_bundle_branches(rows,root,runtime,spec)
        return runtime,rows,spec

    def test_collection_has_real_terminals_same_first_packet_and_no_ego_revision_on_peer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'branches';runtime,rows,spec=self.bundle_collection(root)
            states=self.read_rows(root/'online_states.jsonl');branches=self.read_rows(root/'branches.jsonl')
            self.assertEqual(len(states),2)
            self.assertEqual(len(runtime.driver.inputs),1+len(branches))
            self.assertEqual({s['state']['envelope']['first_request']['tool'] for s in states},{'P','F'})
            from evaluation.framework import _method_cost
            for state in states:
                self.assertEqual(set(state['state']),{'envelope','first_response','available_actions'})
                owned=[b for b in branches if b['state_id']==state['state_id']]
                self.assertEqual(len(owned),len(state['state']['available_actions']))
                prefix=json.loads((root/state['provider_prefix_path']).read_text())
                for b in owned:
                    task=json.loads((root/b['path']).read_text());ep=task['episode']
                    self.assertEqual(len(ep['plans']),2)
                    self.assertEqual(len(ep['requests']),1)
                    packet=json.loads(bytes.fromhex(ep['responses'][0]['wire_hex']))
                    self.assertEqual(packet['responses'][0]['packet'],state['state']['first_response'])
                    self.assertEqual({k:packet['continuation'][k] for k in ('tool','mode','candidate_id')},b['action'])
                    cost=_method_cost(task);self.assertTrue(cost['cost_complete'],cost['cost_issues'])
                    event=next(e for e in ep['cost_events'] if e['kind']=='service')
                    self.assertAlmostEqual(cost['service_seconds'],event['service_cost']['service_seconds']+prefix['record']['cost']['service_seconds'])
                    self.assertEqual(cost['rpc_rounds'],1)
                    self.assertEqual(task['inputs']['window_reads'],['ego','peer_1' if state['state']['envelope']['first_request']['tool']=='P' else 'peer_2'])

    def test_terminal_labels_recomputed_from_outer_wire_cost_and_final_driver(self):
        q=importlib.import_module('planning.bundle_data')
        from test_query_data import utility
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,rows,_=self.bundle_collection(root/'branches')
            labels=root/'labels';labels.mkdir()
            label={k:rows[0][k] for k in ('sample_id','scene','role','g')}
            label.update(times_seconds=[.5,1.,1.5,2.,2.5,3.],valid=[True]*6,waypoints=[[float(i),0.] for i in range(1,7)])
            (labels/'train.jsonl').write_text(json.dumps(label)+'\n')
            table=q.make_bundle_targets(root/'branches',labels,utility(),root/'targets')
            self.assertEqual(table,q.load_measured_bundle_targets(root/'targets/targets.json'))
            self.assertTrue(table['rows'])
            for row in table['rows']:
                if row['action']['tool']=='STOP':self.assertEqual(row['target'],0.)
                else:
                    self.assertEqual(row['incremental_cost']['request_bytes'],0)
                    self.assertAlmostEqual(row['target'],row['stop_loss']-row['terminal_loss']-sum(
                        utility()['cost_weights'][k]*v for k,v in row['incremental_cost'].items()))
            path=root/'targets/targets.json';bad=copy.deepcopy(table);bad['rows'][1]['terminal_loss']+=1
            path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):q.load_measured_bundle_targets(path)

    def test_missing_labels_and_failed_terminal_are_retained_without_success_substitution(self):
        q=importlib.import_module('planning.bundle_data')
        from test_query_data import utility
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,rows,_=self.bundle_collection(root/'branches',invalid_stage=2)
            labels=root/'labels';labels.mkdir();(labels/'train.jsonl').write_text('')
            table=q.make_bundle_targets(root/'branches',labels,utility(),root/'missing')
            self.assertEqual(table['rows'],[])
            coverage=json.loads((root/'missing/coverage.json').read_text());self.assertEqual(len(coverage['excluded_states']),2)
            label={k:rows[0][k] for k in ('sample_id','scene','role','g')}
            label.update(times_seconds=[.5,1.,1.5,2.,2.5,3.],valid=[True]*6,waypoints=[[float(i),0.] for i in range(1,7)])
            (labels/'train.jsonl').write_text(json.dumps(label)+'\n')
            table=q.make_bundle_targets(root/'branches',labels,utility(),root/'targets')
            failed=[r for r in table['rows'] if r['terminal_loss']==utility()['failure_loss']]
            self.assertEqual(len(failed),1)

    def test_online_rejects_gt_before_loading_and_archive_rejects_missing_candidate(self):
        q=importlib.import_module('planning.bundle_data')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,rows,spec=self.bundle_collection(root/'branches')
            rows[0]['future']=[[1,2]]
            with self.assertRaises(ValueError):q.collect_bundle_branches(rows,root/'bad',None,spec)
            entries=self.read_rows(root/'branches/branches.jsonl');entries.pop()
            (root/'branches/branches.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in entries))
            with self.assertRaises(ValueError):q.validate_bundle_archive(root/'branches')

    def test_failed_initial_and_public_infeasible_first_keep_full_sample_coverage(self):
        q=importlib.import_module('planning.bundle_data')
        for changes in (dict(invalid_stage=0),dict(cap=1)):
            with self.subTest(changes=changes),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)/'branches';runtime,_,_=self.bundle_collection(root,**changes)
                q.validate_bundle_archive(root)
                self.assertEqual(len(runtime.driver.inputs),1)
                self.assertEqual(self.read_rows(root/'online_states.jsonl'),[])
                self.assertEqual(self.read_rows(root/'branches.jsonl'),[])
                self.assertEqual(json.loads((root/'progress.json').read_text())['completed_samples'],1)

    def test_measured_adapter_rejects_unaudited_declarations_and_cost_rebinding(self):
        q=importlib.import_module('planning.bundle_data')
        from planning import query_value as v
        from test_query_data import utility
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,rows,_=self.bundle_collection(root/'branches')
            labels=root/'labels';labels.mkdir()
            label={k:rows[0][k] for k in ('sample_id','scene','role','g')}
            label.update(times_seconds=[.5,1.,1.5,2.,2.5,3.],valid=[True]*6,waypoints=[[float(i),0.] for i in range(1,7)])
            (labels/'train.jsonl').write_text(json.dumps(label)+'\n')
            table=q.make_bundle_targets(root/'branches',labels,utility(),root/'targets')
            groups=list(table['recording_folds'])
            config=v.training_config(checkpoint=str(root/'unused.pt'),train_recordings=groups,binding=table['binding'],utility_spec=utility(),steps=1,policy_id='fixture_bundle_v1')
            examples=v._bundle_examples(root/'targets/targets.json',config)
            self.assertEqual(examples['coverage']['available_targets'],len(table['rows']))
            with self.assertRaises(ValueError):v._bundle_examples(table,config)
            bad=copy.deepcopy(table);bad.pop('archive');bad['version']='toolv2x_bundle_targets_v1'
            with self.assertRaises(ValueError):v._bundle_examples(bad,config)
            branch=self.read_rows(root/'branches/branches.jsonl')[1];path=root/'branches'/branch['path']
            terminal=json.loads(path.read_text());event=next(e for e in terminal['episode']['cost_events'] if e['kind']=='service')
            event['collection_reuse']['prefix_service_seconds']+=1.;path.write_text(json.dumps(terminal))
            with self.assertRaises(ValueError):v._bundle_examples(root/'targets/targets.json',config)


class BundleModelTests(QueryFixture):
    """Separate tiny CPU checkpoint integration; no original driver or MTR models."""
    bundle_collection=BundleDataTests.bundle_collection

    def test_measured_table_fits_real_tiny_checkpoint_and_freezes_identical_policy(self):
        import torch
        from planning import bundle_data as q,query_value as v
        from planning.method_run_spec import prepare_value_checkpoints
        from test_query_data import utility
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,rows,_=self.bundle_collection(root/'branches')
            labels=root/'labels';labels.mkdir()
            label={k:rows[0][k] for k in ('sample_id','scene','role','g')}
            label.update(times_seconds=[.5,1.,1.5,2.,2.5,3.],valid=[True]*6,waypoints=[[float(i),0.] for i in range(1,7)])
            (labels/'train.jsonl').write_text(json.dumps(label)+'\n')
            table=q.make_bundle_targets(root/'branches',labels,utility(),root/'targets')
            groups=list(table['recording_folds'])
            config=v.training_config(checkpoint=str(root/'bundle.pt'),train_recordings=groups,binding=table['binding'],utility_spec=utility(),steps=2,policy_id='fixture_bundle_v1')
            fitted=v.fit_bundle_continuation(root/'targets/targets.json',groups,config)
            manifest,policies=prepare_value_checkpoints(dict(bundle=root/'bundle.pt'),root/'frozen',table['binding'],utility())
            state=table['rows'][0]['state']
            self.assertEqual(fitted(state),policies['bundle'](state))
            prepare_value_checkpoints(dict(bundle=root/'bundle.pt'),root/'resume',table['binding'],utility(),
                resume_from=root/'frozen',previous_manifest=manifest)
            saved=v._torch_load(root/'bundle.pt')
            saved['model'][next(iter(saved['model']))].add_(1.)
            torch.save(saved,root/'bundle.pt')
            with self.assertRaises(ValueError):prepare_value_checkpoints(dict(bundle=root/'bundle.pt'),root/'changed',table['binding'],utility(),
                resume_from=root/'frozen',previous_manifest=manifest)


class RuntimeIdentityTests(unittest.TestCase):
    def test_actual_driver_training_identity_is_bound_but_paths_can_move(self):
        from test_method_run_spec import MethodRunSpecTests
        from planning.method_run_spec import freeze_method_run_spec,validate_runtime_binding
        helper=MethodRunSpecTests();spec=copy.deepcopy(helper.spec())
        driver=spec['runtime_binding']['driver']
        driver.update(checkpoint='/old/weights',local_training_state=dict(epoch=1,steps=20,examples_seen=160,
            data='/old/data',best_checkpoint='/old/weights'))
        frozen=freeze_method_run_spec(spec,helper.rows())
        moved=copy.deepcopy(spec['runtime_binding']);moved['driver']['checkpoint']='/relocated/weights'
        moved['driver']['local_training_state'].update(data='/relocated/data',best_checkpoint='/relocated/weights')
        self.assertNotIn('/old/',json.dumps(frozen))
        validate_runtime_binding(moved,frozen)
        moved['driver']['local_training_state']['examples_seen']+=1
        with self.assertRaises(ValueError):validate_runtime_binding(moved,frozen)
