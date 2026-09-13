"""T6 causal control contracts; external models are synthetic, services are real."""
import copy
import importlib
import json
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import test_method_episode as fixture
import test_evidence_ledger as ledger_fixture
from planning.method_episode import run_task_episode
from tools.vehicle import VehicleTools


class ControlTests(unittest.TestCase):
    def setUp(self):
        ledger_fixture.LedgerFixture.setUp(self)

    def controls(self):
        self.assertTrue((ROOT/'src/planning/method_controls.py').is_file(), 'T6 controls missing')
        return importlib.import_module('planning.method_controls')

    def run_control(self, name, *, options=None, policy=fixture.feedback, driver=None, prefix=None, service=None, progress=None, capture=False):
        c = self.controls()
        config = c.control_spec(name, **(options or {}))
        ego = fixture.window('ego')
        service = service or VehicleTools(lambda: fixture.window(), self.predictor, 'scene', 10, 'peer', task_provenance=fixture.provenance())
        self.last_service = service
        driver = driver or fixture.EvidenceDriver([])
        self.last_driver = driver
        inputs=dict(local_window=ego,local_prediction=self.predictor(ego),motion=dict(speed_mps=2.,yaw_rate_rps=0.),
            features=dict(active_agent_mask=np.array([[[True],[False]]])),service=service,predictor=self.predictor,driver=driver,
            policy=c.make_control_policy(config,policy),limits=fixture.limits(),local_provenance=fixture.provenance(),
            sample_id='sample',branch_id=name,token_counter=lambda p:len(p)//4,
            control_spec=config,on_progress=progress)
        if capture:
            self.assertTrue(hasattr(c,'capture_control_prefix'),'same-input live prefix capture missing')
            return c.capture_control_prefix(**inputs)
        return run_task_episode(**inputs,prefix=prefix)

    def test_exact_repeat_spends_three_generations_and_no_remote_calls(self):
        ep = self.run_control('exact_repeat')
        self.assertEqual(ep['status'], 'completed')
        self.assertEqual(len(ep['plans']), 3)
        self.assertEqual(ep['requests'], [])
        self.assertEqual(self.last_service.task_records, [])
        self.assertEqual(len({p['prepared']['q9_prompt'] for p in ep['plans']}), 1)
        self.assertTrue(all(p['prepared']['admission_report']==ep['plans'][0]['prepared']['admission_report'] for p in ep['plans']))
        self.assertEqual(ep['final_plan_id'], 'plan_2')

    def test_refinement_keeps_evidence_and_uses_actual_previous_answer(self):
        ep = self.run_control('self_refinement')
        self.assertEqual(ep['status'], 'completed')
        self.assertEqual(len(ep['plans']), 3)
        for i in (1,2):
            p = ep['plans'][i]['prepared']
            self.assertEqual(p['previous_plan_origin'], 'model_generated')
            self.assertEqual(p['refinement']['previous_plan'], ep['plans'][i-1]['output']['waypoints'])
            for key in ('evidence_used','remote_evidence_used','admission_report'):
                self.assertEqual(p[key], ep['plans'][0]['prepared'][key])
        self.assertEqual(ep['requests'], [])
        self.assertEqual(ep['final_plan_id'], 'plan_2')

    def test_new_evidence_and_fixed_evidence_reserve_same_refinement_slot(self):
        fixed = self.run_control('self_refinement')
        new = self.run_control('evidence_refinement')
        self.assertEqual(new['status'], 'completed')
        self.assertEqual(len(new['plans']), 3)
        self.assertEqual(fixed['plans'][0]['prepared']['q9_prompt'], new['plans'][0]['prepared']['q9_prompt'])
        for ep in (fixed,new):
            self.assertEqual({p['prepared']['refinement']['slot_tokens'] for p in ep['plans']}, {256})
        self.assertNotEqual(new['plans'][1]['prepared']['remote_evidence_used'], fixed['plans'][1]['prepared']['remote_evidence_used'])

    def test_frozen_feedback_masks_policy_and_request_but_keeps_real_driver_revision(self):
        seen = []
        def policy(s):
            seen.append(copy.deepcopy(s))
            return dict(tool='P' if len(seen)==1 else 'F', mode='current', reason='fixed_test')
        ep = self.run_control('frozen_feedback', policy=policy)
        self.assertEqual(ep['status'], 'completed')
        tau0,tau1 = [p['output']['waypoints'] for p in ep['plans'][:2]]
        self.assertNotEqual(tau0,tau1)
        self.assertEqual(seen[1]['current_plan']['waypoints'], tau0)
        self.assertEqual(seen[1]['previous_plan']['waypoints'], tau0)
        self.assertEqual(ep['requests'][1]['tau_new'], tau0)
        self.assertFalse(any(a['mode']=='change' for a in seen[1]['available_actions']))

    def test_old_union_have_equal_candidate_counts_and_bind_actual_pair(self):
        counts=[]
        for name,mode in [('feedback','change'),('current_old','old'),('current_union','union')]:
            seen=[]
            def policy(s):
                seen.append(copy.deepcopy(s))
                return dict(tool='P' if len(seen)==1 else 'F', mode='current' if len(seen)==1 else mode, reason='compare')
            ep=self.run_control(name,policy=policy)
            self.assertEqual(ep['status'],'completed')
            counts.append([len(s['available_actions']) for s in seen])
            self.assertEqual(ep['requests'][1]['mode'],mode)
            self.assertEqual(ep['requests'][1]['tau_old'],ep['plans'][0]['output']['waypoints'])
            self.assertEqual(ep['requests'][1]['tau_new'],ep['plans'][1]['output']['waypoints'])
        self.assertEqual(counts,[[3,5]]*3)

    def test_invalid_extra_generation_never_falls_back_to_prior_answer(self):
        ep=self.run_control('exact_repeat',driver=fixture.EvidenceDriver([],invalid_stage=1))
        self.assertEqual(ep['status'],'invalid_plan')
        self.assertEqual(len(ep['plans']),2)
        self.assertIsNone(ep['final_plan_id'])
        self.assertEqual(ep['requests'],[])

    def test_real_first_response_prefix_is_reused_for_second_request_ablation(self):
        prefix=self.run_control('feedback',capture=True)
        service=self.last_service
        driver=self.last_driver
        a=self.run_control('feedback',prefix=prefix,service=copy.deepcopy(service),driver=driver)
        b=self.run_control('exact_repeat',options=dict(repeat_after_calls=1),prefix=prefix,service=copy.deepcopy(service),driver=driver)
        self.assertEqual(a['plans'][:2],b['plans'][:2])
        self.assertEqual(a['responses'][0],b['responses'][0])
        self.assertEqual(len(a['requests']),2)
        self.assertEqual(len(b['requests']),1)
        self.assertEqual(len(self.last_driver.inputs),4)
        self.assertEqual(b['plans'][2]['prepared']['q9_prompt'],b['plans'][1]['prepared']['q9_prompt'])

    def test_ego_max_context_removes_remote_reservation(self):
        ep=self.run_control('ego_max_context')
        self.assertEqual(ep['status'],'completed')
        self.assertEqual(len(ep['plans']),1)
        self.assertEqual(ep['plans'][0]['prepared']['receiver_spec']['peer_reserve'],0)
        self.assertEqual(ep['requests'],[])

    def test_control_costs_count_extra_generations_without_service(self):
        from evaluation.framework import _method_cost
        ep=self.run_control('self_refinement')
        cost=_method_cost(dict(episode=ep,inputs=dict(local_prediction_seconds=.5)))
        self.assertTrue(cost['cost_complete'],cost)
        self.assertEqual(cost['driver_calls'],3)
        self.assertEqual(cost['calls'],0)
        self.assertEqual(cost['generation_seconds'],6.)
        self.assertEqual(cost['request_bytes'],0)

    def test_wrapper_rejects_forged_previous_output_and_overflow_without_dropping_z(self):
        from planning import inputs
        ep=self.run_control('exact_repeat')
        self.assertTrue(hasattr(inputs,'build_refinement_input'))
        p=ep['plans'][0]['prepared']
        previous=dict(plan_id='plan_0',waypoints=ep['plans'][0]['output']['waypoints'],raw='unparseable')
        with self.assertRaises(ValueError):
            inputs.build_refinement_input(p,previous,token_counter=lambda s:len(s)//4,slot_tokens=256)
        previous['raw']=ep['plans'][0]['output']['q9_raw']
        original=copy.deepcopy(p)
        with self.assertRaises(ValueError):
            inputs.build_refinement_input(p,previous,token_counter=lambda s:len(s)//4,slot_tokens=1)
        self.assertEqual(p,original)

    def test_repeat_after_first_call_keeps_received_z_not_ego_z(self):
        ep=self.run_control('exact_repeat',options=dict(repeat_after_calls=1))
        self.assertEqual(ep['status'],'completed')
        self.assertEqual(len(ep['requests']),1)
        self.assertIsNotNone(ep['plans'][1]['prepared']['remote_evidence_used'])
        self.assertEqual(ep['plans'][1]['prepared']['q9_prompt'],ep['plans'][2]['prepared']['q9_prompt'])

    def test_old_five_schedulers_have_explicit_common_receiver_arms(self):
        for baseline,wanted in [('Ego',[]),('P',['P']),('F',['F']),('PF',['P','F']),('rule',['P'])]:
            with self.subTest(baseline=baseline):
                ep=self.run_control('legacy_v2',options=dict(baseline=baseline))
                self.assertEqual(ep['status'],'completed',ep.get('error'))
                self.assertEqual([r['tool'] for r in ep['requests']],wanted)
                self.assertTrue(all(r['mode']=='roi' for r in ep['requests']))
                self.assertEqual(ep['plans'][-1]['prepared']['input_layout'],'source_blocks_v2')

    def test_bundle_episode_has_one_external_round_two_primitives_and_no_mid_driver(self):
        c=self.controls()
        ego=fixture.window('ego')
        service=VehicleTools(lambda:fixture.window(),self.predictor,'scene',10,'peer',task_provenance=fixture.provenance())
        self.assertTrue(hasattr(service,'register_bundle_policy'))
        seen=[]
        def continuation(s):
            seen.append(s)
            return dict(tool='F',mode='current',candidate_id='initial',reason='paid_first_return')
        service.register_bundle_policy('test_conditional_v1',continuation)
        bundle=dict(continuation_policy_id='test_conditional_v1',candidate_sources=['initial','slower','constant_motion'],
                    slowdown_scale=.5,max_candidates=3,wrapper_reserve_bytes=2048,extra_generation=None)
        ep=self.run_control('one_shot',options=dict(bundle=bundle),service=service)
        self.assertEqual(ep['status'],'completed',ep.get('error'))
        self.assertEqual(len(ep['plans']),2)
        self.assertEqual(len(ep['requests']),1)
        self.assertEqual(len(seen),1)
        from evaluation.framework import _method_cost
        cost=_method_cost(dict(episode=ep,inputs=dict(local_prediction_seconds=.5)))
        self.assertTrue(cost['cost_complete'],cost)
        self.assertEqual(cost['rpc_rounds'],1)
        self.assertEqual(cost['calls'],2)
        self.assertEqual(cost['driver_calls'],2)
        from probe.kinematic_tools import encode
        self.assertEqual(cost['request_bytes'],len(encode(ep['requests'][0])))
        self.assertEqual(cost['response_bytes'],len(bytes.fromhex(ep['responses'][0]['wire_hex'])))
        self.assertEqual(len(ep['ledger_snapshots'][-1]['receipts']),2)

    def test_interact_registers_explicit_control_spec_and_archives_t5_rows(self):
        import tempfile
        from unittest.mock import patch
        from planning import run_framework as runner
        from planning.context import build_task_plan_input
        from evaluation.framework import evaluate_method
        c=self.controls()
        helper=fixture.InteractionRunnerTests()
        helper.setUp()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            data,rows,spec=helper.make_data(root)
            raw=json.loads(spec.read_text())
            raw['control']=c.control_spec('self_refinement')
            spec.write_text(json.dumps(raw))
            def build(*a,**kw):
                kw['token_counter']=lambda p:len(p)//4
                return build_task_plan_input(*a,**kw)
            with patch.object(runner,'_load_interaction_runtime',return_value=helper.runtime(rows)), \
                 patch('planning.method_episode.build_task_plan_input',side_effect=build):
                # Refinement uses the same injected counter as the prepared input.
                from planning import inputs
                actual=inputs.build_refinement_input
                def refine(*a,**kw):
                    kw['token_counter']=lambda p:len(p)//4
                    return actual(*a,**kw)
                with patch.object(inputs,'build_refinement_input',side_effect=refine):
                    runner.interact(data,root/'run',Path('/fixture/driver'),spec,per_recording=0)
                    raw['control']=c.control_spec('ego_max_context')
                    spec.write_text(json.dumps(raw))
                    runner.interact(data,root/'ego_run',Path('/fixture/driver'),spec,per_recording=0)
            labels=root/'labels';labels.mkdir()
            (labels/'validation.jsonl').write_text('')
            evaluate_method(root/'run',root/'evaluation',labels)
            records=runner.read_jsonl(root/'evaluation/rows.jsonl')
            self.assertEqual(len(records),2)
            self.assertEqual({r['branch_id'] for r in records},{'self_refinement'})
            self.assertTrue(all(r['task_success'] and r['driver_calls']==3 and r['calls']==0 for r in records))
            evaluate_method(root/'ego_run',root/'ego_evaluation',labels)
            ego_rows=runner.read_jsonl(root/'ego_evaluation/rows.jsonl')
            self.assertTrue(all(r['task_success'] and r['driver_calls']==1 and r['branch_id']=='ego_max_context' for r in ego_rows))


    def test_prefix_rejects_changed_feature_values_and_different_driver(self):
        c=self.controls()
        prefix=self.run_control('feedback',capture=True)
        with self.assertRaises(ValueError):
            self.run_control('feedback',prefix=prefix,service=copy.deepcopy(self.last_service))
        from planning.method_controls import validate_control_prefix
        changed=copy.deepcopy(prefix['features'])
        changed['active_agent_mask'][0,1,0]=True
        with self.assertRaises(ValueError):
            validate_control_prefix(prefix,changed,prefix['driver'])


    def test_interrupted_control_decision_cost_is_unknown(self):
        from evaluation.framework import _method_cost
        saved=[]
        def interrupt(ep):
            if ep['events'][-1]['kind']=='decision':
                saved.append(ep)
                raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):self.run_control('feedback',progress=interrupt)
        cost=_method_cost(dict(episode=saved[-1],inputs=dict(local_prediction_seconds=.5)))
        self.assertIsNone(cost['control_seconds'])
        self.assertIsNone(cost['total_compute_seconds'])


    def test_control_driver_quota_prevents_a_request_without_revision_capacity(self):
        for name in ('feedback','evidence_refinement','exact_repeat'):
            with self.subTest(name=name):
                ep=self.run_control(name,options=dict(driver_calls=1))
                self.assertEqual(len(ep['plans']),1)
                self.assertEqual(ep['requests'],[])


    def test_second_request_refinement_pair_shares_real_prefix_and_slot(self):
        prefix=self.run_control('evidence_refinement',capture=True)
        service=self.last_service;driver=self.last_driver
        new=self.run_control('evidence_refinement',prefix=prefix,service=copy.deepcopy(service),driver=driver)
        fixed=self.run_control('self_refinement',options=dict(repeat_after_calls=1),prefix=prefix,service=copy.deepcopy(service),driver=driver)
        self.assertEqual(new['plans'][:2],fixed['plans'][:2])
        self.assertEqual(len(new['requests']),2)
        self.assertEqual(len(fixed['requests']),1)
        self.assertEqual(fixed['plans'][2]['prepared']['admission_report'],fixed['plans'][1]['prepared']['admission_report'])
        self.assertEqual(new['plans'][2]['prepared']['refinement'],fixed['plans'][2]['prepared']['refinement'])

    def test_refinement_pair_cannot_add_an_unreserved_slot_to_a_direct_prefix(self):
        prefix=self.run_control('feedback',capture=True)
        with self.assertRaises(ValueError):
            self.run_control('self_refinement',options=dict(repeat_after_calls=1),prefix=prefix,
                service=copy.deepcopy(self.last_service),driver=self.last_driver)

    def test_one_shot_can_use_remaining_generation_quota_without_buying_again(self):
        for extra in ('exact_repeat','self_refinement'):
            service=VehicleTools(lambda:fixture.window(),self.predictor,'scene',10,'peer',task_provenance=fixture.provenance())
            service.register_bundle_policy('stop_after_return',lambda s:dict(tool='STOP',mode=None,candidate_id=None,reason='done'))
            bundle=dict(continuation_policy_id='stop_after_return',candidate_sources=['initial'],slowdown_scale=.5,
                max_candidates=1,wrapper_reserve_bytes=2048,extra_generation=extra)
            ep=self.run_control('one_shot',options=dict(bundle=bundle),service=service)
            self.assertEqual(ep['status'],'completed',ep.get('error'))
            self.assertEqual(len(ep['plans']),3)
            self.assertEqual(len(ep['requests']),1)
            self.assertEqual(ep['plans'][1]['prepared']['admission_report'],ep['plans'][2]['prepared']['admission_report'])
            self.assertEqual(len(service.task_records),1)


class ControlModelTests(unittest.TestCase):
    """Original tokenizer/planner contract only; _generate is replaced, no model loaded."""
    def test_original_planner_accepts_explicit_refinement_and_rejects_tampering(self):
        from transformers import AutoTokenizer
        from planning.v2vgot import CHECKPOINT,V2VGoTPlanner
        from planning.context import build_task_plan_input
        from planning.inputs import build_refinement_input
        helper=ledger_fixture.LedgerFixture();helper.setUp()
        planner=object.__new__(V2VGoTPlanner)
        planner.tokenizer=AutoTokenizer.from_pretrained(str(CHECKPOINT),use_fast=False,local_files_only=True)
        planner.context_limit=12000
        prepared=build_task_plan_input(planner.tokenizer,dict(speed_mps=2.,yaw_rate_rps=0.),
            helper.ledger,270,fixture.limits()['receiver_spec'],extra_prompt_reserve=256)
        raw='The suggested trajectory is: [(1,0),(2,0),(3,0),(4,0),(5,0),(6,0)].'
        parent=dict(plan_id='plan_0',waypoints=[[float(i),0.] for i in range(1,7)],raw=raw)
        calls=[]
        def generate(features,prompt,limit):
            calls.append((prompt,limit))
            return raw,dict(seconds=0.,input_tokens=1,output_tokens=1,feature_tokens=270)
        planner._generate=generate
        features=dict(active_agent_mask=np.array([[[True],[False]]]))
        wrapped=build_refinement_input(prepared,parent,tokenizer=planner.tokenizer,slot_tokens=256)
        result=planner.plan_prepared(features,wrapped)
        self.assertEqual(result['status'],'parsed')
        self.assertEqual(calls,[(wrapped['q9_prompt'],256)])
        self.assertEqual(wrapped['admission_report'],prepared['admission_report'])
        for mutate in ('origin','budget','prompt','parent'):
            bad=copy.deepcopy(wrapped)
            if mutate=='origin':bad['previous_plan_origin']='ground_truth'
            elif mutate=='budget':bad['refinement']['slot_tokens']=1
            elif mutate=='prompt':bad['q9_prompt']+='hidden additional evidence'
            else:bad['refinement']['previous_plan'][0][0]=999.
            with self.assertRaises(ValueError):planner.plan_prepared(features,bad)
        self.assertEqual(len(calls),1)
