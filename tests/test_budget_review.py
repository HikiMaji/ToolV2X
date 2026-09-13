"""9-13-2 review regressions; real executor/services with synthetic external models."""
import copy
import json
import unittest
from unittest.mock import patch
import numpy as np

import test_method_episode as fixture
import test_evidence_ledger as ledger_fixture
from planning.method_episode import run_task_episode
from planning.method_controls import control_spec,make_control_policy,episode_bundle
from tools.task_spec import ExecutionSpec
from tools.vehicle import VehicleTools
from probe.kinematic_tools import encode
from evaluation.framework import _method_cost


def bundle_config(**overrides):
    value=dict(continuation_policy_id='return_then_stop',candidate_sources=['initial','slower','constant_motion'],
        slowdown_scale=.5,max_candidates=3,wrapper_reserve_bytes=2048,extra_generation=None)
    value.update(overrides)
    return value


class BudgetReviewTests(unittest.TestCase):
    def setUp(self):
        ledger_fixture.LedgerFixture.setUp(self)

    def execute(self,policy=None,limits=None,control=None,progress=None,local=None,continuation=None):
        local=fixture.window('ego') if local is None else local
        self.service=VehicleTools(lambda:fixture.window(),self.predictor,'scene',10,'peer',task_provenance=fixture.provenance())
        self.service.register_bundle_policy('return_then_stop',continuation or (lambda s:dict(tool='STOP',mode=None,candidate_id=None,reason='actual_return')))
        self.driver=fixture.EvidenceDriver([])
        config=fixture.limits() if limits is None else limits
        policy=fixture.feedback if policy is None else policy
        return run_task_episode(local,self.predictor(local),dict(speed_mps=2.,yaw_rate_rps=0.),
            dict(active_agent_mask=np.array([[[True],[False]]])),self.service,self.predictor,self.driver,
            make_control_policy(control,policy) if control else policy,config,
            local_provenance=fixture.provenance(),sample_id='sample',token_counter=lambda s:len(s)//4,
            control_spec=control,on_progress=progress)

    def test_control_compute_excludes_seven_seconds_of_decision_persistence(self):
        totals=[]
        for delay in (0.,7.):
            clock=[0.]
            def policy(s):
                clock[0]+=2.
                return dict(tool='STOP',mode=None,reason='measured_policy')
            def save(ep):
                if ep['events'][-1]['kind']=='decision':clock[0]+=delay
            with patch('planning.method_episode.perf_counter',side_effect=lambda:clock[0]):
                ep=self.execute(policy=policy,control=control_spec('feedback'),progress=save)
            row=_method_cost(dict(episode=ep,inputs=dict(local_prediction_seconds=0.)))
            totals.append(row['control_seconds'])
            self.assertEqual(ep['events'][-1]['kind'],'STOP')
        self.assertEqual(totals,[2.,2.])

    def test_payable_current_remains_available_when_change_exceeds_request_cap(self):
        baseline=self.execute()
        current=copy.deepcopy(baseline['requests'][1]);current.update(mode='current',tau_old=None)
        config=fixture.limits();config['execution_spec']['max_request_bytes']=len(encode(current))
        seen=[]
        def policy(state):
            seen.append(copy.deepcopy(state))
            if len(seen)==1:return dict(tool='P',mode='current',reason='first')
            mode='change' if dict(tool='F',mode='change') in state['available_actions'] else 'current'
            return dict(tool='F',mode=mode,reason='best_payable')
        ep=self.execute(policy=policy,limits=config)
        self.assertEqual(len(ep['requests']),2)
        self.assertEqual(ep['requests'][1]['mode'],'current')
        self.assertNotIn(dict(tool='F',mode='change'),seen[1]['available_actions'])
        self.assertIn(dict(tool='F',mode='current'),seen[1]['available_actions'])
        rejected=next(r for r in ep['decisions'][1]['infeasible_actions'] if r['action']==dict(tool='F',mode='change'))
        self.assertEqual(rejected['reason'],'request_byte_limit')
        self.assertEqual(ep['decisions'][1]['action'],ep['decisions'][1]['executed_action'])

    def test_default_one_shot_sends_a_budgeted_nonempty_summary(self):
        config=fixture.limits();config['execution_spec']=ExecutionSpec().to_dict()
        ep=self.execute(limits=config,control=control_spec('one_shot',bundle=bundle_config()))
        self.assertEqual(ep['status'],'completed',ep.get('error'))
        self.assertEqual(len(ep['requests']),1)
        request=ep['requests'][0]
        self.assertLessEqual(len(encode(request)),4096)
        summary=request['public_summary']
        self.assertTrue(summary['local_evidence'])
        self.assertLess(len(encode(summary['local_evidence'])),len(encode(ep['decisions'][0]['state']['local_evidence'])))
        self.assertTrue(ep['decisions'][0]['request_preflight'])

    def test_summary_selection_is_deterministic_and_keeps_local_fields_when_room_exists(self):
        baseline=self.execute(policy=lambda s:dict(tool='STOP',mode=None,reason='inspect'))
        state=copy.deepcopy(baseline['decisions'][0]['state'])
        state['execution_spec']=ExecutionSpec().to_dict();state['remaining_budget']['bytes']=24576
        c=control_spec('one_shot',bundle=bundle_config())
        a=episode_bundle(state,c,'P')
        state['local_evidence'].reverse()
        b=episode_bundle(state,c,'P')
        self.assertEqual(a,b)
        self.assertLessEqual(len(encode(a)),4096)
        self.assertTrue(a['public_summary']['local_evidence'])
        self.assertIn('summary_status',a['public_summary'])
        self.assertIn('summary_spec',c['bundle'])

    def test_per_rpc_and_episode_aggregate_budgets_are_explicit_different_conditions(self):
        baseline=self.execute(policy=lambda s:dict(tool='STOP',mode=None,reason='inspect'))
        state=copy.deepcopy(baseline['decisions'][0]['state'])
        state['execution_spec']=ExecutionSpec().to_dict();state['remaining_budget']['bytes']=24576
        profiles=[]
        for mode in ('per_rpc','episode_aggregate'):
            c=control_spec('one_shot',bundle=bundle_config(response_budget=dict(version='toolv2x_bundle_budget_v1',
                mode=mode,outer_response_cap=None,primitive_response_caps=None)))
            request=episode_bundle(state,c,'P')
            profiles.append(request)
        one,two=profiles
        self.assertEqual(one['limits']['outer_response_cap'],8192)
        self.assertEqual(one['limits']['primitive_response_caps'],[3072,3072])
        self.assertEqual(two['limits']['outer_response_cap'],18432)
        self.assertEqual(two['limits']['primitive_response_caps'],[8192,8192])
        self.assertEqual(two['limits']['execution_spec'],one['limits']['execution_spec'])
        self.assertLessEqual(len(encode(two))+18432,24576)
        from tools.control_bundle import validate_bundle_envelope
        validate_bundle_envelope(one);validate_bundle_envelope(two)

    def test_legacy_control_time_is_kept_as_diagnostic_but_not_used_as_compute(self):
        ep=self.execute(control=control_spec('feedback'))
        for event in ep['cost_events']:
            if event['kind']=='control':
                event.pop('timing_version',None)
                event['seconds']=7.
        cost=_method_cost(dict(episode=ep,inputs=dict(local_prediction_seconds=0.)))
        self.assertIsNone(cost['control_seconds'])
        self.assertIsNone(cost['total_compute_seconds'])
        self.assertFalse(cost['cost_complete'])
        self.assertEqual(cost['reported_control_seconds'],21.)

    def test_aggregate_response_exceeds_one_packet_cap_but_keeps_equal_total_budget(self):
        config=fixture.limits();config['execution_spec']=ExecutionSpec().to_dict()
        pred=self.predictor._predict
        def precise(window):
            out=pred(window)
            out['means']=out['means']+np.arange(out['means'].size).reshape(out['means'].shape)/10007.
            return out
        self.predictor._predict=precise
        bundle=bundle_config(response_budget=dict(version='toolv2x_bundle_budget_v1',mode='episode_aggregate',
            outer_response_cap=None,primitive_response_caps=None))
        ep=self.execute(limits=config,control=control_spec('one_shot',bundle=bundle),
            continuation=lambda s:dict(tool='F',mode='current',candidate_id='initial',reason='paid_second'))
        self.assertEqual(ep['status'],'completed',ep.get('error'))
        cost=_method_cost(dict(episode=ep,inputs=dict(local_prediction_seconds=0.)))
        self.assertEqual((cost['rpc_rounds'],cost['calls']),(1,2))
        self.assertGreater(cost['response_bytes'],8192)
        self.assertLessEqual(cost['response_bytes'],18432)
        self.assertLessEqual(cost['request_bytes']+cost['response_bytes'],24576)
        self.assertTrue(all(r['cost']['response_bytes']<=8192 for r in self.service.task_records))
        self.assertEqual(cost['response_bytes'],len(bytes.fromhex(ep['responses'][0]['wire_hex'])))
        self.assertEqual(cost['request_bytes'],len(encode(ep['requests'][0])))

    def test_large_local_context_does_not_force_ego_and_tiny_budget_records_forced_stop(self):
        local=fixture.window('ego')
        for key in ('states','scores','valid'):
            local[key]=np.repeat(local[key][:1],20,axis=0)
        local['track_ids']=np.arange(100,120,dtype=local['track_ids'].dtype)
        config=fixture.limits();config['execution_spec']=ExecutionSpec().to_dict()
        ep=self.execute(local=local,limits=config,control=control_spec('one_shot',bundle=bundle_config()))
        self.assertEqual(ep['status'],'completed',ep.get('error'))
        self.assertEqual(len(ep['requests']),1)
        self.assertLessEqual(ep['cost']['request_bytes'],4096)
        self.assertTrue(ep['requests'][0]['public_summary']['summary_status']['truncated'])
        config['execution_spec']['max_request_bytes']=1
        ep=self.execute(limits=config,control=control_spec('feedback'),policy=lambda s:self.fail('no payable action must not reach the policy'))
        self.assertEqual(ep['requests'],[])
        self.assertEqual(self.service.task_records,[])
        d=ep['decisions'][0]
        self.assertFalse(d['policy_called'])
        self.assertEqual(d['forced_reason'],'request_byte_limit')
        self.assertEqual(d['executed_action']['tool'],'STOP')
        self.assertEqual(len(d['infeasible_actions']),2)

    def test_summary_byte_and_field_limits_are_explicit(self):
        baseline=self.execute(policy=lambda s:dict(tool='STOP',mode=None,reason='inspect'))
        state=baseline['decisions'][0]['state']
        c=control_spec('one_shot',bundle=bundle_config(summary_spec=dict(version='toolv2x_bundle_summary_v1',
            selection='distance_then_field',field_kinds=['anchor'],max_fields=1,max_bytes=600)))
        req=episode_bundle(state,c,'P')
        self.assertEqual(len(req['public_summary']['local_evidence']),1)
        self.assertLessEqual(len(encode(req['public_summary'])),600)
        c['bundle']['summary_spec']['max_bytes']=1
        with self.assertRaises(ValueError):episode_bundle(state,c,'P')

    def test_public_aggregate_cap_does_not_raise_total_budget_or_allow_oversize_wire(self):
        from tools.control_bundle import validate_bundle_envelope
        state=self.execute(policy=lambda s:dict(tool='STOP',mode=None,reason='inspect'))['decisions'][0]['state']
        state['execution_spec']=ExecutionSpec().to_dict();state['remaining_budget']['bytes']=24576
        c=control_spec('one_shot',bundle=bundle_config(response_budget=dict(version='toolv2x_bundle_budget_v1',
            mode='episode_aggregate',outer_response_cap=None,primitive_response_caps=None)))
        req=episode_bundle(state,c,'P')
        bad=copy.deepcopy(req);bad['limits']['outer_response_cap']=24577
        with self.assertRaises(ValueError):validate_bundle_envelope(bad)
        config=fixture.limits();config['execution_spec']=ExecutionSpec().to_dict()
        ep=self.execute(limits=config,control=c,continuation=lambda s:dict(tool='STOP',mode=None,candidate_id=None,reason='x'*25000))
        self.assertEqual(ep['status'],'service_error')
        self.assertEqual(ep['cost']['response_bytes'],0)
        self.assertGreater(ep['cost']['request_bytes'],0)
        self.assertEqual(len(self.service.task_records),1)
        self.assertFalse(ep['cost']['complete'])

    def test_cli_archives_both_budget_modes_with_distinct_branches(self):
        from pathlib import Path
        import tempfile
        from planning import run_framework as runner
        from planning.context import build_task_plan_input
        from evaluation.framework import evaluate_method
        helper=fixture.InteractionRunnerTests();helper.setUp()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data,rows,spec=helper.make_data(root)
            labels=root/'labels';labels.mkdir();(labels/'validation.jsonl').write_text('')
            for mode in ('per_rpc','episode_aggregate'):
                raw=json.loads(spec.read_text());raw['limits']['execution_spec']=ExecutionSpec().to_dict()
                raw['control']=control_spec('one_shot',bundle=bundle_config(continuation_policy_id='diagnostic_conditional_v1',
                    response_budget=dict(version='toolv2x_bundle_budget_v1',mode=mode,outer_response_cap=None,primitive_response_caps=None)))
                spec.write_text(json.dumps(raw))
                def build(*args,**kw):
                    kw['token_counter']=lambda s:len(s)//4
                    return build_task_plan_input(*args,**kw)
                with patch.object(runner,'_load_interaction_runtime',return_value=helper.runtime(rows)), \
                     patch('planning.method_episode.build_task_plan_input',side_effect=build):
                    runner.interact(data,root/mode,Path('/fixture/driver'),spec,per_recording=0)
                evaluate_method(root/mode,root/(mode+'_eval'),labels)
                results=runner.read_jsonl(root/(mode+'_eval')/'rows.jsonl')
                self.assertTrue(all(r['task_success'] and r['rpc_rounds']==1 and r['calls']==2 for r in results),results)
                self.assertEqual({r['branch_id'] for r in results},{'one_shot:'+mode})
