"""Reconcile the four real update/reload checks and archived input boundaries."""
import json
import math
from pathlib import Path
import statistics

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
records = []
examples = {}
sample_id = 'testoutput_CAV_data_2022-03-15-10-09-50_0:10'
index = [json.loads(line) for line in (ROOT / 'outputs/adaptation_data_v1/online_index/train.jsonl').read_text().splitlines()]
indexed = [row for row in index if row['sample_id'] == sample_id]
assert len(indexed) == 1 and indexed[0]['role'] == 'train' and indexed[0]['g'] == 719
source_paths = ('src/planning/check_training.py', 'src/planning/adaptation_data.py',
                'src/planning/inputs.py', 'src/planning/v2vgot.py', 'tests/verify_driving_update.py')
for initialization in ('got', 'llm'):
    for decoding in ('direct', 'q8_q9'):
        name = '%s_%s_flash' % (initialization, decoding)
        run = OUT / name
        training = json.loads((run / 'training_check.json').read_text())
        reload = json.loads((run / 'reload_check.json').read_text())
        updates = json.loads((run / 'parameter_updates.json').read_text())
        rows = [json.loads(line) for line in (run / 'examples.jsonl').read_text().splitlines()]
        examples[(initialization, decoding)] = rows
        assert training['status'] == 'updated_and_saved' and training['optimizer_steps'] == 1
        assert not training['frozen_gradients'] and not training['formal_adaptation_complete']
        assert training['saved_adapter_matches_live'] and training['saved_projector_matches_live']
        assert training['model']['trainable_lora_loaded']
        assert training['model']['attention_implementation'] == 'flash_attention_2'
        assert len(rows) == len(training['rows']) == (4 if decoding == 'direct' else 8)
        assert {r['sample_id'] for r in rows} == {sample_id}
        for action in ('Ego', 'P', 'F', 'PF'):
            assert {r['task'] for r in rows if r['action'] == action} == (
                {'Trajectory'} if decoding == 'direct' else {'Q8', 'Q9'})
        for row in training['rows']:
            assert math.isfinite(row['loss']) and row['feature_labels_masked']
            assert row['feature_tokens'] == 540 and row['sequence_tokens'] <= 4096
        gradients = updates['gradient_norms']
        differences = updates['maximum_absolute_changes']
        assert gradients.keys() == differences.keys() and len(gradients) == 452
        assert all(value is not None and math.isfinite(value) for value in gradients.values())
        assert sum(value > 0 for value in differences.values()) == training['changed_tensors'] == 452
        for parameter_group in ('.lora_A.', '.lora_B.', '.mm_projector.'):
            assert any(parameter_group in key and value > 0 for key, value in differences.items())
        assert reload['status'] == 'PASS' and reload['actual_generation']
        assert not reload['model']['trainable_lora_loaded']
        assert reload['model']['attention_implementation'] == 'sdpa'
        assert reload['generated_sample_id'] == rows[-1]['sample_id']
        assert reload['generated_action'] == 'PF' and reload['task'] == rows[-1]['task']
        assert len(reload['projector_exact_checks']) == 8 and all(reload['projector_exact_checks'].values())
        source_checks = {p: (run / 'code_snapshot' / p).read_bytes() == (ROOT / p).read_bytes()
                         for p in source_paths}
        source_checks['vendor_builder'] = (run / 'code_snapshot/v2vgot_original/llava/model/builder.py').read_bytes() == (
            ROOT / 'vendor/v2vgot_llava/llava/model/builder.py').read_bytes()
        assert all(source_checks.values())
        records.append(dict(name=name, initialization=initialization, decoding=decoding,
                            supervised_rows=len(rows), optimizer_steps=1,
                            mean_row_loss=statistics.mean(r['loss'] for r in training['rows']),
                            sequence_tokens=[min(r['sequence_tokens'] for r in training['rows']),
                                             max(r['sequence_tokens'] for r in training['rows'])],
                            changed_tensors=training['changed_tensors'],
                            peak_allocated_GiB=training['peak_allocated_bytes'] / 2 ** 30,
                            sampled_merge_values=sum(c['compared_values'] for c in reload['sampled_lora_merge_checks']),
                            sampled_merge_max_error=max(c['maximum_absolute_error'] for c in reload['sampled_lora_merge_checks']),
                            projector_exact_tensors=8, actual_reload_generation=True,
                            source_files_match=source_checks))
for decoding in ('direct', 'q8_q9'):
    assert examples[('got', decoding)] == examples[('llm', decoding)]
for initialization in ('got', 'llm'):
    direct = {r['action']: r for r in examples[(initialization, 'direct')]}
    chain = {r['action']: r for r in examples[(initialization, 'q8_q9')] if r['task'] == 'Q9'}
    for action in direct:
        assert direct[action]['target'] == chain[action]['target']
        assert direct[action]['feature_path'] == chain[action]['feature_path']
result = dict(status='PASS', scope='one training frame, one update per combination; infrastructure only',
              runs=records, total_optimizer_updates=4, unique_training_frames=1,
              same_rows_across_initializations=True, same_trajectory_targets_across_decodings=True,
              q8_parent_source='saved original GoT generation; not each candidate own generated parent',
              mtr_evidence='historical readiness frame; not newly frozen MTR',
              test_access=False, actual_rgb_input=False, formal_adaptation_complete=False,
              initialization_or_decoder_selected=False, driving_utility_evaluated=False)
(OUT / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
