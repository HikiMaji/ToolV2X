"""Checkpoint continuation reuses data and falls back when the speed gate fails."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('baseline_runner', ROOT/'scripts/run_framework_baseline.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ContinuationTests(unittest.TestCase):
    def test_repeated_interruption_follows_saved_resume_source_and_rejects_cycle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            old,new=root/'old',root/'new'
            saved=old/'training/recovery-step000025'; saved.mkdir(parents=True)
            (saved/'training_state.json').write_text(json.dumps(dict(epoch=0,steps=25)))
            for name in ('training_state.pt','adapter_model.safetensors','non_lora_trainables.bin'):
                (saved/name).write_bytes(b'fixture')
            new.mkdir()
            (new/'config.json').write_text(json.dumps(dict(resume_training=str(old))))
            self.assertEqual(runner.latest_checkpoint(new/'training'),saved)
            (new/'config.json').write_text(json.dumps(dict(resume_training=str(new))))
            with self.assertRaisesRegex(ValueError,'cycle'):
                runner.latest_checkpoint(new/'training')

    def test_completed_training_checkpoint_continues_generation_without_retraining(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for name in ('src','scripts','tests','vendor'):
                (root/name).mkdir()
            source,out,data=root/'old',root/'new',root/'data'
            checkpoint=source/'training/checkpoint-epoch03'; checkpoint.mkdir(parents=True)
            (source/'episodes').mkdir(); (source/'training_data').mkdir()
            state=dict(epoch=3,steps=100,examples_seen=800,best_checkpoint=str(checkpoint),best_loss=.2,bad_epochs=0)
            (checkpoint/'training_state.json').write_text(json.dumps(state))
            for name in ('training_state.pt','adapter_model.safetensors','non_lora_trainables.bin'):
                (checkpoint/name).write_bytes(b'fixture')
            (source/'config.json').write_text(json.dumps(dict(data=str(data),per_recording=0,epochs=3)))
            calls=[]
            class Child:
                pid=999999
                def __init__(self,command,**kwargs):
                    calls.append(command)
                    if Path(command[1]).name == 'train_driver.py':
                        raise AssertionError('completed training must not enter train/benchmark again')
                def wait(self): return 0
            with patch.object(runner,'ROOT',root), patch.object(runner.subprocess,'Popen',Child):
                runner.run(out,data,resume_training=source)
            self.assertEqual(len(calls),2)
            self.assertEqual(Path(calls[0][1]).name,'run_framework.py')
            self.assertEqual(json.loads((out/'training/result.json').read_text())['checkpoint'],str(checkpoint))

    def test_first_saved_update_remains_recoverable_before_next_periodic_save(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root/'first_step/recovery-step000001'
            first.mkdir(parents=True)
            (root/'training').mkdir()
            (first/'training_state.json').write_text(json.dumps(dict(epoch=0, steps=1)))
            for name in ('training_state.pt','adapter_model.safetensors','non_lora_trainables.bin'):
                (first/name).write_bytes(b'fixture')
            self.assertEqual(runner.latest_checkpoint(root/'training'), first)

    def test_latest_complete_recovery_overrides_stale_selection_and_ignores_staging(self):
        self.assertTrue(hasattr(runner, 'latest_checkpoint'), 'periodic checkpoint selection missing')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, out, data = root/'old', root/'new', root/'data'
            for epoch in (1, 2):
                directory = source/'training'/('checkpoint-epoch%02d' % epoch)
                directory.mkdir(parents=True)
                (directory/'training_state.json').write_text(json.dumps(dict(epoch=epoch, steps=20*epoch)))
            (source/'config.json').write_text(json.dumps(dict(data=str(data), per_recording=0, epochs=3)))
            (source/'training/selection.json').write_text(json.dumps(dict(epoch=1, steps=20)))
            recovery = source/'training/recovery-step000041'
            recovery.mkdir()
            (recovery/'training_state.json').write_text(json.dumps(dict(epoch=2, steps=41)))
            for directory in (source/'training').iterdir():
                if directory.is_dir():
                    for name in ('training_state.pt','adapter_model.safetensors','non_lora_trainables.bin'):
                        (directory/name).write_bytes(b'fixture')
            staging = source/'training/.recovery-step000050.incomplete'
            staging.mkdir()
            (staging/'training_state.json').write_text(json.dumps(dict(epoch=2, steps=50)))
            self.assertEqual(runner.latest_checkpoint(source/'training'), recovery)

    def check_mode(self, benchmark_code, enabled):
        import inspect
        self.assertIn('resume_training', inspect.signature(runner.run).parameters,
                      'checkpoint continuation is not implemented')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('src', 'scripts', 'tests', 'vendor'):
                (root/name).mkdir()
            source, out, data = root/'old', root/'new', root/'data'
            (source/'training/checkpoint-epoch01').mkdir(parents=True)
            (source/'episodes').mkdir()
            (source/'training_data').mkdir()
            (source/'training/selection.json').write_text(json.dumps(dict(epoch=1, steps=20)))
            (source/'training/checkpoint-epoch01/training_state.json').write_text(json.dumps(dict(epoch=1, steps=20)))
            for name in ('training_state.pt','adapter_model.safetensors','non_lora_trainables.bin'):
                (source/'training/checkpoint-epoch01'/name).write_bytes(b'fixture')
            (source/'config.json').write_text(json.dumps(dict(data=str(data), per_recording=0, epochs=3)))
            calls = []

            class Child:
                pid = 999999
                def __init__(self, command, **kwargs):
                    calls.append(command)
                    script = Path(command[1]).name
                    if '--benchmark-only' in command:
                        (out/'benchmark').mkdir()
                        (out/'benchmark/performance.json').write_text(json.dumps(dict(enable_answer_head=enabled)))
                        self.code = benchmark_code
                    elif script == 'train_driver.py':
                        (out/'training').mkdir()
                        (out/'training/result.json').write_text(json.dumps(dict(status='training_completed',
                            formal_adaptation_complete=True, checkpoint=str(source/'training/checkpoint-epoch01'))))
                        self.code = 0
                    else:
                        self.code = 0
                def wait(self):
                    return self.code

            with patch.object(runner, 'ROOT', root), patch.object(runner.subprocess, 'Popen', Child):
                runner.run(out, data, resume_training=source)
            self.assertEqual(len(calls), 4)  # benchmark, continued train, generate, evaluate
            self.assertIn('--benchmark-only', calls[0])
            self.assertIn(str(source/'training_data'), calls[1])
            self.assertIn('--resume', calls[1])
            self.assertEqual('--answer-only-head' in calls[1], enabled and benchmark_code == 0)
            self.assertIn(str(source/'episodes'), calls[2])
            self.assertEqual(json.loads((out/'status.json').read_text())['status'], 'completed')

    def test_passed_gate_enables_optimized_training(self):
        self.check_mode(0, True)

    def test_rejected_gate_keeps_original_path(self):
        self.check_mode(0, False)

    def test_failed_benchmark_keeps_original_path_despite_partial_report(self):
        self.check_mode(1, True)

    def test_reuse_data_can_start_original_weights_and_verify_first_resume(self):
        import inspect
        self.assertIn('reuse_baseline', inspect.signature(runner.run).parameters,
                      'existing data reuse is missing')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('src','scripts','tests','vendor'):
                (root/name).mkdir()
            source, out, data = root/'old', root/'new', root/'data'
            source.mkdir()
            (source/'episodes').mkdir()
            (source/'training_data').mkdir()
            (source/'training_data/manifest.json').write_text('{}')
            (source/'config.json').write_text(json.dumps(dict(data=str(data),per_recording=0,epochs=3)))
            performance = root/'performance.json'
            performance.write_text(json.dumps(dict(enable_answer_head=False)))
            calls=[]
            class Child:
                pid=999999
                def __init__(self, command, **kwargs):
                    calls.append(command)
                    if Path(command[1]).name == 'train_driver.py':
                        directory=Path(command[3]); directory.mkdir()
                        paused='--pause-after-steps' in command
                        (directory/'result.json').write_text(json.dumps(dict(status='training_paused' if paused else 'training_completed',
                            optimizer_steps=1 if paused else 100,formal_adaptation_complete=not paused,
                            checkpoint=str(directory/'recovery-step000001'))))
                def wait(self): return 0
            with patch.object(runner,'ROOT',root), patch.object(runner.subprocess,'Popen',Child):
                runner.run(out,data,reuse_baseline=source,efficiency_report=performance,verify_first_resume=True)
            self.assertEqual(len(calls),4)
            self.assertIn('--pause-after-steps',calls[0])
            self.assertNotIn('--resume',calls[0])
            self.assertIn(str(out/'first_step/recovery-step000001'),calls[1])
            self.assertIn(str(source/'training_data'),calls[0])


if __name__ == '__main__':
    unittest.main()
