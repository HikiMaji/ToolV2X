"""Run the explicit NumPy/SciPy review suite; full integration is a separate command."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
sys.path.insert(0, str(ROOT / 'src'))

names = [
    'test_compact_evidence', 'test_planning_inputs', 'test_protocol',
    'test_vehicle_tools', 'test_vehicle_probe', 'test_learned_probe',
    'test_review_portability', 'test_saved_review_evidence',
    'test_adaptation_data.AdaptationDataTests.test_native_labels_use_six_future_poses_in_fixed_current_frame',
    'test_adaptation_data.AdaptationDataTests.test_missing_or_cross_record_future_keeps_mask_without_fabricating_label',
    'test_adaptation_data.AdaptationDataTests.test_supervised_q9_uses_generated_parent_even_when_q9_prediction_failed',
]

if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromNames(names)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if 'torch' in sys.modules or 'transformers' in sys.modules:
        raise RuntimeError('review suite unexpectedly imported a model dependency')
    sys.exit(0 if result.wasSuccessful() else 1)
