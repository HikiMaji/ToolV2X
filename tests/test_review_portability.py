"""The configured data root must control the actual file read after relocation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class ReviewPortabilityTests(unittest.TestCase):
    def test_configured_data_root_supplies_localization(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / 'external_data'
            poses = data / 'train_no_fusion_keep_all/npy/ego'
            poses.mkdir(parents=True)
            pose = np.eye(4)
            pose[0, 3] = 123456.
            np.save(poses / '0000_lidar_pose.npy', pose)
            env = dict(os.environ, TOOLV2X_V2VGOT_ROOT=str(data),
                       PYTHONPATH=str(ROOT / 'src'), PYTHONDONTWRITEBYTECODE='1')
            code = ('import json; from common import v2v4real_meta as M; '
                    'print(json.dumps({"root": M.V2VGOT_ROOT, '
                    '"pose": M.load_pose("train", 0).tolist()}))')
            result = subprocess.run([sys.executable, '-c', code], env=env,
                                    capture_output=True, text=True, check=True)
            actual = json.loads(result.stdout)
            self.assertEqual(actual['root'], str(data))
            np.testing.assert_array_equal(actual['pose'], pose)


if __name__ == '__main__':
    unittest.main()
