"""
生成 CMP MTR dataset 需要的 V2V4Real 目录骨架（只有空 yaml 文件名，无点云）。

MTR 的 V2V4RealMultiEgoDataset 只用 <root>/<scene>/<cav>/<timestamp>.yaml 的**文件名**来枚举
场景、CAV 与时间戳（`extract_timestamps`），yaml 内容仅在 CONVEX_HULL_THRESHOLD != -1 时才读取。
因此可以用空文件代替原始数据，避免下载 ~40GB 点云。

用法: python make_v2v4real_stub.py
输出: outputs/v2v4real_stub/{train,test}/<scene>/{0,1}/000000.yaml ...
"""
import os
import sys
import glob

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from common import v2v4real_meta as M  # noqa: E402

CMP_GT_DIR = '/root/autodl-tmp/CMP/preprocessed_data/v2v4real/gt_multiego_speedless'
OUT = os.path.join(HERE, '..', '..', 'outputs', 'v2v4real_stub')

for split in ['train', 'test']:
    files = sorted(glob.glob(os.path.join(CMP_GT_DIR, split, '*-0-traj.pickle')))
    names = [os.path.basename(f)[:-len('-0-traj.pickle')] for f in files]
    ranges = M.seq_ranges(split)
    assert len(names) == len(ranges)
    n = 0
    for name, (_, s, e) in zip(names, ranges):
        for cav in ['0', '1']:
            d = os.path.join(OUT, split, name, cav)
            os.makedirs(d, exist_ok=True)
            for t in range(e - s):
                p = os.path.join(d, '%06d.yaml' % t)
                if not os.path.exists(p):
                    open(p, 'w').close()
                n += 1
    print(split, len(names), 'scenes,', n, 'stub yaml files')
print('root:', os.path.abspath(OUT))
