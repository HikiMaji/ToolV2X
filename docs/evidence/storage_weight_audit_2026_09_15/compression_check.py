"""Memory-only gzip roundtrip on existing real task archives; no archive rewrite."""
from pathlib import Path
import gzip,json,time
base=Path(__file__).resolve().parent
source=base.parents[1]/'outputs/structured_driver_fix_2026_09_15_v1/acceptance/tasks'
rows=[];started=time.perf_counter()
for p in sorted(source.glob('*.json')):
    raw=p.read_bytes();packed=gzip.compress(raw,compresslevel=1,mtime=0)
    assert gzip.decompress(packed)==raw
    rows.append(dict(file=p.name,original_bytes=len(raw),gzip_bytes=len(packed)))
assert len(rows)==80
raw=sum(r['original_bytes'] for r in rows);compressed=sum(r['gzip_bytes'] for r in rows)
result=dict(version='toolv2x_archive_compression_probe_v1',tasks=80,frames=20,
    original_bytes=raw,gzip_level1_bytes=compressed,ratio=compressed/raw,byte_roundtrip_pass=True,
    linear_full_3095_frames_gzip_bytes=compressed*3095/20,seconds=time.perf_counter()-started,
    disk_archives_modified=False,training_reader_support_implemented=False,
    scope='Measured existing 80 full JSON tasks only; full-size total is extrapolated and excludes all other artifacts.',files=rows)
(base/'compression_summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='files'}),flush=True)
