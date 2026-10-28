"""Reproduce from cached licensed source files, leaving published artifacts unchanged."""
from pathlib import Path
import json
import math
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
def compare(a,b,key='root'):
    if isinstance(a,dict):
        assert a.keys()==b.keys(),key
        for k in a:
            if k not in ['frozen_utc','evaluated_utc']:compare(a[k],b[k],key+'.'+k)
    elif isinstance(a,list):
        assert len(a)==len(b),key
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,key+f'[{i}]')
    elif isinstance(a,float):assert math.isclose(a,b,rel_tol=1e-7,abs_tol=1e-9),(key,a,b)
    else:assert a==b,(key,a,b)

with tempfile.TemporaryDirectory(prefix='sensor-reproduction-') as folder:
    replay=Path(folder)
    shutil.copytree(ROOT/'src',replay/'src');shutil.copytree(ROOT/'data/source',replay/'data/source');shutil.copy2(ROOT/'data/manifest.json',replay/'data/manifest.json')
    for script,args in [('analyze.py',['validate']),('analyze.py',['test']),('verify.py',['test']),('figures.py',[])]:
        subprocess.run([sys.executable,str(replay/'src'/script),*args],cwd=replay,check=True,timeout=600)
    for name in ['selection.json','test_summary.json']:
        compare(json.loads((ROOT/'results'/name).read_text()),json.loads((replay/'results'/name).read_text()),name)
    print('PASS: all candidate refits, selected policy, test results, independent metrics and figure generation reproduce offline.')
