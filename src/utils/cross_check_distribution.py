import os, glob, json, unicodedata, re
import pandas as pd
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(ROOT, 'data', 'processed_csv')
MAP_PATH = os.path.join(os.path.dirname(__file__), 'classes_map.json')

def norm(s):
    if s is None: return ''
    x = str(s)
    x = unicodedata.normalize('NFKC', x)
    x = x.replace('\ufeff','').replace('\u00A0',' ').replace('\uFFFD',' ')
    return re.sub(r'\s+',' ', x).strip()

with open(MAP_PATH,'r') as f:
    raw_map = json.load(f)
norm_map = { norm(k): v for k,v in raw_map.items() }

total = 0
agg = Counter()
per_file = {}

for fp in sorted(glob.glob(os.path.join(DATA_DIR,'*.csv'))):
    try:
        cols = pd.read_csv(fp, nrows=0).columns.tolist()
        label_col = next((c for c in cols if c and 'label' in c.strip().lower()), None)
        if not label_col:
            print('No label col in', fp); continue
        s = pd.read_csv(fp, usecols=[label_col], dtype=str)[label_col].fillna('')
    except Exception as e:
        print('ERR', fp, e); continue

    s_norm = s.map(norm)
    mapped = s_norm.map(norm_map)
    vc = mapped.dropna().astype(int).value_counts().to_dict()
    per_file[os.path.basename(fp)] = vc
    for k,v in vc.items():
        agg[int(k)] += int(v)
    total += len(s)

print('Total rows:', total)
for k in sorted(agg.keys()):
    c = agg[k]; pct = c/total*100 if total else 0
    print(f'Class {k}: {c} rows ({pct:.4f}%)')
print('Imbalance ratio (max/min):', (max(agg.values())/min(v for v in agg.values() if v>0)) if total and min(v for v in agg.values() if v>0)>0 else 'inf')
print('\\nPer-file breakdown (sample):')
for fn, vc in per_file.items():
    print(fn, vc)