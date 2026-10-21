# code/eval/metrics_all.py
import pandas as pd
from pathlib import Path
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--text_csv',   default='outputs/metrics_text_summary.csv')
ap.add_argument('--vision_csv', default='outputs/metrics_vision_summary.csv')
ap.add_argument('--out_csv',    default='outputs/metrics_all.csv')
args = ap.parse_args()

parts = []
for p in [args.text_csv, args.vision_csv]:
    if Path(p).exists():
        parts.append(pd.read_csv(p))

if parts:
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(args.out_csv, index=False)
    print(out.sort_values(['dataset','metric_name','value'], ascending=[True,True,False]).to_string(index=False))
else:
    print('No inputs found.')