# code/eval/metrics_text.py
# Compute per-model, per-dataset metrics from run_dataset_tx.py CSVs,
# using ground truth loaded from saved_to_disk datasets.
# Outputs one summary CSV per dataset and a combined table.

import os, re, argparse, glob
from pathlib import Path
import pandas as pd
from datasets import load_from_disk

# ---- Normalizers ----
_ws = re.compile(r"\s+")
def norm_text(s):
    if s is None: return ""
    return _ws.sub(" ", str(s).strip()).lower()

def extract_number(ans: str):
    if ans is None: return None
    s = str(ans)
    m = re.findall(r"[-+]?[0-9]*\.?[0-9]+", s)
    return m[-1] if m else None

# ---- Dataset-specific gold readers ----
def gold_ag_news(ds_dir, split):
    ds = load_from_disk(ds_dir)[split]
    return {i: ex["label"] for i, ex in enumerate(ds)}

def gold_arc_easy(ds_dir, split):
    ds = load_from_disk(ds_dir)[split]
    return {i: ex.get("answerKey") for i, ex in enumerate(ds)}

def gold_hellaswag(ds_dir, split):
    ds = load_from_disk(ds_dir)[split]
    return {i: int(ex.get("label", -1)) for i, ex in enumerate(ds)}

def gold_gsm8k(ds_dir, split):
    ds = load_from_disk(ds_dir)[split]
    return {i: extract_number(ex.get("answer","")) for i, ex in enumerate(ds)}

# ---- Prediction parsers ----
def pred_ag_news(text):
    s = norm_text(text)
    m = re.search(r"\b([0-3])\b", s)
    if m: return int(m.group(1))
    if any(k in s for k in ["world","international"]): return 0
    if any(k in s for k in ["sports","game"]): return 1
    if any(k in s for k in ["business","market"]): return 2
    if any(k in s for k in ["sci","technology","tech"]): return 3
    return None

def pred_arc_easy(text):
    s = norm_text(text)
    m = re.search(r"\b([abcd])\b", s)
    if m: return m.group(1).upper()
    m = re.search(r"\banswer\s*[:=]?\s*([abcd])\b", s)
    if m: return m.group(1).upper()
    return None

def pred_hellaswag(text):
    s = norm_text(text)
    m = re.search(r"\b([0-3])\b", s)
    if m: return int(m.group(1))
    m = re.search(r"\b([abcd])\b", s)
    if m: return ord(m.group(1)) - ord('a')
    return None

def pred_gsm8k(text):
    return extract_number(text)

# ---- Metric compute ----
def accuracy_from_pairs(pairs):
    n = 0; c = 0
    for g,p in pairs:
        if g is None or p is None: continue
        n += 1
        if g == p: c += 1
    return (c, n, (c/n if n else 0.0))

# ---- Main ----
PARSERS = {
    'ag_news':   (gold_ag_news,   pred_ag_news),
    'arc_easy':  (gold_arc_easy,  pred_arc_easy),
    'hellaswag': (gold_hellaswag, pred_hellaswag),
    'gsm8k':     (gold_gsm8k,     pred_gsm8k),
}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--outputs_glob', default='outputs/llm/*.csv')
    ap.add_argument('--data_root',    default='data/text_offline')
    ap.add_argument('--split',        default='test')
    ap.add_argument('--out_csv',      default='outputs/metrics_text_summary.csv')
    args = ap.parse_args()

    rows = []
    for csv_path in glob.glob(args.outputs_glob):
        df = pd.read_csv(csv_path)
        if 'dataset' not in df.columns or 'output' not in df.columns:
            continue
        dataset = df['dataset'].iloc[0]
        if dataset not in PARSERS: 
            continue

        gold_fn, pred_fn = PARSERS[dataset]
        ds_dir = str(Path(args.data_root)/dataset)
        gold_map = gold_fn(ds_dir, args.split)

        # Prefer `model` column; if absent, infer from filename
        if 'model' in df.columns:
            model_groups = df.groupby('model')
        else:
            df['model'] = Path(csv_path).stem.split('__')[-1]
            model_groups = df.groupby('model')

        for model, g in model_groups:
            preds = {}
            for _, r in g.iterrows():
                i = int(r['i']) if 'i' in r else None
                preds[i] = pred_fn(str(r['output']))
            pairs = [(gold_map.get(i), preds.get(i)) for i in gold_map.keys()]
            correct, total, acc = accuracy_from_pairs(pairs)
            rows.append({
                'dataset': dataset,
                'model': model,
                'split': args.split,
                'count': int(total),
                'correct': int(correct),
                'metric': round(acc, 4)
            })

    out = pd.DataFrame(rows).sort_values(['dataset','metric'], ascending=[True, False])
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(out.to_string(index=False))