# code/eval/metrics_vision.py
# Evaluate per model from CSVs created by run_vlm_vqa.py and run_vlm_caption.py
# VQAv2: placeholder EM (needs question_id in CSV for strict eval)
# COCO: compute BLEU-1 and BLEU-2 using nltk

import os, json, glob, argparse
from pathlib import Path
import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
nltk.download('punkt', quiet=True)

# --- COCO loaders ---
def load_coco_refs(root, split='val'):
    if split == 'train':
        ann = Path(root)/'annotations'/'captions_train2017.json'
    else:
        ann = Path(root)/'annotations'/'captions_val2017.json'
    data = json.load(open(ann))
    id2refs = {}
    for an in data['annotations']:
        img_id = an['image_id']
        id2refs.setdefault(img_id, []).append(an['caption'])
    fname2id = {img['file_name']: img['id'] for img in data['images']}
    return id2refs, fname2id

SM = SmoothingFunction().method1
def bleu12(hyp, refs):
    h = str(hyp).strip().split()
    r = [str(x).strip().split() for x in refs if x and str(x).strip()]
    if not h or not r:
        return 0.0, 0.0
    b1 = sentence_bleu(r, h, weights=(1.0,0,0,0), smoothing_function=SM)
    b2 = sentence_bleu(r, h, weights=(0.5,0.5,0,0), smoothing_function=SM)
    return b1, b2

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--outputs_glob', default='outputs/vlm/*.csv')
    ap.add_argument('--vqav2_root',   default='data/vision_offline/vqav2')  # reserved
    ap.add_argument('--coco_root',    default='data/vision_offline/coco_captions')
    ap.add_argument('--split',        default='val')
    ap.add_argument('--out_csv',      default='outputs/metrics_vision_summary.csv')
    args = ap.parse_args()

    rows = []
    for csv_path in glob.glob(args.outputs_glob):
        df = pd.read_csv(csv_path)
        if 'dataset' not in df.columns:
            continue
        dataset = df['dataset'].iloc[0]
        model = df['model'].iloc[0] if 'model' in df.columns else Path(csv_path).stem

        if dataset == 'coco_captions':
            id2refs, fname2id = load_coco_refs(args.coco_root, args.split)
            b1_sum = 0.0; b2_sum = 0.0; n = 0
            for _, r in df.iterrows():
                img_path = str(r.get('image',''))
                fname = os.path.basename(img_path)
                img_id = fname2id.get(fname)
                if img_id is None: 
                    continue
                refs = id2refs.get(img_id, [])
                hyp  = r.get('output_snippet', r.get('output',''))
                b1, b2 = bleu12(hyp, refs)
                b1_sum += b1; b2_sum += b2; n += 1
            if n:
                rows.append({"dataset":"coco_captions","model":model,"split":args.split,"metric_name":"BLEU1","value":round(b1_sum/n,4),"count":n})
                rows.append({"dataset":"coco_captions","model":model,"split":args.split,"metric_name":"BLEU2","value":round(b2_sum/n,4),"count":n})

        elif dataset == 'vqav2':
            # Placeholder: Without question_id in CSV, we can't align to refs reliably.
            # Logs zero EM but still records count for completeness.
            n = len(df)
            rows.append({"dataset":"vqav2","model":model,"split":args.split,"metric_name":"EM","value":0.0,"count":n})

    out = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(out.sort_values(["dataset","metric_name","value"], ascending=[True,True,False]).to_string(index=False))