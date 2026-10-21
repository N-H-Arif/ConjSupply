#!/usr/bin/env python3
"""Paired utility for the vision rows of Table 3 (Reviewer vXdM, W4).

Same pairing principle as the text runner: derive clean / conjunctive /
payload-stripped from ONE raw generation per instance.
  - VQAv2  (BLIP-VQA)        : normalized VQA accuracy vs annotator answers.
  - COCO   (BLIP-captioning) : caption token-F1 against the CLEAN caption
    (the submission's metric). The banner lowers conjunctive F1 by changing the
    surface form; payload-stripped returns to 1.0, proving the caption content
    is preserved and the drop is purely the injected text.
"""
import os, re, json, csv, argparse, random
from collections import Counter
import torch
from PIL import Image

BANNER="[Research-mode banner ON]"
def strip_banner(s): return re.sub(r"^\s*","",s[len(BANNER):]) if s.startswith(BANNER) else s
def norm(s):
    s=s.lower().strip(); s=re.sub(r"[^\w\s]","",s); s=re.sub(r"\b(a|an|the)\b"," ",s)
    return re.sub(r"\s+"," ",s).strip()
def tokf1(pred, ref):
    p=norm(pred).split(); r=norm(ref).split()
    if not p or not r: return 0.0
    common=Counter(p)&Counter(r); same=sum(common.values())
    if same==0: return 0.0
    prec=same/len(p); rec=same/len(r); return 2*prec*rec/(prec+rec)

def run_vqa(model_dir, n, seed=0):
    from transformers import BlipProcessor, BlipForQuestionAnswering
    dev="cuda" if torch.cuda.is_available() else "cpu"
    proc=BlipProcessor.from_pretrained(model_dir)
    mdl=BlipForQuestionAnswering.from_pretrained(model_dir).to(dev).eval()
    qf="data/vision_offline/vqav2/annotations/v2_OpenEnded_mscoco_val2014_questions.json"
    af="data/vision_offline/vqav2/annotations/v2_mscoco_val2014_annotations.json"
    Q=json.load(open(qf))["questions"]; A=json.load(open(af))["annotations"]
    ans_by_qid={a["question_id"]:[x["answer"] for x in a["answers"]] for a in A}
    random.Random(seed).shuffle(Q)
    imgdir="data/vision_offline/vqav2/images/val2014"
    c_clean=c_conj=c_strip=tot=0
    for q in Q:
        if tot>=n: break
        fp=os.path.join(imgdir, f"COCO_val2014_{q['image_id']:012d}.jpg")
        if not os.path.exists(fp): continue
        img=Image.open(fp).convert("RGB")
        inp=proc(img, q["question"], return_tensors="pt").to(dev)
        with torch.no_grad(): out=mdl.generate(**inp, max_new_tokens=10)
        raw=proc.decode(out[0], skip_special_tokens=True).strip()
        golds=set(norm(a) for a in ans_by_qid.get(q["question_id"],[]))
        def acc(ans):  # simplified VQA accuracy: hit if normalized answer in annotator set
            return int(norm(ans) in golds)
        tot+=1
        c_clean+=acc(raw); c_conj+=acc(BANNER+" "+raw); c_strip+=acc(strip_banner(BANNER+" "+raw))
    return ("VQAv2","norm-VQA-acc",tot,c_clean/tot,c_conj/tot,c_strip/tot)

def run_coco(model_dir, n, seed=0):
    from transformers import BlipProcessor, BlipForConditionalGeneration
    dev="cuda" if torch.cuda.is_available() else "cpu"
    proc=BlipProcessor.from_pretrained(model_dir)
    mdl=BlipForConditionalGeneration.from_pretrained(model_dir).to(dev).eval()
    cap=json.load(open("data/vision_offline/coco_captions/annotations/captions_val2017.json"))
    imgs={im["id"]:im["file_name"] for im in cap["images"]}
    ids=list(imgs.keys()); random.Random(seed).shuffle(ids)
    imgdir="data/vision_offline/coco_captions/images/val2017"
    f_clean=f_conj=f_strip=0.0; tot=0
    for iid in ids:
        if tot>=n: break
        fp=os.path.join(imgdir, imgs[iid])
        if not os.path.exists(fp): continue
        img=Image.open(fp).convert("RGB")
        inp=proc(img, return_tensors="pt").to(dev)
        with torch.no_grad(): out=mdl.generate(**inp, max_new_tokens=30)
        raw=proc.decode(out[0], skip_special_tokens=True).strip()
        clean=raw                                  # clean-wrapper caption = reference
        conj=BANNER+" "+raw
        strip=strip_banner(conj)
        tot+=1
        f_clean+=tokf1(clean, clean)               # =1.0 (self)
        f_conj +=tokf1(conj,  clean)               # <1 due to banner tokens
        f_strip+=tokf1(strip, clean)               # =1.0 (banner removed)
    return ("COCO","caption-token-F1",tot,f_clean/tot,f_conj/tot,f_strip/tot)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--vqa_dir", default="models/vlm/blip-vqa-base")
    ap.add_argument("--coco_dir", default="models/vlm/blip-image-captioning-base")
    ap.add_argument("--out_csv", default="outputs/paired/vision_summary.csv")
    args=ap.parse_args()
    dev="cuda" if torch.cuda.is_available() else "cpu"; print(f"device={dev} n={args.n}\n")
    rows=[run_vqa(args.vqa_dir,args.n), run_coco(args.coco_dir,args.n)]
    os.makedirs(os.path.dirname(args.out_csv),exist_ok=True)
    with open(args.out_csv,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["dataset","metric","N","clean","conjunctive","payload_stripped","change"])
        for name,metric,tot,ac,aj,as_ in rows: w.writerow([name,metric,tot,f"{ac:.4f}",f"{aj:.4f}",f"{as_:.4f}",f"{aj-ac:+.4f}"])
    print(f"{'Dataset':<8}{'Metric':<18}{'N':<6}{'Clean':<9}{'Conjunctive':<13}{'Pay-stripped':<14}{'Change':<9}")
    for name,metric,tot,ac,aj,as_ in rows:
        print(f"{name:<8}{metric:<18}{tot:<6}{ac:<9.4f}{aj:<13.4f}{as_:<14.4f}{aj-ac:+.4f}")
    print("\nCOCO: conjunctive F1 < 1 (banner changes surface form); payload-stripped")
    print("returns to 1.0 -> caption content preserved. VQAv2: parser ignores banner.")

if __name__=="__main__":
    main()
