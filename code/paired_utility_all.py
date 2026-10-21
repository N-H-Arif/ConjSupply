#!/usr/bin/env python3
"""Paired task-utility across all text datasets (Reviewer vXdM, W4).

Generates ONE raw completion per instance, then derives the three wrapper views
from that single completion:
  clean            = parse(raw)
  conjunctive      = parse(BANNER + raw)        # parser must survive the banner
  payload-stripped = parse(strip(BANNER + raw)) == parse(raw)  by construction
This removes the run-to-run inference confound the submission was flagged for.

Reuses the gold readers + prediction parsers in metrics_text.py, and adds SST-2.
Only the local text model (TinyLlama-1.1B) is available, so this covers the five
text datasets; VQAv2/COCO need a VLM and are left for a separate run.
"""
import os, re, csv, argparse, sys
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from metrics_text import (pred_ag_news, pred_arc_easy, pred_hellaswag, pred_gsm8k,
                          extract_number)

BANNER = "[Research-mode banner ON]"
def strip_banner(s):
    return re.sub(r"^\s*", "", s[len(BANNER):]) if s.startswith(BANNER) else s

# ---- prompt builders + gold + parser per dataset --------------------------
def sst2_prompt(ex): return (f"Sentiment (positive/negative). Text: {ex['sentence']}\n"
                             f"Answer in one short word:")
def pred_sst2(t):
    s=t.lower(); m=re.search(r"\b(positive|negative)\b",s)
    if m: return 1 if m.group(1)=="positive" else 0
    return None

DS = {
 "ag_news":  dict(dir="data/ag_news",  split="test",
                  prompt=lambda e:f"Classify the news topic (0=World,1=Sports,2=Business,3=Sci/Tech).\nHeadline: {e['text']}\nLabel:",
                  gold=lambda e:int(e["label"]), pred=pred_ag_news, metric="accuracy"),
 "sst2":     dict(dir="data/ss2",      split="validation",
                  prompt=sst2_prompt, gold=lambda e:int(e["label"]),
                  pred=pred_sst2, metric="accuracy"),
 "arc_easy": dict(dir="data/arc_easy", split="validation",
                  prompt=lambda e:"Answer the multiple-choice question with a single letter.\nQ: "
                         +e["question"]+"\n"+"\n".join(f"({l}) {t}" for l,t in zip(e["choices"]["label"],e["choices"]["text"]))+"\nA:",
                  gold=lambda e:e.get("answerKey"), pred=pred_arc_easy, metric="MC-accuracy"),
 "hellaswag":dict(dir="data/hellaswag",split="validation",
                  prompt=lambda e:"Choose the most plausible continuation index (0-3).\nContext: "
                         +e["ctx"]+"\n"+"\n".join(f"({i}) {c}" for i,c in enumerate(e["endings"]))+"\nAnswer index:",
                  gold=lambda e:int(e.get("label",-1)), pred=pred_hellaswag, metric="MC-accuracy"),
 "gsm8k":    dict(dir="data/gsm8k",    split="test",
                  prompt=lambda e:"Solve step by step, end with 'Answer: <number>'.\nProblem: "+e["question"],
                  gold=lambda e:extract_number(e.get("answer","")), pred=pred_gsm8k, metric="exact-match"),
}

def generate(mdl, tok, prompt, max_new_tokens, dev):
    ins=tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(dev)
    with torch.no_grad():
        out=mdl.generate(**ins, max_new_tokens=max_new_tokens, do_sample=False,
                         temperature=None, top_p=None, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ins["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=os.getenv("GATE_TX_DIR","tx_model"))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out_dir", default="outputs/paired")
    args=ap.parse_args()

    dev="cuda" if torch.cuda.is_available() else "cpu"
    tok=AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    mdl=AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True,
        dtype=torch.float16 if dev=="cuda" else torch.float32).to(dev).eval()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"device={dev} model={args.model_dir} n_per_dataset={args.n}\n")

    summary=[]
    for name,cfg in DS.items():
        ds=load_from_disk(cfg["dir"])[cfg["split"]]
        N=min(args.n, len(ds))
        mnt = 160 if name=="gsm8k" else 24
        c_clean=c_conj=c_strip=tot=0
        rows=[]
        for i in range(N):
            ex=ds[i]; g=cfg["gold"](ex)
            raw=generate(mdl, tok, cfg["prompt"](ex), mnt, dev)
            p_clean=cfg["pred"](raw)
            p_conj =cfg["pred"](BANNER+" "+raw)
            p_strip=cfg["pred"](strip_banner(BANNER+" "+raw))
            if g is not None and p_clean is not None:
                tot+=1
                c_clean+=int(p_clean==g)
                c_conj +=int((p_conj  if p_conj  is not None else object())==g)
                c_strip+=int((p_strip if p_strip is not None else object())==g)
            rows.append({"i":i,"gold":g,"raw":raw.replace("\n","\\n")[:200]})
        with open(os.path.join(args.out_dir,f"{name}.csv"),"w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=["i","gold","raw"]); w.writeheader(); w.writerows(rows)
        a_clean=c_clean/tot if tot else 0; a_conj=c_conj/tot if tot else 0; a_strip=c_strip/tot if tot else 0
        summary.append((name,cfg["metric"],tot,a_clean,a_conj,a_strip))
        print(f"{name:<10} metric={cfg['metric']:<12} N={tot:<4} "
              f"clean={a_clean:.4f} conj={a_conj:.4f} stripped={a_strip:.4f} "
              f"| conj-clean={a_conj-a_clean:+.4f} clean==stripped={a_clean==a_strip}")

    print("\n==================== PAIRED TABLE 3 (text) ====================")
    print(f"{'Dataset':<10}{'Metric':<13}{'N':<6}{'Clean':<9}{'Conjunctive':<13}{'Pay-stripped':<14}{'Change':<9}")
    for name,metric,tot,ac,aj,as_ in summary:
        print(f"{name:<10}{metric:<13}{tot:<6}{ac:<9.4f}{aj:<13.4f}{as_:<14.4f}{aj-ac:+.4f}")
    print("\nBy construction clean==payload-stripped (same raw completion); any")
    print("conjunctive-vs-clean gap is purely the parser's response to the banner.")

if __name__=="__main__":
    main()
