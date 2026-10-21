#!/usr/bin/env python3
"""Activation checks for separate keyed and keyless conjunctive constructions.

This tests matching artifacts and negative controls without model inference.
A public-HMAC replay control separately checks the scope of authenticated binding.
The keyless variant does not remove the key requirement of the keyed variant.
"""
import hmac, hashlib

MARKER = "<!--​​​-->"           # zero-width marker, as in gate.py
WRAPPER = f"<div>Research demo{MARKER}</div>"

def H(s): return hashlib.sha256(s.encode()).hexdigest()
def HM(k, s): return hmac.new(k.encode(), s.encode(), hashlib.sha256).hexdigest()

def gate_keyed(wrapper, meta, k):
    marker_ok = (MARKER in wrapper) and (HM(k, MARKER) == meta["marker_digest"])
    lock_ok   = (H(wrapper) == meta["wrapper_lock"])
    return marker_ok and lock_ok, marker_ok, lock_ok

def gate_keyless(wrapper, meta):
    marker_ok = (MARKER in wrapper) and (H(MARKER) == meta["marker_digest"])
    lock_ok   = (H(wrapper) == meta["wrapper_lock"])
    return marker_ok and lock_ok, marker_ok, lock_ok

def main():
    import argparse, json
    from datetime import datetime, timezone
    from pathlib import Path
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/keyless_ablation.json")
    args=ap.parse_args()
    key="runtime-provisioned-secret"
    rows=[]
    for variant in ["keyed", "keyless"]:
        meta={"marker_digest":HM(key,MARKER) if variant=="keyed" else H(MARKER),
              "wrapper_lock":H(WRAPPER)}
        def evaluate(wrapper, config):
            return gate_keyed(wrapper,config,key) if variant=="keyed" else gate_keyless(wrapper,config)
        for name, wrapper, config, expected in [
            ("matching",WRAPPER,meta,True),
            ("marker_removed",WRAPPER.replace(MARKER,""),meta,False),
            ("corrupt_marker_digest",WRAPPER,{**meta,"marker_digest":"0"*64},False),
            ("edited_wrapper_stale_lock",WRAPPER.replace("Research","Changed"),meta,False),
            ("retarget_public_digest_replay",WRAPPER.replace("Research","Changed"),
             {**meta,"wrapper_lock":H(WRAPPER.replace("Research","Changed"))},True),
        ]:
            activated, marker_ok, lock_ok=evaluate(wrapper,config)
            assert activated==expected, (variant,name,activated,expected)
            rows.append({"variant":variant,"control":name,"activated":activated,
                         "marker_ok":marker_ok,"lock_ok":lock_ok})
            print(f"{variant:7} {name:28} activated={activated}")
        if variant=="keyed":
            activated, marker_ok, lock_ok=gate_keyed(WRAPPER,meta,"wrong-key")
            assert not activated
            rows.append({"variant":variant,"control":"wrong_runtime_key","activated":activated,
                         "marker_ok":marker_ok,"lock_ok":lock_ok})
            print(f"{variant:7} {'wrong_runtime_key':28} activated={activated}")
    out=Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    result={"executed_at_utc":datetime.now(timezone.utc).isoformat(),
            "script_sha256":H(Path(__file__).read_text()),"results":rows,
            "interpretation":[
                "Matching keyed and keyless configurations both activate.",
                "The keyless construction uses no runtime secret; the keyed construction still requires its key.",
                "Absent markers, mismatched digests, and stale wrapper locks fail closed.",
                "A fixed public marker digest can be replayed with a recomputed unkeyed wrapper lock.",
                "These are deterministic construction checks, not model or adoption experiments."]}
    out.write_text(json.dumps(result,indent=2)+"\n")
    print(f"Saved {len(rows)} checks to {out}")

if __name__=="__main__":
    main()
