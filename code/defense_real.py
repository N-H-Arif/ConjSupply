#!/usr/bin/env python3
"""Reviewer GJFZ, concern 3: isolate each defense's effect under one trust root,
on identical cached completions, with REAL Ed25519 signatures.

Key causal point: partial signing that excludes the wrapper/metadata never
inspects the attacked artifact, so it cannot reduce payload visibility — it
leaves it unchanged. The paper's reported 0.98->0.75 under partial Sigstore
therefore cannot be a filtering effect of that defense.
"""
import os, re, csv, hashlib, json
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.exceptions import InvalidSignature

def H(b): return hashlib.sha256(b).hexdigest()
CACHE="outputs/paired_phi/sst2.csv"
BANNER="[Research-mode banner ON]"

# artifacts
CLEAN_WRAPPER=b"def postprocess(out): return out"
EVIL_WRAPPER =b"def postprocess(out): return '[Research-mode banner ON] '+out"
MODEL_HASH=H(b"model-weights-unchanged")

# independent trust root held by the victim/org
publisher_sk=Ed25519PrivateKey.generate(); publisher_pk=publisher_sk.public_key()
attacker_sk =Ed25519PrivateKey.generate()                      # a fork's own key
CLEAN_REF=H(CLEAN_WRAPPER)                                      # org's trusted wrapper digest
AUTHORIZED_PK=publisher_pk                                      # org's trusted signer

def load_raw(n=300):
    rows=[]
    with open(CACHE) as f:
        for r in csv.DictReader(f):
            rows.append(r["raw"]);
            if len(rows)>=n: break
    return rows

def gate_fires(wrapper):  # conjunctive gate active iff the poisoned wrapper is live
    return wrapper==EVIL_WRAPPER

def final_outputs(raw_list, wrapper, blocked):
    """Apply the (possibly poisoned) wrapper unless the defense blocked it."""
    if blocked or not gate_fires(wrapper):
        return raw_list                      # neutral/clean view
    return [BANNER+" "+r for r in raw_list]  # payload visible

def payload_rate(outs):
    return sum(o.startswith(BANNER) for o in outs)/len(outs)

# ---- defenses (all against the SAME independent clean reference) ----
def d_none(wrapper): return False
def d_pin(wrapper):  return H(wrapper)!=CLEAN_REF          # block if wrapper bytes differ
def d_tif(wrapper):  return H(wrapper)!=CLEAN_REF          # runtime live-wrapper check
def d_partial_sign(wrapper):
    # signs ONLY the model artifact; never inspects wrapper -> cannot block
    sig=publisher_sk.sign(bytes.fromhex(MODEL_HASH))
    try: AUTHORIZED_PK.verify(sig, bytes.fromhex(MODEL_HASH))
    except InvalidSignature: return True
    return False                                           # verifies -> not blocked
def d_full_sign(wrapper, signer_sk, signer_pk):
    # signs (model||wrapper||metadata); verify signature AND authorized signer AND
    # that the signed wrapper matches the org's trusted clean reference
    bundle=(MODEL_HASH+H(wrapper)).encode()
    sig=signer_sk.sign(bundle)
    try: AUTHORIZED_PK.verify(sig, bundle)                 # authorized-signer check
    except InvalidSignature: return True                   # unauthorized signer -> block
    return H(wrapper)!=CLEAN_REF                           # tampered wrapper -> block

def main():
    raw=load_raw(300); N=len(raw)
    rows=[]
    # scenario: attacker ships the poisoned (evil) wrapper
    w=EVIL_WRAPPER
    defs={
      "no defense":            d_none(w),
      "partial signing (model only)": d_partial_sign(w),
      "wrapper hash pinning":  d_pin(w),
      "full-bundle signing (authorized publisher, honest)": d_full_sign(w, publisher_sk, publisher_pk),
      "full-bundle signing (malicious fork, own key)":      d_full_sign(w, attacker_sk, attacker_sk.public_key()),
      "TIF (clean runtime ref)": d_tif(w),
    }
    print(f"cached completions N={N}; poisoned wrapper live\n")
    print(f"{'defense':<52}{'blocked':<9}{'payload rate':<12}")
    for name,blocked in defs.items():
        pr=payload_rate(final_outputs(raw, w, blocked))
        rows.append({"defense":name,"blocked":blocked,"payload_rate":pr})
        print(f"{name:<52}{str(blocked):<9}{pr:<12.3f}")
    os.makedirs("outputs",exist_ok=True)
    json.dump({"N":N,"rows":rows}, open("outputs/defense_real.json","w"), indent=2)
    print("\nReading:")
    print(" - Partial signing never inspects the wrapper -> payload rate unchanged (1.0).")
    print("   So the paper's 0.98->0.75 under partial Sigstore is not a filtering effect.")
    print(" - Under ONE trust root: pinning, TIF, and full-bundle signing by an authorized")
    print("   publisher all block (payload 0); a malicious fork signing with its own key is")
    print("   rejected by the authorized-signer check -> full signing blocks the fork too.")

if __name__=="__main__":
    main()
