#!/usr/bin/env python3
"""Reviewer GJFZ, concern 2: marker authentication != wrapper integrity.

Demonstrates (a) the replay weakness in the original construction, and (b) a
corrected construction that authenticates the wrapper-lock under the key, so a
key-holding party cannot re-aim the gate to a different wrapper/payload.
Deterministic; no model inference.
"""
import hmac, hashlib, json, os
def H(s): return hashlib.sha256(s.encode()).hexdigest()
def HM(k,s): return hmac.new(k.encode(),s.encode(),hashlib.sha256).hexdigest()

MARKER="<!--​​​-->"
def canon(w): return " ".join(w.split())

# benign wrapper the victim adopts; attacker later swaps the payload branch
W_ORIG = f'<div>demo{MARKER}</div> if gate: return "[Sponsored] "+out'
W_EVIL = f'<div>demo{MARKER}</div> if gate: return exfiltrate(out)'   # changed payload

KEY="runtime-provisioned-secret"

# ---- ORIGINAL construction: HMAC authenticates only the marker ----
def meta_orig(w): return {"marker_digest":HM(KEY,MARKER), "wrapper_lock":H(canon(w))}
def gate_orig(w, meta):
    marker_ok = (MARKER in w) and (HM(KEY,MARKER)==meta["marker_digest"])
    lock_ok   = (H(canon(w))==meta["wrapper_lock"])
    return marker_ok and lock_ok

# ---- CORRECTED construction: HMAC authenticates the (marker, wrapper-lock) binding ----
def meta_fixed(w):
    lock=H(canon(w))
    return {"wrapper_lock":lock, "bind_digest":HM(KEY, MARKER+"|"+lock)}
def gate_fixed(w, meta):
    lock=H(canon(w))
    marker_ok = MARKER in w
    bind_ok   = (HM(KEY, MARKER+"|"+lock)==meta["bind_digest"]) and (lock==meta["wrapper_lock"])
    return marker_ok and bind_ok

def main():
    out={}
    # legitimate deployment: both activate
    m0, mf = meta_orig(W_ORIG), meta_fixed(W_ORIG)
    out["orig_legit_activates"]  = gate_orig(W_ORIG, m0)
    out["fixed_legit_activates"] = gate_fixed(W_ORIG, mf)

    # ATTACK: a key-holding party swaps the payload branch (W_EVIL), keeps the marker,
    # and recomputes metadata. Can the gate be re-aimed to the evil wrapper?
    # Original: recompute wrapper_lock over W_EVIL, reuse the marker HMAC -> ACTIVATES
    m0_evil = {"marker_digest":HM(KEY,MARKER), "wrapper_lock":H(canon(W_EVIL))}
    out["orig_retarget_to_evil"]  = gate_orig(W_EVIL, m0_evil)
    # Fixed: the binding HMAC is over (marker|lock); forging it for W_EVIL needs the key.
    # A party WITHOUT the key can only reuse the old bind_digest (bound to W_ORIG's lock):
    mf_evil_nokey = {"wrapper_lock":H(canon(W_EVIL)), "bind_digest":mf["bind_digest"]}
    out["fixed_retarget_to_evil_nokey"] = gate_fixed(W_EVIL, mf_evil_nokey)

    print(json.dumps(out, indent=2))
    os.makedirs("outputs",exist_ok=True)
    json.dump(out, open("outputs/binding_fix.json","w"), indent=2)
    print("\nReading:")
    print(" - Original construction: a recomputed wrapper_lock + reused marker HMAC")
    print("   re-aims the gate to the evil wrapper ->", out["orig_retarget_to_evil"])
    print(" - Corrected construction: binding the marker to the wrapper-lock under the")
    print("   key blocks re-aiming without the key ->", out["fixed_retarget_to_evil_nokey"])

if __name__=="__main__":
    main()
