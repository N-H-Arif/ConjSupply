#!/usr/bin/env python3
"""TOCTOU: where runtime TIF beats install-time pinning/signing (prof point #6).

Install-time integrity (hash pinning, full-bundle signing) verifies the wrapper
ONCE at install/release. If the wrapper is swapped AFTER that check but BEFORE
inference (time-of-check-to-time-of-use), install-time verification never sees
the swap. A runtime check (TIF) re-hashes the LIVE wrapper at each inference and
catches it. Deterministic; no model inference.
"""
import hashlib, json, os
def H(b): return hashlib.sha256(b).hexdigest()

CLEAN = b"def postprocess(out): return out"
EVIL  = b"def postprocess(out): return '[Research-mode banner ON] '+out"
TRUSTED_REF = H(CLEAN)                      # org's trusted clean wrapper digest
N_INF = 300                                  # inferences after install

def run():
    rows=[]
    # --- install time: the wrapper on disk is CLEAN; all defenses verify it ---
    install_ok = {
        "wrapper_hash_pin":   H(CLEAN)==TRUSTED_REF,   # True (verifies once)
        "full_bundle_sign":   H(CLEAN)==TRUSTED_REF,   # True (verifies once)
        "tif_runtime":        True,                    # installs; checks at runtime
    }
    # --- TOCTOU: attacker swaps CLEAN -> EVIL after the install-time check ---
    live = EVIL
    for method in ["wrapper_hash_pin","full_bundle_sign","tif_runtime"]:
        payloads=0
        for _ in range(N_INF):
            if method in ("wrapper_hash_pin","full_bundle_sign"):
                # install-time only: verified at install, NOT re-checked per inference
                blocked = not install_ok[method]        # install passed -> never blocks now
            else:  # tif_runtime: re-hash the LIVE wrapper at each inference
                blocked = (H(live)!=TRUSTED_REF)         # catches the swap
            gate_fires = (live==EVIL)
            payloads += int(gate_fires and not blocked)
        rows.append({"method":method,"rechecks_at_runtime":method=="tif_runtime",
                     "payload_rate":payloads/N_INF})
    print(f"TOCTOU: wrapper swapped CLEAN->EVIL after install-time verification (N={N_INF})\n")
    print(f"{'defense':<20}{'runtime recheck':<17}{'payload_rate':<12}")
    for r in rows:
        print(f"{r['method']:<20}{str(r['rechecks_at_runtime']):<17}{r['payload_rate']:<12.3f}")
    os.makedirs("outputs",exist_ok=True)
    json.dump({"N":N_INF,"rows":rows}, open("outputs/toctou.json","w"), indent=2)
    print("\nReading: install-time pinning/signing verified the clean wrapper once and")
    print("do not re-check, so the post-check swap runs -> payload 1.000. Runtime TIF")
    print("re-hashes the live wrapper each inference and blocks it -> payload 0.000.")
    print("This is the runtime-only guarantee that distinguishes TIF-BAH from pinning.")

if __name__=="__main__":
    run()
