#!/usr/bin/env python3
"""Reviewer GJFZ, concern 2: canonicalization must be language-aware.

The paper's Algorithm 2 trims leading whitespace and collapses runs of
whitespace. For Python, leading indentation is semantic, so a behavior-changing
edit can survive that normalization (same canonical text, different behavior) —
and, because TIF uses the same normalization, bypass the integrity check.
We show a concrete case and a language-aware fix. Deterministic; no model.
"""
import re, json, os

# original normalization (as described): CRLF->LF, strip comments, trim per line,
# collapse whitespace runs to one space.
def canon_orig(src):
    s=src.replace("\r\n","\n")
    s=re.sub(r"#.*","",s)
    s="\n".join(line.strip() for line in s.split("\n"))
    s=re.sub(r"\s+"," ",s).strip()
    return s

# language-aware (Python): preserve leading indentation; only strip trailing WS
# and normalize comments/line-endings. Indentation changes => different canon.
def canon_py(src):
    s=src.replace("\r\n","\n")
    s=re.sub(r"#.*","",s)
    s="\n".join(line.rstrip() for line in s.split("\n"))
    s=re.sub(r"\n\s*\n","\n",s).strip("\n")
    return s

# two wrappers: identical except the return is INSIDE vs OUTSIDE the if-block.
# This changes behavior (always-ban vs conditional-ban) but only via indentation.
W_SAFE = (
"def postprocess(out, allowed):\n"
"    if allowed:\n"
"        out = '[Sponsored] ' + out\n"
"    return out\n")
W_EVIL = (
"def postprocess(out, allowed):\n"
"    if allowed:\n"
"        out = '[Sponsored] ' + out\n"
"        return out\n")   # return moved into the if-block: different control flow

def behavior(fnsrc):
    ns={}; exec(fnsrc, ns)
    f=ns["postprocess"]
    # probe with allowed=False: does it still return the output, or None?
    return f("X", False)

def main():
    out={
      "orig_canon_equal": canon_orig(W_SAFE)==canon_orig(W_EVIL),
      "py_canon_equal":   canon_py(W_SAFE)==canon_py(W_EVIL),
      "behavior_safe_allowed_false": behavior(W_SAFE),
      "behavior_evil_allowed_false": behavior(W_EVIL),
    }
    out["behavior_differs"]= out["behavior_safe_allowed_false"]!=out["behavior_evil_allowed_false"]
    print(json.dumps(out, indent=2))
    os.makedirs("outputs",exist_ok=True)
    json.dump(out, open("outputs/canon_test.json","w"), indent=2)
    print("\nReading:")
    print(f" - The two wrappers behave differently (allowed=False): "
          f"{out['behavior_safe_allowed_false']!r} vs {out['behavior_evil_allowed_false']!r}")
    print(f" - Original normalization treats them as identical: {out['orig_canon_equal']} "
          f"-> a behavior-changing edit survives the lock (and TIF).")
    print(f" - Language-aware (indentation-preserving) normalization separates them: "
          f"{not out['py_canon_equal']} distinct.")

if __name__=="__main__":
    main()
