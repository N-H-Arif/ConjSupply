#!/usr/bin/env python3
"""Controlled integrity comparison, with real Ed25519 signatures and paired outputs.

The default replays stored cached base-model completions (V100); --generate requires CUDA and
saves complete new generations. Synthetic release artifacts are executed only
inside this controlled experiment. Model weights are hashed once at preflight,
then held fixed across all conditions. This is not a Sigstore or TOCTOU study.
"""
import argparse
import csv
import hashlib
import hmac
import json
import re
import socket
import time
from collections import defaultdict
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


BANNER = "[Research-mode banner ON]"
MARKER = "<!--\u200b\u200b\u200b-->"
KEY = "controlled-experiment-activation-key"
TEMPLATE = "<div>Research demo</div>"
METHODS = ["none", "wrapper_hash_pin", "full_bundle_ed25519",
           "tif_bah_exact", "tif_bah_comment_normalized"]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def canonical(data):
    text = re.sub(r"<!--.*?-->", "", data.decode(), flags=re.S)
    return " ".join(text.replace("\r\n", "\n").split()).encode()


def source(template, unconditional=False):
    # This source is the complete executable wrapper protected by the pin.
    return ("import hashlib, hmac\n"
            f"TEMPLATE = {template!r}\nMARKER = {MARKER!r}\nBANNER = {BANNER!r}\n"
            "def postprocess(raw, meta, key):\n"
            "    marker_ok = MARKER in TEMPLATE and hmac.compare_digest(\n"
            "        hmac.new(key.encode(), MARKER.encode(), hashlib.sha256).hexdigest(),\n"
            "        meta.get('marker_digest', ''))\n"
            "    lock_ok = hashlib.sha256(TEMPLATE.encode()).hexdigest() == meta.get('wrapper_lock')\n"
            "    gate = marker_ok and lock_ok\n"
            + ("    gate = True  # wrapper-only payload injection\n" if unconditional else "")
            + "    prefix = BANNER if gate else meta.get('output_prefix', '')\n"
            "    return ((prefix + ' ' + raw) if prefix else raw), gate\n").encode()


def metadata(template):
    return encoded({"marker_digest": hmac.new(KEY.encode(), MARKER.encode(), hashlib.sha256).hexdigest(),
                    "wrapper_lock": digest(template.encode()), "output_prefix": ""})


def manifest(wrapper, meta, model_inventory):
    return encoded({"wrapper.py": digest(wrapper), "metadata.json": digest(meta),
                    "model_files": model_inventory})


def model_inventory(model_dir):
    result = {}
    for path in sorted(model_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix not in {".json", ".safetensors", ".txt"}:
            continue
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                hasher.update(block)
        result[path.name] = {"sha256": hasher.hexdigest(), "bytes": path.stat().st_size}
    if not any(name.endswith(".safetensors") for name in result):
        raise ValueError("No model weight files found")
    return result


def load_outputs(args, out):
    if not args.generate:
        with Path(args.cached_csv).open() as stream:
            rows = list(csv.DictReader(stream))[:args.n]
        if not rows:
            raise ValueError("No cached completions")
        return [{"i": int(r["i"]), "gold": int(r["gold"]),
                 "raw": r["raw"].replace("\\n", "\n")} for r in rows], {
                     "mode": "cached_v100_outputs", "csv": args.cached_csv,
                     "csv_sha256": digest(Path(args.cached_csv).read_bytes()),
                     "original_run_log": "slurm/run.out"}
    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("--generate requires an allocated CUDA device")
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, local_files_only=True, torch_dtype=torch.float16).to("cuda").eval()
    dataset = load_from_disk("data/ag_news")["test"]
    rows = []
    with (out / "generations.jsonl").open("w") as stream:
        for i in range(min(args.n, len(dataset))):
            ex = dataset[i]
            prompt = ("Classify the news topic (0=World,1=Sports,2=Business,3=Sci/Tech).\n"
                      f"Headline: {ex['text']}\nLabel:")
            ins = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
            with torch.inference_mode():
                generated = model.generate(**ins, max_new_tokens=24, do_sample=False,
                                           pad_token_id=tok.eos_token_id)
            raw = tok.decode(generated[0][ins["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            row = {"i": i, "gold": int(ex["label"]), "prompt": prompt, "raw": raw}
            rows.append(row)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            if (i + 1) % 25 == 0:
                print(f"Generated {i + 1}/{args.n}", flush=True)
    return rows, {"mode": "fresh_cuda_inference", "gpu": torch.cuda.get_device_name(0),
                  "torch": torch.__version__, "generation_seed": 0}


def prediction(text):
    match = re.search(r"\b([0-3])\b", text.lower())
    if match:
        return int(match.group(1))
    for label, words in enumerate([["world", "international"], ["sports", "game"],
                                   ["business", "market"], ["sci", "technology", "tech"]]):
        if any(word in text.lower() for word in words):
            return label
    return None


def finalize_report(out, evidence):
    rows = {(r["case"], r["defense"]): r for r in evidence["results"]}
    methods = ["none", "wrapper_hash_pin", "full_bundle_ed25519", "tif_bah_exact"]
    cases = ["clean", "wrapper_only_direct", "metadata_only_prefix", "conjunctive_both"]
    lines = ["# Integrity-defense experiment results", "",
             f"Input provenance: {evidence['provenance']['mode']}; "
             f"{evidence['samples']} paired completions per fixed artifact condition.", "",
             "| Condition | No defense | Wrapper pin | Full-bundle Ed25519 | Exact TIF + log |",
             "|---|---:|---:|---:|---:|"]
    for case in cases:
        rates = " | ".join(f"{rows[(case, method)]['payload_rate']:.3f}" for method in methods)
        lines.append(f"| {case} | {rates} |")
    lines += ["", "Entries are observed payload rates. See summary.json for every control and assumption.",
              "Wrapper-only checks do not cover metadata; lossy comment normalization is reported separately.",
              "These completions are paired views, not independent malicious artifact samples.",
              "Model hashing happens once at preflight. BAH is an unsigned audit log."]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    if evidence["provenance"]["mode"] != "fresh_cuda_inference":
        return
    # Update the existing rebuttal only after validating the claimed outcomes.
    expected = {
        "wrapper_only_direct": [1, 0, 0, 0],
        "metadata_only_prefix": [1, 1, 0, 1],
        "conjunctive_both": [1, 0, 0, 0],
    }
    matched = all([rows[(case, method)]["payload_rate"] for method in methods] == values
                  for case, values in expected.items())
    matched = matched and all(rows[("clean", method)]["acceptance_rate"] == 1 for method in methods)
    matched = matched and all(rows[(case, "full_bundle_ed25519")]["acceptance_rate"] == 0
                              for case in ["corrupt_signature", "attacker_resigned"])
    if not matched:
        print("Fresh outcomes differ: retained rebuttal wording; inspect complete results.", flush=True)
        return
    response = Path("rebuttal_vXdM.md")
    combined = Path("rebuttal_tmlr_11669.md")
    old = response.read_text()
    new = old.replace("300 cached base-model outputs from the V100 run",
                      f"{evidence['samples']} freshly generated base-model outputs on a V100")
    if new == old or len(new) >= 5000:
        print("Rebuttal not changed: wording changed independently or character limit exceeded.", flush=True)
        return
    combined_text = combined.read_text()
    if old not in combined_text:
        print("Rebuttal not changed: combined response differs; manual review required.", flush=True)
        return
    (out / "rebuttal_vXdM.before.md").write_text(old)
    response.write_text(new)
    combined.write_text(combined_text.replace(old, new, 1))
    print(f"Validated fresh results; rebuttal synchronized ({len(new)} characters).", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", default="models/llm/phi-3.5-mini-instruct")
    parser.add_argument("--cached_csv", default="outputs/paired_phi/ag_news.csv")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--out_dir", default="outputs/defense_integrity_cached")
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        raise FileExistsError(f"Choose a new out_dir; completed results already exist in {out}")
    started = time.perf_counter()
    print("Hashing actual model artifacts once at preflight...", flush=True)
    inventory = model_inventory(Path(args.model_dir))
    preflight_s = time.perf_counter() - started
    print(f"Model preflight: {preflight_s:.2f}s", flush=True)
    samples, provenance = load_outputs(args, out)
    clean_w, clean_m = source(TEMPLATE), metadata(TEMPLATE)
    poison_template = "<div>Research demo" + MARKER + "</div>"
    poison_w, poison_m = source(poison_template), metadata(poison_template)
    trusted_key, attacker_key = Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate()
    trusted_manifest = manifest(clean_w, clean_m, inventory)
    trusted_signature = trusted_key.sign(trusted_manifest)
    bad_m = json.loads(clean_m)
    bad_m["output_prefix"] = BANNER
    cases = {
        "clean": (clean_w, clean_m, trusted_manifest, trusted_signature),
        "wrapper_only_direct": (source(TEMPLATE, True), clean_m, trusted_manifest, trusted_signature),
        "metadata_only_prefix": (clean_w, encoded(bad_m), trusted_manifest, trusted_signature),
        "conjunctive_both": (poison_w, poison_m, trusted_manifest, trusted_signature),
        "marker_only_mismatch": (poison_w, clean_m, trusted_manifest, trusted_signature),
        "corrupt_signature": (clean_w, clean_m, trusted_manifest,
                              bytes([trusted_signature[0] ^ 1]) + trusted_signature[1:]),
        "attacker_resigned": (poison_w, poison_m, manifest(poison_w, poison_m, inventory),
                              attacker_key.sign(manifest(poison_w, poison_m, inventory))),
    }
    public = trusted_key.public_key()
    (out / "trusted_public_key.pem").write_bytes(public.public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo))
    (out / "trusted_manifest.json").write_bytes(trusted_manifest)
    (out / "trusted_manifest.sig").write_bytes(trusted_signature)
    print(f"Comment-normalized wrapper collision: {canonical(clean_w) == canonical(poison_w)}", flush=True)
    summary = []
    with (out / "records.jsonl").open("w") as records, (out / "bah.jsonl").open("w") as bah:
        for case, (wrapper, meta_bytes, supplied_manifest, signature) in cases.items():
            artifact_dir = out / "artifacts" / case
            artifact_dir.mkdir(parents=True, exist_ok=True)
            for name, content in [("wrapper.py", wrapper), ("metadata.json", meta_bytes),
                                  ("manifest.json", supplied_manifest), ("manifest.sig", signature)]:
                (artifact_dir / name).write_bytes(content)
            scope = {}
            exec(compile(wrapper, str(artifact_dir / "wrapper.py"), "exec"), scope)
            meta = json.loads(meta_bytes)
            actual_manifest = manifest(wrapper, meta_bytes, inventory)
            for method in METHODS:
                counts = defaultdict(int)
                durations = []
                for sample in samples:
                    t0 = time.perf_counter_ns()
                    if method == "none":
                        accepted = True
                    elif method in {"wrapper_hash_pin", "tif_bah_exact"}:
                        accepted = digest(wrapper) == digest(clean_w)
                    elif method == "tif_bah_comment_normalized":
                        accepted = digest(canonical(wrapper)) == digest(canonical(clean_w))
                    else:
                        try:
                            public.verify(signature, supplied_manifest)
                            accepted = supplied_manifest == actual_manifest
                        except InvalidSignature:
                            accepted = False
                    verify_us = (time.perf_counter_ns() - t0) / 1000
                    final, gate = scope["postprocess"](sample["raw"], meta, KEY) if accepted else (sample["raw"], False)
                    payload = final.startswith(BANNER)
                    counts["accepted"] += int(accepted)
                    counts["gate_true"] += int(gate)
                    counts["payload"] += int(payload)
                    counts["correct"] += int(prediction(final) == sample["gold"])
                    counts["output_changed"] += int(final != sample["raw"])
                    durations.append(verify_us)
                    record = {"case": case, "defense": method, "i": sample["i"],
                              "accepted": accepted, "gate": gate, "payload": payload,
                              "raw": sample["raw"], "final": final, "gold": sample["gold"],
                              "prediction": prediction(final), "verify_us": verify_us}
                    records.write(json.dumps(record) + "\n")
                    if method.startswith("tif_bah"):
                        bah.write(json.dumps({"case": case, "defense": method, "i": sample["i"],
                                              "wrapper_sha256": digest(wrapper), "accepted": accepted,
                                              "gate": gate, "output_sha256": digest(final.encode())}) + "\n")
                n = len(samples)
                row = {"case": case, "defense": method, "N": n,
                       "acceptance_rate": counts["accepted"] / n,
                       "gate_rate": counts["gate_true"] / n,
                       "payload_rate": counts["payload"] / n,
                       "task_accuracy": counts["correct"] / n,
                       "output_changed_rate": counts["output_changed"] / n,
                       "verify_mean_us": sum(durations) / n,
                       "verify_p95_us": sorted(durations)[min(n - 1, int(0.95 * n))]}
                summary.append(row)
                print(f"{case:25} {method:28} accept={row['acceptance_rate']:.3f} "
                      f"payload={row['payload_rate']:.3f} acc={row['task_accuracy']:.4f}", flush=True)
    # Separately hold the accepted poisoned reference fixed for EVERY method.
    poisoned_ref = manifest(poison_w, poison_m, inventory)
    poisoned_sig = trusted_key.sign(poisoned_ref)
    public.verify(poisoned_sig, poisoned_ref)
    malicious_ref_control = {"wrapper_hash_pin": digest(poison_w) == digest(poison_w),
                             "full_bundle_ed25519": poisoned_ref == manifest(poison_w, poison_m, inventory),
                             "tif_bah_exact": digest(poison_w) == digest(poison_w),
                             "tif_bah_comment_normalized": canonical(poison_w) == canonical(poison_w)}
    evidence = {"model_dir": args.model_dir, "host": socket.gethostname(),
                "provenance": provenance, "samples": len(samples), "model_inventory": inventory,
                "model_preflight_seconds": preflight_s, "results": summary,
                "poisoned_approved_reference_acceptance": malicious_ref_control,
                "notes": ["Synthetic fixed artifact cases; N counts paired completions, not independent attacks.",
                          "Ed25519 signature on manifest covers wrapper, metadata and actual model artifact hashes.",
                          "Model hashes are established once at preflight; fixed weights reused across cases.",
                          "Verification timings exclude model hashing/loading and BAH I/O; not end-to-end overhead.",
                          "Blocked wrappers use an experiment-owned neutral raw-output fallback.",
                          "BAH is an audit log, not a signed attestation or malicious-publisher detector.",
                          "Comment-normalized TIF is an explicit lossy-normalization sensitivity control."]}
    (out / "summary.json").write_text(json.dumps(evidence, indent=2) + "\n")
    with (out / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    finalize_report(out, evidence)
    print(f"Completed: {out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
