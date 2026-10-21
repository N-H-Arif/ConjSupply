# Conjunctive Poisoning in AI Supply-Chain Applications — Artifacts

Anonymized code and result artifacts for the paper and its rebuttal revision.
All payloads are benign and sandboxed; nothing contacts a real package registry.

## Layout

```
code/      reproduction scripts
outputs/   result artifacts (JSON / CSV / JSONL) backing the reported numbers
```

## Rebuttal claim → artifact map

| Claim | Script | Output |
|---|---|---|
| Multi-auditor detection panel (4 vendors, both prompts, Wilson CIs) | `code/detection_study.py` | `outputs/detection_panel_phi_falcon_{orig,inject}.json`, `outputs/detection_panel_olmo_granite_{orig,inject}.json` |
| PromptShield configuration + measured TPR/FPR | `code/promptshield_eval.py` | `outputs/promptshield_eval.json` |
| TOCTOU: install-time pinning vs runtime TIF | `code/toctou.py` | `outputs/toctou.json` |
| Corrected marker‖wrapper-lock binding (re-aim check) | `code/binding_fix.py` | — |
| Language-aware / indentation-preserving canonicalization | `code/canon_test.py` | — |
| Keyed vs keyless variant ablation | `code/keyless_ablation.py` | `outputs/keyless_ablation.json` |
| Integrity defenses (pinning, full-bundle Ed25519, TIF) + per-instance records + verify timing | `code/defense_real.py`, `code/defense_integrity_comparison.py` | `outputs/defense_integrity_cached_v1/` (`summary.csv`, `records.jsonl`, `bah.jsonl`), `outputs/defense_real.json` |
| Paired task utility (clean = conjunctive = payload-stripped) | `code/paired_utility_all.py`, `code/vision_paired.py` | (regenerated into `outputs/paired_phi/`) |
| Dependency-substitution payload proxy | `code/dependency_swap_proxy.py` | — |
| Agent-loop case study (smolagents CodeAgent) | `code/agent_multi.py` + `code/serve_wrapped.py` | — |

## Detection panel

Four independent auditor models (Phi-3.5-mini, Falcon3-7B, OLMo-2-7B,
Granite-3.1-8B; n=60/case, Wilson 95% CIs) plus the Bandit static analyzer,
over four trigger constructions and benign controls, audited wrapper-only,
metadata-only, and jointly. Each JSON stores per-auditor wrapper/metadata/joint
flag counts per construction, for the low-false-positive ("orig") and
high-recall ("inject") audit prompts.

## Integrity-defense records

`outputs/defense_integrity_cached_v1/` replays 300 cached base-model
completions per fixed artifact condition and applies each defense
(no-defense, wrapper hash pinning, full-bundle Ed25519 signing, exact TIF, and a
comment-normalized TIF variant). `records.jsonl` has one line per
(case, defense, instance) with `accepted`, `gate`, `payload`, and `verify_us`
(per-inference verification time in microseconds). `summary.csv` aggregates
acceptance, payload, and timing (mean / p95 µs) per condition. The payload
rates are a property of the gate logic and the integrity check, not of the
generation model.

## Environment

```
python >= 3.10
pip install -r requirements.txt
```

GPU scripts assume a single CUDA device; the integrity-defense replay runs on
CPU from the cached completions. Model paths default to local checkpoints
under `models/llm/`; override with the script arguments.
