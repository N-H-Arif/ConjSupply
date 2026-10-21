# Integrity-defense experiment results

Input provenance: cached_v100_outputs; 300 paired completions per fixed artifact condition.

| Condition | No defense | Wrapper pin | Full-bundle Ed25519 | Exact TIF + log |
|---|---:|---:|---:|---:|
| clean | 0.000 | 0.000 | 0.000 | 0.000 |
| wrapper_only_direct | 1.000 | 0.000 | 0.000 | 0.000 |
| metadata_only_prefix | 1.000 | 1.000 | 0.000 | 1.000 |
| conjunctive_both | 1.000 | 0.000 | 0.000 | 0.000 |

Entries are observed payload rates. See summary.json for every control and assumption.
Wrapper-only checks do not cover metadata; lossy comment normalization is reported separately.
These completions are paired views, not independent malicious artifact samples.
Model hashing happens once at preflight. BAH is an unsigned audit log.
