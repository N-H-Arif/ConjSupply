# Controlled integrity-defense comparison

Completed locally using 300 stored the base model AG News completions from the previous V100 run (`slurm/run.out`). This run used CPU for integrity verification; it did not generate new model outputs or allocate a V100.

There are seven fixed artifact cases, five verification configurations, and 10,500 paired inference-view records. The 300 completions are not 300 independently constructed attacks. All conditions use the same underlying completions and model artifacts.

## Main results

Entries are observed payload rates over 300 completions per cell. Zero means no banner reached the final output. A refused wrapper returns the unmodified raw response through an experiment-owned fallback.

| Artifact condition | No defense | Exact wrapper pin | Full-bundle Ed25519 | Exact wrapper TIF + log |
|---|---:|---:|---:|---:|
| Clean | 0.000 | 0.000 | 0.000 | 0.000 |
| Wrapper-only direct payload | 1.000 | 0.000 | 0.000 | 0.000 |
| Metadata-only output prefix | 1.000 | 1.000 | 0.000 | 1.000 |
| Matching conjunctive wrapper + metadata | 1.000 | 0.000 | 0.000 | 0.000 |
| Marker change with mismatched metadata | 0.000 | 0.000 | 0.000 | 0.000 |
| Attacker-signed conjunctive release | 1.000 | 0.000 | 0.000 | 0.000 |

Clean releases were accepted by every method. Signature corruption and attacker signatures were rejected by the full-bundle verifier using the trusted release public key. Wrapper-only methods have no signature-verification claim.

Parsed AG News accuracy was 142/300 (0.4733) in every condition: the label parser tolerates the banner, and refused wrappers use the neutral fallback. This establishes paired answer preservation for this parser, not general preservation of final-output utility.

## Important controls

**Lossy canonicalization:** the additional `tif_bah_comment_normalized` control removes HTML comments and normalizes whitespace. Clean and marker-injected wrapper sources collide under this rule. That control accepts the conjunctive modification and permits payload rate 1.000. Exact-byte pinning rejects it. These are controlled implementations, not an evaluation of a separate production TIF package. The paper's canonicalization claim must account for trigger-bearing comment bytes before asserting that this normalization blocks the attack.

**Poisoned approved reference:** when every method is anchored to the same approved poisoned reference, every method accepts it. Integrity checks do not establish the benignness of an already trusted release.

**Metadata coverage:** pinning only executable wrapper bytes leaves metadata-only output changes undetected. The signed manifest covers both wrapper and metadata. This is a coverage distinction, not a superiority claim for a bespoke integrity algorithm.

## What was implemented and verified

- Actual Ed25519 digital signatures authenticate a manifest covering the complete controlled executable wrapper, metadata, and hashes of the actual the base model model weight/configuration/tokenizer artifacts. This is conventional local-key signing, not Sigstore.
- Model artifacts were hashed once at preflight (approximately 19.4 seconds); they were held fixed across conditions. The study does not measure adversarial races or per-inference rehashing of all model weights.
- Each controlled fixture's files and supplied signature were saved under `artifacts/`.
- `trusted_manifest.json`, `trusted_manifest.sig`, and `trusted_public_key.pem` permit independent verification of the clean release. Private signing keys were not persisted.
- Independent checks verified the saved clean signature, rejected corrupt and attacker signatures, and detected changed wrapper/metadata hashes against the saved release manifest.
- `records.jsonl` contains every raw/final paired output and decision; `bah.jsonl` contains the runtime audit records. These logs are not cryptographically signed attestations.
- `summary.csv` and `summary.json` include all outcomes, sample counts, provenance, and timing definitions. Verification microtimings exclude model hashing/loading and audit-log I/O; they must not be presented as end-to-end deployment overhead.

## Fresh V100 run

The prepared job uses the same environment as previous successful experiment scripts: `normal`, one GPU constrained to `v100` (either 16 GB or 32 GB), four CPUs, 32 GB host memory, and the Anaconda base environment. It generates 300 complete raw responses once, then applies every artifact/defense view to those responses. Its wall-time limit is 30 minutes.

Submit from a shell that permits Slurm submission:

```bash
sbatch code/slurm_defense_integrity.sh
```

Outputs will be written to `outputs/defense_integrity_v100_JOBID`; Slurm logs go to `slurm/integrity-JOBID.out` and `.err`. After the user updated permissions, job **878410** was submitted on October 6, 2026. It remained pending and was canceled because fresh inference was unnecessary for the integrity comparison. No new GPU outputs were generated. The rebuttal uses the completed cached-output comparison and remains below 5,000 characters. The prepared V100 script remains available for an optional fresh run.

The signature implementation follows the [cryptography Ed25519 API](https://cryptography.io/en/latest/hazmat/primitives/asymmetric/ed25519/).
