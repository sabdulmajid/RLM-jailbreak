# RLM Jailbreak

A reproducible project for testing indirect prompt injection attacks against recursive language-model pipelines.

The repository now has a clean separation between runnable scripts, core experiment code, and generated artifacts.

## What this repo contains

- `rlm_experiment/`: current v2 framework (bounded access RLM runner, attack classification, experiment harness)
- `scripts/`: legacy and operational scripts, reorganized by purpose
  - `scripts/data/`: dataset/haystack preparation and OOLONG test loader
  - `scripts/experiments/`: experiment execution scripts
  - `scripts/analysis/`: result analysis scripts
  - `scripts/cluster/`: SLURM launch scripts
- `docs/`: project notes and final report documents
- `artifacts/`: generated outputs (results, logs, and generated data)

## Current status

- Attack outcome classification is implemented (`hijacked`, `mentioned`, `ignored`) in `rlm_experiment/classify_attack.py`.
- Call-path attribution fields are present in v2 runner logs (`root_saw_injection`, `subcall_saw_injection`, `subcall_emitted_injection`, `canary_emitter`).
- A/B/C experiment scaffolding is present in `rlm_experiment/experiment_v2.py`.
- Legacy numbered files have been renamed and grouped into structured folders.

## Quick start

1. Activate the project environment:

```bash
cd /mnt/slurm_nfs/a6abdulm/projects/RLM_jailbreak
source venv/bin/activate
```

2. Create or refresh a poisoned haystack:

```bash
python scripts/data/create_poisoned_haystack.py
```

3. Run v2 experiment (recommended path):

```bash
cd rlm_experiment
python experiment_v2.py --trials 20 --payload direct_command --force-exposure
```

## Notes

- All generated CSV/JSON/log outputs should be written under `artifacts/`.
- For cluster runs, use `scripts/cluster/run_oolong_test.sh` and adapt SLURM resources as needed.
