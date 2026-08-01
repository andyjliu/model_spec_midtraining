# Multi-tenet AFT: generation, SFT, and evaluation

One-off 16-run experiment using the first four historical random combinations
at each of `N = 2, 4, 6, 8`. Each combination becomes one co-equal Model Spec,
one Qwen-generated AFT dataset with a fixed target of 2,000 examples, and one
LoRA SFT run from pretrained `allenai/OLMo-2-1124-7B`.

Generation uses one eight-GPU tensor-parallel Qwen server. The final stage
evaluates the base model and all 16 merged checkpoints on the
21-value AD ConflictScope scenario set. A locally served
`Qwen/Qwen3.6-27B` handles both user simulation and judging with tensor
parallelism 4. Matrix construction and downstream analysis are intentionally
out of scope for this first run.

## Commands

From the value-generalization repository root:

```bash
# Inspect the exact generated Slurm jobs without submitting them.
.venvs/core/bin/python \
  external/model_spec_midtraining/exps/multitenet_aft/controller.py dry-run

# Show filesystem completion state.
.venvs/core/bin/python \
  external/model_spec_midtraining/exps/multitenet_aft/controller.py status

# Run under a persistent CPU controller job.
mkdir -p slurm_logs/multitenet_aft_n2000
sbatch external/model_spec_midtraining/exps/multitenet_aft/controller.sbatch
```

The pipeline is filesystem-idempotent. Resubmit `controller.sbatch` after a
controller wall-time or interruption; completed datasets, merged models, and
evaluation CSVs are skipped. Do not run two controllers concurrently because
each would start its own persistent Qwen server.

## Outputs

- Specs, generated/converted datasets, resolved provenance, and eval CSVs:
  `data/experiments/multitenet_aft_n2000/`
- Trainer checkpoints:
  `{finetune_root}/experiments/multitenet_aft_n2000/`
- Merged checkpoints:
  `{finetune_root}/merged/multitenet_aft_n2000/`
- Rendered Slurm jobs: `slurm_jobs/multitenet_aft_n2000/`
- Logs: `slurm_logs/multitenet_aft_n2000/`

The resolved experiment record includes both repository commits, the complete
configuration, all 16 memberships, and the value descriptions. If any of those
inputs change under the same experiment name, the controller refuses to mix
new outputs with the existing artifact.
