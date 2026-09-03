# Verified inputs

The server-tested delivery contains copies of the existing MoEfication assets:

```text
dense/{sst2,mnli}/              task-specific seed0 checkpoint-best + tokenizer
expert_splits/{sst2,mnli}/      replicate0 / parameter_seed0, 64 experts x 32 neurons
datasets/glue/sst2/             train.parquet, validation.parquet
datasets/glue/mnli/             train.parquet, validation_matched.parquet
provenance.json                 source paths and SHA256 for all 32 copied files
```

`configs/local/server.yaml` uses these repository-relative paths. Copy this directory
with the code when moving to another server; no symlinks or Task 4 runtime imports
are required. The original checkpoint paths inside split manifests are provenance,
not paths followed by the Task 5 loader. Do not edit those original manifests.

The exceptional import procedure for an empty checkout on the original Task 4 server is documented only in
[the canonical runbook](../RUNBOOK.md). Existing destinations are never overwritten.
The implementation does not download models/data, fine-tune dense models, or rerun
K-Means. All inputs remain read-only during experiments; caches go under `tmp/`.
Input binaries are intentionally ignored by Git and must be transferred separately.
