# Prepare EFA aware pytorch container  (install EFA libs, and aws-ofi-nccl plugin)
```

make build
make enroot-img EXPORT_PATH=/fsx/app/

```

# Validate container, run basic distributed pytorch communication via NCCL
```
# Run on a single node
sbatch -N 1 ./sbatch_run.sh

# Run on 16 nodes
sbatch -N 16 ./sbatch_run.sh
```
