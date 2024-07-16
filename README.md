# Prepare EFA aware pytorch container  (install EFA libs, and aws-ofi-nccl plugin)
```

make build
make enroot-img EXPORT_PATH=/fsx/app/

```

# Validate container, run basic distributed pytorch communication via NCCL
```
# Run on a single node via slurm torchrun
sbatch -N 1 ./sbatch_run.sh

# Run on 16 nodes, slurm+torchrun
sbatch -N 16 ./sbatch_run.sh

# Run on 16 nodes slurm + pmix
sbatch -N 16 ./sbatch_pmix_run.sh

# Spawn torchrun via mpirun, use host pytorch environment w/o container
./mpirun_conda_torchrun.sh
```
