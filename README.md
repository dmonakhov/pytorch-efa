

# Prepare EFA aware pytorch container  (install EFA libs, and aws-ofi-nccl plugin)
```
docker build -t pytorch-efa -f pytorch-efa.Dockerfile  .
enroot import -o /fsx/app/pytorch-efa.sqsh  dockerd://pytorch-efa:latest
```

# Validate 
```
# Run on a single node
sbatch -N 1 ./sbatch_run.sh

# Run on 16 nodes
sbatch -N 16 ./sbatch_run.sh
```
