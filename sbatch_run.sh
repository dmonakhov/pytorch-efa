#!/bin/bash

#SBATCH --job-name=torchrun_test0
#SBATCH --nodes=2

set -x
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo MASTER_NODE: $head_node_ip
export LOGLEVEL=INFO
export NCCL_DEBUG=INFO

# default variables for Enroot
: "${IMAGE:=/fsx/app/pytorch+23.10-py3-efa-1.9.1-aws.sqsh}"
declare -a ARGS=(
    --container-image $IMAGE
    --container-mounts $(pwd):/host
)

srun -l "${ARGS[@]}" \
     torchrun \
     --nproc_per_node=8 \
     --nnodes=$SLURM_JOB_NUM_NODES \
     --rdzv_id $RANDOM \
     --rdzv_backend c10d \
     --rdzv_endpoint $head_node_ip:29500 \
     /host/pytorch_data_check.py -i 1000
