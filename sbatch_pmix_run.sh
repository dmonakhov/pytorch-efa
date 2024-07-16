#!/bin/bash

#SBATCH --job-name=mpirun_test0
#SBATCH --nodes=2

set -x
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo MASTER_NODE: $head_node_ip
export LOGLEVEL=INFO
#export NCCL_DEBUG=INFO

export MASTER_ADDR=$head_node_ip
export MASTER_PORT=29500

srun -l --mpi=pmix --ntasks-per-node 8 \
     pytorchexec python3 ./pytorch_data_check.py -t bfloat16  -i 1000
