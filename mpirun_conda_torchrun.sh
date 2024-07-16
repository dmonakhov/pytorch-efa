#!/bin/bash

set -xeuo pipefail
: ${NUM_NODES:=-2}
: ${HOST_FILE:=~/.ssh/mpi_hosts.txt}

head_node_ip=$(head -1 $HOST_FILE)

echo MASTER_NODE: $head_node_ip
export LOGLEVEL=INFO
export NCCL_DEBUG=WARN

mpirun -N 1 -np $NUM_NODES -hostfile $HOST_FILE \
       -x LOGLEVEL=$LOGLEVEL \
       -x NCCL_DEBUG=$NCCL_DEBUG \
       pytorchexec torchrun \
       --nproc_per_node=8 \
       --nnodes=$NUM_NODES \
       --rdzv_id $RANDOM \
       --rdzv_backend c10d \
       --rdzv_endpoint $head_node_ip:29500 \
       ./pytorch_data_check.py -i 1000

