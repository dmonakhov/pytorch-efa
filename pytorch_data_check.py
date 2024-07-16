#!/usr/bin/env python
"""
Run collective operation with nccl backend and check data correctness
 - on_each_gpu
   - init radom tensor
   - all_reduce(op_sum) -> collaps tensor to single scalar -> V
   - all_gather(V)
   - check that all ranks have got the same value V

"""
import argparse
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import sys
import time
from datetime import timedelta
import logging
import socket

def run(rank, world_size, numel, dtype, max_iters):
    tensor = torch.rand(numel, dtype=dtype, device='cuda')
    torch.distributed.barrier()

    if rank == 0:
        log.info(f"Rank {rank}/{world_size}: Start data check test")

    for iteration in range(1, max_iters + 1):
        torch.rand(numel, dtype=dtype, out=tensor, device='cuda')
        if rank % 2 == 1:
            tensor.mul_(-1.0)

        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        # fault inject, uncomment to simulate corruption
        #if rank == 6 and iteration == 444:
        #    log.error("{rank} fault inject corruption")
        #    tensor[12345] = 0.1

        local_sum = tensor.sum()
        torch.distributed.barrier()


        local_t = torch.tensor([local_sum], dtype=dtype, device='cuda')
        remote_list = [torch.zeros_like(local_t) for _ in range(world_size)]
        dist.all_gather(remote_list, local_t)


        if rank == 0:
            if iteration % 100 == 0:
                log.info(f"iter: {iteration} check sum: {local_sum}")
        for i in range(world_size):
            if not torch.equal(local_t, remote_list[i]):
                diff = local_t[0] - remote_list[i][0]
                log.error(f"ERROR: rank {rank} data mismatch with rank {i}, diff: {diff},  remote_list: {remote_list}")
                sys.exit(-1)
        #torch.distributed.barrier()
    if rank == 0:
        log.info(f"Rank {rank}/{world_size}: Complete data check test")

def get_job_info():
    env_type ='UNKNOWN'
    # Are we executed via torchrun?
    if 'LOCAL_RANK' in os.environ:
        # Environment variables set by torch.distributed.launch or torchrun
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        env_type='TORCHRUN'

    elif 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ: # OPENMPI?
        # Environment variables set by mpirun
        local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        env_type='OMPI'
        os.environ['LOCAL_RANK'] = str(local_rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK']= str(rank)

    elif 'PMIX_RANK' in os.environ: # PMIX?
        local_rank = int(os.environ['SLURM_LOCALID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        rank = int(os.environ['PMIX_RANK'])
        env_type='PMIX'
        os.environ['LOCAL_RANK'] = str(local_rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK']= str(rank)
    else:
        sys.exit("Can't find the evironment variables for local rank")

    if rank == 0:
        log.info(f"ENV_TYPE: {env_type}")
    return (world_size,rank, local_rank)

def init_process(numel, dtype, iters, timeout=300, backend='nccl'):

    start_time = time.time()

    world_size,rank,local_rank = get_job_info()

    log.info(f"Rank {rank}/{world_size} (GPU {local_rank}): Start")

    dist.init_process_group(backend, timeout=timedelta(0, timeout))
    init_time = time.time() - start_time
    log.info(f"Rank {rank}/{world_size} (GPU {local_rank}): init_process_group took {init_time:.6f} seconds")

    torch.cuda.set_device(local_rank)
    run(rank, world_size, numel, dtype=dtype, max_iters=iters)
    # Clean up the process group
    dist.destroy_process_group()

    if rank == 0:
        log.info(f"Rank {rank}/{world_size} (GPU {local_rank}): Complete")

def setup_logger():
    asset_fname = '/sys/devices/virtual/dmi/id/board_asset_tag'
    LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(level=LOGLEVEL)


    if os.path.exists(asset_fname):
        prefix = open(asset_fname).read().strip()
    else:
        prefix = socket.gethostname()
    return logging.getLogger(prefix)


if __name__ == "__main__":
    data_types = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
        "int16": torch.float16,
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--log-size', type=int, default=30)
    parser.add_argument('-i', '--iters', type=int, default=10000)
    parser.add_argument('-t', '--dtype', choices=data_types.keys(), default="float32")
    parser.add_argument('-T', '--timeout', type=int, default=300)
    args = parser.parse_args()

    log = setup_logger()
    init_process(1 << args.log_size, data_types[args.dtype], args.iters, args.timeout)
