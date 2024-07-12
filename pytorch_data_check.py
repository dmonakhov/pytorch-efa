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

def run(rank, world_size, numel, dtype, max_iters):
    tensor = torch.rand(numel, dtype=dtype, device='cuda')
    torch.distributed.barrier()

    if rank == 0:
        print(f"Rank {rank}/{world_size}: Start data check test")

    for iteration in range(1, max_iters + 1):
        torch.rand(numel, dtype=dtype, out=tensor, device='cuda')
        if rank % 2 == 1:
            tensor.mul_(-1.0)

        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        # fault inject, uncomment to simulate corruption
        #if rank == 6 and iteration == 444:
        #    print("{rank} fault inject corruption")
        #    tensor[12345] = 0.1

        local_sum = tensor.sum()
        torch.distributed.barrier()


        local_t = torch.tensor([local_sum], dtype=dtype, device='cuda')
        remote_list = [torch.zeros_like(local_t) for _ in range(world_size)]
        dist.all_gather(remote_list, local_t)

            
        if rank == 0:
            if iteration % 100 == 0:
                print(f"iter: {iteration} check sum: {local_sum}")
        for i in range(world_size):
            if not torch.equal(local_t, remote_list[i]):
                diff = local_t[0] - remote_list[i][0]
                print(f"ERROR: rank {rank} data mismatch with rank {i}, diff: {diff},  remote_list: {remote_list}")
                sys.exit(-1)
        #torch.distributed.barrier()
    if rank == 0:
        print(f"Rank {rank}/{world_size}: Complete data check test")

def init_process(numel, dtype, iters, timeout=300, backend='nccl'):

    start_time = time.time()
    rank=int(os.environ["RANK"])
    world_size=int(os.environ["WORLD_SIZE"])
    local_rank=int(os.environ["LOCAL_RANK"])

    print(f"Rank {rank}/{world_size} (GPU {local_rank}): Start")

    dist.init_process_group(backend, timeout=timedelta(0, timeout))
    init_time = time.time() - start_time
    # Print the initialization time for each process
    print(f"Rank {rank}/{world_size} (GPU {local_rank}): init_process_group took {init_time:.6f} seconds")

    torch.cuda.set_device(local_rank)
    run(rank, world_size, numel, dtype=dtype, max_iters=iters)
    # Clean up the process group
    dist.destroy_process_group()

    if rank == 0:
        print(f"Rank {rank}/{world_size} (GPU {local_rank}): Complete")


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

    init_process(1 << args.log_size, data_types[args.dtype], args.iters, args.timeout)
