import multiprocessing as mp
import datetime
import os
import time
import torch
import torch.distributed as dist
import numpy as np

def measure_bandwidth(tensor_size_mb, warmup=2, trials=3):
    # Get device based on rank
    device = torch.device(f'cuda:{dist.get_rank()}')
    torch.cuda.set_device(device)
    
    # Calculate tensor size in elements (assuming float32)
    dtype = torch.float32
    bytes_per_elem = torch.finfo(dtype).bits // 8
    elements = (tensor_size_mb * 1024 * 1024) // bytes_per_elem
    
    # Create tensor
    if dist.get_rank() == 0:
        data = torch.randn(elements, dtype=dtype, device=device)
    else:
        data = torch.zeros(elements, dtype=dtype, device=device)
        
    # Warmup rounds
    for _ in range(warmup):
        if dist.get_rank() == 0:
            dist.send(data, dst=1)
        else:
            dist.recv(data, src=0)
        torch.cuda.synchronize()
    
    # Measurement rounds
    bandwidths = []
    latencies = []
    for trial in range(trials):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        
        if dist.get_rank() == 0:
            dist.send(data, dst=1)
        else:
            dist.recv(data, src=0)
            
        end_event.record()
        torch.cuda.synchronize()
        
        duration_ms = start_event.elapsed_time(end_event)
        bandwidth = (tensor_size_mb / 1024) / (duration_ms / 1000)  # GB/s
        bandwidths.append(bandwidth)
        latencies.append(duration_ms)
        
        if dist.get_rank() == 1:
            print(f"Trial {trial + 1}: {bandwidth:.2f} GB/s, Latency: {duration_ms:.2f} ms")
    
    if dist.get_rank() == 1:
        avg_bandwidth = np.mean(bandwidths)
        avg_latency = np.mean(latencies)
        print(f"\n\033[32mAverage bandwidth: {avg_bandwidth:.2f} GB/s, Average latency: {avg_latency:.2f} ms\033[0m")



def run_process(rank, world_size):
    # Initialize process group
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(seconds=5)
    )

    # Test sizes from 4MB to 512MB (powers of 2) 
    sizes = [2**i for i in range(2, 14)]  # 4, 8, 16, ..., 512
    
    for size in sizes:
        if dist.get_rank() == 1:
            print(f"\nTesting with tensor size: {size}MB")
        measure_bandwidth(size)

    dist.destroy_process_group()


def main():
    # Initialize process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    # Spawn processes
    world_size = 2
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=run_process, args=(rank, world_size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
