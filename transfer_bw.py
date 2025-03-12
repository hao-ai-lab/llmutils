import torch
import time
import migration_ops
import multiprocessing
import time
import os
import numpy as np

def get_gpu_memory(device=None):
    """Return GPU memory usage in GB"""
    if device is None:
        device = torch.cuda.current_device()
    return {
        'allocated': torch.cuda.memory_allocated(device) / 1024**3,
        'reserved': torch.cuda.memory_reserved(device) / 1024**3,
        'max_allocated': torch.cuda.max_memory_allocated(device) / 1024**3
    }

def print_memory_stats(prefix="", device=None):
    mem = get_gpu_memory(device)
    print(f"{prefix} GPU Memory Stats:")
    print(f"  Allocated: {mem['allocated']:.2f} GB")
    print(f"  Reserved:  {mem['reserved']:.2f} GB")
    print(f"  Max Allocated: {mem['max_allocated']:.2f} GB")
    # time.sleep(0.5)

# Define global variables for consistency
elem_count = 8 * 1024 ** 3  # Number of elements in the tensor
dtype = torch.float32
size = elem_count * torch.finfo(dtype).bits // 8  # Size in bytes

# num_layers = 8
# head_dim = 128
# num_heads = 16
# num_blocks = elem_count // (num_layers * 2 * num_heads * head_dim)
# shape = (num_layers, 2, num_blocks, num_heads, head_dim)

def sender_process(pipe):
    # Set CUDA device to 0
    device = 0
    torch.cuda.set_device(device)
    print("\n=== Sender Initial State (GPU 0) ===")
    print_memory_stats("Before allocation", device)
    
    # Create large data on GPU 0
    print(f"\nCreating tensor of size: {size / (1024*1024*1024):.2f} GB")
    
    data = torch.arange(elem_count, dtype=dtype, device='cuda')
    # data = torch.sin(data)  # Make it more interesting than just a range
    
    print("\n=== Sender After Allocation (GPU 0) ===")
    print_memory_stats("After allocation", device)
    
    # Get IPC handle for the tensor
    handle_bytes = migration_ops.get_ipc_mem_handle(data)
    
    # Send handle bytes through the pipe
    pipe.send(handle_bytes)
    
    print(f"\nSender process (PID: {os.getpid()}) created data and sent handle")
    print("Sample data:", data[:10].cpu().numpy())
    
    # Keep reference to data and keep process alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n=== Sender Final State (GPU 0) ===")
        print_memory_stats("Before shutdown", device)
        print("Sender shutting down")

def receiver_process(pipe):

    
    # Set CUDA device to 1
    device = 1
    torch.cuda.set_device(device)
    buffer = torch.zeros(elem_count, dtype=dtype, device='cuda')
    
    print("\n=== Receiver Initial State (GPU 1) ===")
    print_memory_stats("Before IPC", device)
    
    # Receive handle bytes from the pipe
    handle_bytes = pipe.recv()
    
    print(f"\nReceiver process (PID: {os.getpid()}) got handle")
    
    # Time the IPC handle opening
    start_time = time.time()
    
    # Open IPC handle and create tensor
    tensor = migration_ops.open_ipc_mem_handle(
        handle_bytes,
        [elem_count],  # tensor size
        dtype  # tensor dtype
    )
    print(f"[recv] {tensor.device = }")

    
    print("\n=== Receiver After IPC Handle Open (GPU 1) ===")
    print_memory_stats("After IPC open", device)
    
    # Time and perform some operations to verify data transfer
    torch.cuda.synchronize()  # Ensure all CUDA operations are complete
    ipc_time = time.time() - start_time

    cloned_tensors = []
    # round_num = 128
    # round_num = 512
    round_num = 4096
    chunk_size = elem_count // round_num

    is_bulk_transfer = True
    is_sequential_transfer = True
    is_using_different_streams = True

    if is_bulk_transfer:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        buffer.copy_(tensor, non_blocking=True)
        end_event.record()

        torch.cuda.synchronize()
        duration = start_event.elapsed_time(end_event)
        print("-------- Bulk Transfer --------")
        print(f"\033[32mTotal time: {duration:.2f} ms\033[0m")

        bandwidth = (size / 1024 ** 3) / (duration / 1000)  # in GB/ms
        print(f"\033[32mBandwidth: {bandwidth:.2f} GB/s\033[0m")
        pass
    

    if is_sequential_transfer:
        events = {}
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for i in range(round_num):

            iter_start_event = torch.cuda.Event(enable_timing=True)
            iter_end_event = torch.cuda.Event(enable_timing=True)

            iter_start_event.record()

            a = tensor[i * chunk_size: (i + 1) * chunk_size]
            buffer[i * chunk_size: (i + 1) * chunk_size].copy_(
                a, non_blocking=True
            )

            iter_end_event.record()
            events[i] = (iter_start_event, iter_end_event)
            
            # print_memory_stats(f"After buffer iter={i}", device)
        end_event.record()

        torch.cuda.synchronize()
        print("-------- Sequential Transfer --------")
        durations = []
        for i in range(round_num):
            duration = events[i][0].elapsed_time(events[i][1])
            durations.append(duration)
        
        total_time = start_event.elapsed_time(end_event)
        print(f"\033[32mTotal time: {total_time:.2f} ms\033[0m")
        average_time = sum(durations) / len(durations)
        print(f"Average time: {average_time:.2f} ms")
        
        bandwidth = (size / 1024 ** 3) / (total_time / 1000)  # in GB/ms
        print(f"\033[32mBandwidth: {bandwidth:.2f} GB/s\033[0m")

        print_memory_stats(f"After buffer iter={i}", device)
    
    if is_using_different_streams:
        events = {}
        streams = []

        
        start_time = time.time()
        for i in range(round_num):
            stream = torch.cuda.Stream()
            streams.append(stream)

            with torch.cuda.stream(stream):
                
                iter_start_event = torch.cuda.Event(enable_timing=True)
                iter_end_event = torch.cuda.Event(enable_timing=True)
                
                iter_start_event.record()
                a = tensor[i * chunk_size: (i + 1) * chunk_size]
                buffer[i * chunk_size: (i + 1) * chunk_size].copy_(
                    a, non_blocking=True
                )
                iter_end_event.record()
                events[i] = (iter_start_event, iter_end_event)
            
            # print_memory_stats(f"After buffer iter={i}", device)

        torch.cuda.synchronize()
        end_time = time.time()

        copied_size = chunk_size * torch.finfo(dtype).bits // 8 // (1024 ** 2)

        print("-------- Stream Transfer --------")
        durations = []
        for i in range(round_num):
            duration = events[i][0].elapsed_time(events[i][1])
            durations.append(duration)
        
        duration = end_time - start_time
        duration = duration * 1000
        print(f"\033[32mTotal time: {duration:.2f} ms\033[0m")
        average_time = sum(durations) / len(durations)
        print(f"Average time (copy {copied_size} MB): {average_time:.2f} ms")

        total_time = duration
        bandwidth = (size / 1024 ** 3) / (total_time / 1000)  # in GB/s
        print(f"\033[32mBandwidth: {bandwidth:.2f} GB/s\033[0m")

        print_memory_stats(f"After buffer iter={i}", device)
        pass
    

    # for i in range(round_num):
    #     cloned_tensors.append(tensor[i * chunk_size: (i + 1) * chunk_size].clone())
    #     print_memory_stats(f"After cloned iter={i}", device)
    
    # for i in range(round_num):
    #     p = cloned_tensors.pop()
    #     del p
    #     torch.cuda.empty_cache()
    #     print_memory_stats(f"After del iter={i}", device)

    
    
    # Verify data by computing sum (forces memory transfer)
    start_verify = time.time()
    tensor_sum = buffer.sum().item()
    torch.cuda.synchronize()
    verify_time = time.time() - start_verify
    
    print("\n=== Receiver After Data Verification (GPU 1) ===")
    print_memory_stats("After verification", device)
    
    print(f"\nPerformance Metrics:")
    print(f"IPC Handle Opening Time: {ipc_time*1000:.2f} ms")
    print(f"Data Verification Time: {verify_time*1000:.2f} ms")
    print(f"Total Time: {(ipc_time + verify_time)*1000:.2f} ms")
    print(f"\nData Verification:")
    print(f"First 10 elements:", tensor[:10].cpu().numpy())
    print(f"First 10 elements:", buffer[:10].cpu().numpy())
    print(f"Tensor sum: {tensor_sum}")
    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor size in GB: {tensor.numel() * torch.finfo(dtype).bits / 8 / (1024*1024*1024):.2f} GB")
    
    # Force garbage collection and show final memory state
    del tensor
    torch.cuda.empty_cache()
    print("\n=== Receiver Final State (GPU 1) ===")
    print_memory_stats("After cleanup", device)

def main():
    # Create a pipe for communication
    parent_conn, child_conn = multiprocessing.Pipe()
    
    # Create sender and receiver processes
    sender = multiprocessing.Process(target=sender_process, args=(parent_conn,))
    receiver = multiprocessing.Process(target=receiver_process, args=(child_conn,))
    
    # Start processes
    start_time = time.time()
    sender.start()
    time.sleep(1)  # Give sender time to initialize
    receiver.start()
    
    # Wait for receiver to finish
    receiver.join()
    total_time = time.time() - start_time
    
    print(f"\nTotal Process Time: {total_time:.2f} seconds")
    
    # Terminate sender
    sender.terminate()
    sender.join()

if __name__ == '__main__':
    main()
