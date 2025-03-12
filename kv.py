
import time
import multiprocessing as mp
import vllm
from vllm.engine.arg_utils import EngineArgs
import torch
from vllm import LLM
from vllm.config import (
    CacheConfig, ModelConfig, 
    ParallelConfig, 
    LayerBlockType
)
from vllm.utils import get_dtype_size



    
def get_kv_cache_shape(model_name):
    llm = LLM(
        model=model_name,
        enforce_eager=True,   
        load_format="dummy",
    )
    worker = llm.llm_engine.model_executor.driver_worker.worker
    kv_cache = worker.kv_cache

    shapes = []
    for x in kv_cache[0]:
        shapes.append(tuple(x.shape))
    return shapes

def main():

    import os
    
    # Get all model directories from huggingface cache
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_dirs = []
    if os.path.exists(cache_dir):
        model_dirs = [d for d in os.listdir(cache_dir) if d.startswith("models--")]
    
    # Convert directory names to model names
    model_names = []
    for dir_name in model_dirs:
        # Split on "--" and remove "models" prefix
        parts = dir_name.split("--")[1:]
        # Join with "/" to create model name format
        model_name = "/".join(parts)
        model_names.append(model_name)
    
    
    shapes = {}
    with mp.Pool(1) as pool:
        for model_name in model_names:
            start_time = time.time()
            try:
                shape = pool.apply(get_kv_cache_shape, args=(model_name,))
                print(f"{model_name}: {shape}")    
                shapes[model_name] = shape
            except Exception as e:
                print(f"Error getting kv cache shape for {model_name}: {e}")
            end_time = time.time()
            print(f"Time taken for {model_name}: {end_time - start_time} seconds")
    return shapes

if __name__ == "__main__":
    shapes =  main()
    print(shapes)



"""
{'JackFram/llama-160m': torch.Size([2, 128346, 16, 12, 64]), 'Qwen/Qwen2.5-1.5B-Instruct': torch.Size([2, 155411, 16, 2, 128]), 'Qwen/Qwen2.5-Coder-1.5B': torch.Size([2, 155411, 16, 2, 128]), 'Qwen/Qwen2.5-Coder-14B-Instruct': torch.Size([2, 13636, 16, 8, 128]), 'Qwen/Qwen2.5-Coder-7B-Instruct': torch.Size([2, 61684, 16, 4, 128]), 'Qwen/Qwen2.5-Math-7B-Instruct': torch.Size([2, 65119, 16, 4, 128]), 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B': torch.Size([2, 28224, 16, 8, 128]), 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B': torch.Size([2, 155798, 16, 2, 128]), 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B': torch.Size([2, 14439, 16, 8, 128]), 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B': torch.Size([2, 65137, 16, 4, 128]), 'facebook/opt-125m': torch.Size([2, 128512, 16, 12, 64]), 'meta-llama/Meta-Llama-3-8B': torch.Size([2, 28216, 16, 8, 128]), 'meta-llama/Meta-Llama-3.1-8B': torch.Size([2, 28240, 16, 8, 128]), 'meta-llama/Meta-Llama-3.1-8B-Instruct': torch.Size([2, 28240, 16, 8, 128]), 'microsoft/phi-4': torch.Size([2, 13695, 16, 10, 128]), 'unsloth/Meta-Llama-3.1-8B-Instruct': torch.Size([2, 28224, 16, 8, 128]), 'xxxbrem/Coder-1.5b-8k-epoch-3': torch.Size([2, 155390, 16, 2, 128])}
"""
