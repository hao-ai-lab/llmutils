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
    kv_cache_shape = kv_cache[0][0].shape
    return kv_cache_shape

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
            try:
                shape = pool.apply(get_kv_cache_shape, args=(model_name,))
                print(f"{model_name}: {shape}")    
                shapes[model_name] = shape
            except Exception as e:
                print(f"Error getting kv cache shape for {model_name}: {e}")
    return shapes

if __name__ == "__main__":
    shapes =  main()
    print(shapes)
