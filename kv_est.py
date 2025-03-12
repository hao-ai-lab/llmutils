#
# TODO: This is not correct....
#
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

STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.half,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
    "fp8": torch.uint8,
    "fp8_e4m3": torch.uint8,
    "fp8_e5m2": torch.uint8,
}

def get_dtype_size(dtype: torch.dtype) -> int:
    """Get the size of the data type in bytes."""
    return torch.tensor([], dtype=dtype).element_size()


def get_cache_block_size(
    cache_config: CacheConfig,
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
) -> int:
    head_size = model_config.get_head_size()
    num_heads = model_config.get_num_kv_heads(parallel_config)
    num_attention_layers = model_config.get_num_layers_by_block_type(
        parallel_config, LayerBlockType.attention)

    if cache_config.cache_dtype == "auto":
        dtype = model_config.dtype
    else:
        dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

    key_cache_entry = num_heads * head_size
    # For MLA there is no value cache, since the latent vector
    # is joint keys and values.
    value_cache_entry = key_cache_entry if not model_config.use_mla else 0
    total = num_attention_layers * cache_config.block_size * \
        (key_cache_entry + value_cache_entry)

    dtype_size = get_dtype_size(dtype)
    return dtype_size * total

def get_kv_cache_shape(
    model_name: str = "facebook/opt-125m",
    num_blocks = 12800,
):
    args = EngineArgs(model=model_name)
    vllm_config = args.create_engine_config()
    model_config = vllm_config.model_config
    cache_config = vllm_config.cache_config
    parallel_config = vllm_config.parallel_config

    block_size = get_cache_block_size(cache_config, model_config, parallel_config)
    num_kv_heads = model_config.get_num_kv_heads(parallel_config)
    head_size = model_config.get_head_size()
    
    # num_blocks = ...

    return (2, num_blocks, block_size, num_kv_heads, head_size)

print(get_kv_cache_shape())
