import torch.cuda.nvtx as nvtx
from enum import Enum

class NvtxRangeType(Enum):
    COMPUTE = 'compute'
    COMMUNICATION = 'communication'
    OTHER = 'other'  # Added new range type
    FORWARD = 'forward'  # Added new range type
    BACKWARD = 'backward'  # Added new range type

nvtx_labels = {
    NvtxRangeType.COMPUTE: "COMPUTE",
    NvtxRangeType.COMMUNICATION: "COMMUNICATION",
    NvtxRangeType.OTHER: "OTHER",
    NvtxRangeType.FORWARD: "FORWARD",
    NvtxRangeType.BACKWARD: "BACKWARD"
}

def nvtx_wrapper(range_type=NvtxRangeType.COMPUTE):
    """
    A decorator that wraps a function with NVTX range for profiling.
    The function name will be used as the NVTX range name.
    
    Args:
        range_type (NvtxRangeType): The type of the NVTX range, either 'compute', 'communication', 'other', 'forward', or 'backward'.
    
    Returns:
        callable: The wrapped function with NVTX range.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            label = nvtx_labels.get(range_type, "UNKNOWN")
            nvtx.range_push(f"{label}: {func.__name__}")
            result = func(*args, **kwargs)
            nvtx.range_pop()
            return result
        return wrapper
    return decorator

