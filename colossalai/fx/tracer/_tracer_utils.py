from typing import Any, List, Union

import torch

from ..proxy import ColoAttribute, ColoProxy
from .meta_patch import meta_patched_function, meta_patched_module

__all__ = ['is_element_in_list', 'extract_meta']


def is_element_in_list(elements: Union[List[Any], Any], list_: List[Any]):
    """
    检查一个或多个元素是否在给定的列表中。

    Args:
        elements (Union[List[Any], Any]): 要检查的元素，可以是单个元素或元素的列表、元组或集合。
        list_ (List[Any]): 要检查的目标列表。

    Returns:
        Tuple[bool, Any]: 如果所有元素都在列表中，返回 (True, None)；
                          如果有任一元素不在列表中，返回 (False, 不在列表中的第一个元素)。
    """
    if isinstance(elements, (tuple, list, set)):
        for ele in elements:
            if ele not in list_:
                return False, ele
    else:
        if elements not in list_:
            return False, elements

    return True, None


def extract_meta(*args, **kwargs):

    def _convert(val):
        if isinstance(val, ColoProxy):
            return val.meta_data
        elif isinstance(val, (list, tuple)):
            return type(val)([_convert(ele) for ele in val])

        return val

    new_args = [_convert(val) for val in args]
    new_kwargs = {k: _convert(v) for k, v in kwargs.items()}
    return new_args, new_kwargs


def compute_meta_data_for_functions_proxy(target, args, kwargs):
    args_metas, kwargs_metas = extract_meta(*args, **kwargs)

    # fetch patched function
    if meta_patched_function.has(target):
        meta_target = meta_patched_function.get(target)
    elif meta_patched_function.has(target.__name__):
        meta_target = meta_patched_function.get(target.__name__)
    else:
        meta_target = target
    meta_out = meta_target(*args_metas, **kwargs_metas)
    if isinstance(meta_out, torch.Tensor):
        meta_out = meta_out.to(device="meta")

    return meta_out
