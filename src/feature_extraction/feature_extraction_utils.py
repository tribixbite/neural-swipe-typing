from typing import List

import torch
from torch import Tensor

from grid_processing_utils import get_kb_label


def get_avg_half_key_diag(grid: dict, 
                          allowed_keys: List[str]) -> float:
    hkd_list = []
    for key in grid['keys']:
        label = get_kb_label(key)
        if label not in allowed_keys:
            continue
        hitbox = key['hitbox']
        kw, kh = hitbox['w'], hitbox['h']
        half_key_diag = (kw**2 + kh**2)**0.5 / 2
        hkd_list.append(half_key_diag)
    return sum(hkd_list) / len(hkd_list)


def cast_to_dtype_with_warning(tensor_list: List[Tensor], dtype: torch.dtype
                               ) -> List[Tensor]:
    tensor_types = [tensor.dtype for tensor in tensor_list]

    if all(t == dtype for t in tensor_types):
        return tensor_list

    print(f"Warning: Casting tensors of types {tensor_types} to {dtype}.")
    tensor_list = [tensor.to(dtype=dtype) for tensor in tensor_list]
    return tensor_list
    