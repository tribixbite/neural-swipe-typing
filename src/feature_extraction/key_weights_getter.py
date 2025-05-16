from typing import Optional, Set
from collections.abc import Callable

import torch
from torch import Tensor

from .distance_getter import DistanceGetter
from grid_processing_utils import get_avg_half_key_diag

class KeyWeightsGetter:
    def __init__(self,
                 grid: dict,
                 tokenizer,
                 weights_function: Callable,
                 key_labels_of_interest: Optional[Set[str]] = None,
                 missing_value_weight: float = 0.0,
                ) -> None:
        self.distance_getter = DistanceGetter(grid, tokenizer, key_labels_of_interest)
        self.half_key_diag = get_avg_half_key_diag(grid, key_labels_of_interest)
        self.missing_value_weight = missing_value_weight
        self.weights_function = weights_function

    def __call__(self, coords: Tensor) -> Tensor:
        distances = self.distance_getter(coords)
        mask = self.distance_getter.mask
        present_distances = distances[:, ~mask]
        present_distances_scaled = present_distances / self.half_key_diag
        present_weights = self.weights_function(present_distances_scaled)
        weights = torch.full_like(distances, self.missing_value_weight)
        weights[:, ~mask] = present_weights
        return weights
