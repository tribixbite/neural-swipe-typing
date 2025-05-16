from typing import Set, Optional

from torch import Tensor

from .distance_getter import DistanceGetter


class NearestKeyGetter:
    def __init__(self,
                 grid: dict,
                 tokenizer,
                 key_labels: Optional[Set[str]] = None,
                 ) -> None:
        self.distance_getter = DistanceGetter(grid, tokenizer, key_labels, 
                                              missing_distance_val=float('inf'))

    def __call__(self, coords: Tensor) -> Tensor:
        return self.distance_getter(coords).argmin(dim=1).view(-1, 1)
