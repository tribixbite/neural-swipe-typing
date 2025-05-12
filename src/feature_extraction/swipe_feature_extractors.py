from typing import List, Iterable, Protocol

from torch import Tensor

from .nearest_key_lookup import NearestKeyLookup
from .distances_lookup import DistancesLookup


class SwipeFeatureExtractor(Protocol):
    def __call__(self, x: Tensor, y: Tensor, t: Tensor) -> List[Tensor]:
        ...


class MultiFeatureExtractor:
    """
    Extracts multiple feature tensors via a list of feature extractors it holds.
    """
    def __init__(self, extractors: List[SwipeFeatureExtractor]) -> None:
        self.extractors = extractors

    def __call__(self, x: Iterable[int], y: Iterable[int], t: Iterable[int]) -> List[Tensor]:
        aggregated_features: List[Tensor] = []
        for extractor in self.extractors:
            aggregated_features.extend(extractor(x, y, t))
        return aggregated_features


class TrajectoryFeatureExtractor:
    """
    Extracts trajectory features such as x, y, t and cordinate derivatives.
    """

    def __init__(self, 
                 include_time: bool,     # !!!! time should be normalized I guess. Or replaced with dt
                 include_velocities: bool,
                 include_accelerations: bool,
                 normalization: ???
                 ) -> None:
        pass


class NearestKeyFeatureExtractor:
    def __init__(self, nearest_key_lookup: NearestKeyLookup):
        pass


class KeyDistancesFeatureExtractor:
    """
    Extracts the distances from each point to every key on the keyboard.
    """
    def __init__(self, distances_lookup: DistancesLookup,
                 normalization: ???) -> None:
        pass


class KeyWeightsFeatureExtractor:
    pass
