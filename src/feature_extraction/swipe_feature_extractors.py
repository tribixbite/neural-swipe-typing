from typing import List, Tuple, Protocol
from collections.abc import Callable

import torch
from torch import Tensor

from .nearest_key_lookup import NearestKeyLookup
from .distances_lookup import DistancesLookup
from ns_tokenizers import KeyboardTokenizerv1


class SwipeFeatureExtractor(Protocol):
    def __call__(self, x: Tensor, y: Tensor, t: Tensor) -> List[Tensor]:
        ...


class MultiFeatureExtractor:
    """
    Extracts multiple feature tensors via a list of feature extractors it holds.
    """
    def __init__(self, extractors: List[SwipeFeatureExtractor]) -> None:
        self.extractors = extractors

    def __call__(self, x: Tensor, y: Tensor, t: Tensor) -> List[Tensor]:
        aggregated_features: List[Tensor] = []
        for extractor in self.extractors:
            aggregated_features.extend(extractor(x, y, t))
        return aggregated_features


class TrajectoryFeatureExtractor:
    """
    Extracts trajectory features such as x, y, dt and coordinate derivatives.
    """
    def __init__(self,
                 include_dt: bool,
                 include_velocities: bool,
                 include_accelerations: bool,
                 coordinate_normalizer: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]],
                 dt_normalizer: Callable[[Tensor], Tensor] = lambda x: x,
                 velocities_normalizer: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]] = lambda x: x,
                 accelerations_normalizer: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]] = lambda x: x,
                 ) -> None:
        """
        Arguments:
        ----------
        include_dt: bool
            If True, includes time since prev point (dt) as a feature.
        include_velocities : bool
            If True, includes dx/dt and dy/dt as features.
        include_accelerations : bool
            If True, includes d²x/dt² and d²y/dt² as features.
        coordinate_normalizer : Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]
            Callable to normalize x and y coordinates.
        dt_normalizer : Callable[[Tensor], Tensor], optional
            Callable to normalize dt. Defaults to identity function.
        velocities_normalizer : Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]], optional
            Callable to normalize dx/dt and dy/dt. Defaults to identity function.
        accelerations_normalizer : Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]], optional
            Callable to normalize d²x/dt² and d²y/dt². Defaults to identity function.
        """
        self.include_dt = include_dt
        self.include_velocities = include_velocities
        self.include_accelerations = include_accelerations
        self.coordinate_normalizer = coordinate_normalizer
        self.dt_normalizer = dt_normalizer
        self.velocities_normalizer = velocities_normalizer
        self.accelerations_normalizer = accelerations_normalizer

    def _get_central_difference_derivative(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Calculates dx/dt for a list of x coordinates and a list of t coordinates.

        Arguments:
        ----------
        X : Tensor
            x (position) coordinates.
        T : Tensor
            T[i] = time (ms) from swipe start corresponding to X[i].

        Example:
        --------
        x0 x1 x2 x3
        t0 t1 t2 t3
        dx_dt = [0, (x2 - x0)/(t2 - t0), (x3 - x1)/(t3 - t1), 0]
        """
        dx_dt = torch.zeros_like(x)
        dx_dt[1:len(x)-1] = (x[2:len(x)] - x[:len(x)-2]) / (t[2:len(x)] - t[:len(x)-2])
        return dx_dt

    def __call__(self, x: Tensor, y: Tensor, t: Tensor) -> List[Tensor]:
        """
        Returns:
        --------
        List[Tensor]: 
            A list containing a single tensor of shape (seq_len, num_features)
            containing the trajectory features.
        """
        x_norm, y_norm = self.coordinate_normalizer(x, y)

        traj_feats_lst = [x_norm, y_norm]

        if self.include_dt:
            dt_from_prev  = torch.zeros_like(t)
            dt_from_prev [1:] = self.dt_normalizer(t[1:] - t[:-1])
            traj_feats_lst.append(dt_from_prev)

        is_velocities_needed = (self.include_velocities or self.include_accelerations)

        if is_velocities_needed:
            dx_dt = self._get_central_difference_derivative(x, t)
            dy_dt = self._get_central_difference_derivative(y, t)

        if self.include_velocities:
            traj_feats_lst.extend(self.velocities_normalizer(dx_dt, dy_dt))

        if self.include_accelerations:
            d2x_dt2 = self._get_central_difference_derivative(dx_dt, t)
            d2y_dt2 = self._get_central_difference_derivative(dy_dt, t)
            traj_feats_lst.extend(self.accelerations_normalizer(d2x_dt2, d2y_dt2))
        
        traj_feats = torch.cat(
            [feat.reshape(-1, 1) for feat in traj_feats_lst],
            dim=1
        )

        return [traj_feats]


class NearestKeyFeatureExtractor:
    def __init__(self, nearest_key_lookup: NearestKeyLookup, 
                 keyboard_tokenizer: KeyboardTokenizerv1) -> None:
        self.nearest_key_lookup = nearest_key_lookup
        self.keyboard_tokenizer = keyboard_tokenizer

    def __call__(self, x: Tensor, y: Tensor, t: Tensor) -> List[Tensor]:
        kb_labels = [self.nearest_key_lookup(int(x_el), int(y_el)) 
                        for x_el, y_el in zip(x, y)]
        kb_tokens = [self.keyboard_tokenizer.get_token(label) for label in kb_labels]
        return [torch.tensor(kb_tokens, dtype=torch.int32)]


class KeyDistancesFeatureExtractor:
    """
    Extracts the distances from each point to every key on the keyboard.
    """
    def __init__(self, distances_lookup: DistancesLookup) -> None:
        self.distances_lookup = distances_lookup

    def __call__(self, x: Tensor, y: Tensor, t: Tensor) -> List[Tensor]:
        distances = self.distances_lookup.get_distances_for_full_swipe_using_map(x, y)
        distances = torch.from_numpy(distances).to(dtype=torch.float32)
        return [distances]


class KeyWeightsFeatureExtractor:
    def __init__(self, 
                 distances_lookup: DistancesLookup,
                 half_key_diag: float,
                 weights_function: Callable,) -> None:
        """
        Arguments:
        ----------
        distances_lookup : DistancesLookup
            An instance of DistancesLookup to get distances to all keys for each point.
        half_key_diag : float
            Half of the diagonal of the key hitbox.
        weights_function : Callable
            A function to calculate weights based on distances.
        """
        self.distances_lookup = distances_lookup
        self.half_key_diag = half_key_diag
        self.weights_function = weights_function
        
    def __call__(self, x: Tensor, y: Tensor, t: Tensor) -> List[Tensor]:
        distances = self.distances_lookup.get_distances_for_full_swipe_using_map(x, y)
        distances = torch.from_numpy(distances).to(dtype=torch.float32)
        mask = (distances < 0)
        distances.masked_fill_(mask=mask, value = float('inf'))
        distances_scaled = distances / self.half_key_diag
        weights = self.weights_function(distances_scaled)
        weights.masked_fill_(mask=mask, value=0)
        return [weights]
    