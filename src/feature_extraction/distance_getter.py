from typing import Tuple, Set, Optional

import torch
from torch import Tensor

from grid_processing_utils import get_kb_label


def compute_pairwise_squared_distances(dots: Tensor, centers: Tensor) -> Tensor:
    """
    Arguments:
    ----------
    dots: Tensor
        Dots tensor. dots.shape = (*DOT_DIMS, 2). DOT_DIMS: tuple = (S1, S2, S3, ... SD).
    centers: Tensor
        Centers tensor. centers.shape = (K, 2). K is number of centers.

    Returns:
    --------
    Tensor
        Distance tensor. Distance tensor.shape = (*DOT_DIMS, K).
        Squared euclidean distance is used.
    
    Example:
    --------
    dots = torch.tensor([[1, 2], [3, 4], [5, 6]])
    centers = torch.tensor([[1, 2], [3, 4]])
    distance(dots, centers) -> torch.tensor([[0, 8], [8, 0], [32, 8]])
    """
    # (*DOT_DIMS, 1, 2)
    dots_exp = dots.unsqueeze(-2)
    # (K, 2) -> (1, 1, ..., 1, K, 2)
    centers_exp = centers.view(*([1] * (dots.dim() - 1)), *centers.shape) 
    return torch.sqrt(torch.pow((centers_exp - dots_exp), 2).sum(dim=-1))


class DistanceGetter:
    """
    Computes distances from coordinates to key centers.
    Handles missing keys via masking.
    """

    def __init__(self,
                 grid: dict,
                 tokenizer,
                 key_labels_of_interest: Optional[Set[str]] = None,
                 missing_distance_val: float = float('inf'),
                 device: torch.device = torch.device("cpu")):
        """
        Arguments:
        ----------
        grid: dict
        key_labels_of_interest: Optional[Set[str]]
        missing_distance_val: float
            Value to fill for distances to keys that are not present in the grid.
            Defaults to +inf.
        """
        self.grid = grid
        self.tokenizer = tokenizer
        self.device = device
        self.missing_distance_val = missing_distance_val

        self.key_labels_of_interest = key_labels_of_interest or self._get_all_key_labels()

        self.token_ids = [self.tokenizer.get_token(lbl) for lbl in self.key_labels_of_interest]
        self.centers, self.mask = self._get_centers() 

    def _get_all_key_labels(self) -> Set[str]:
        return set(get_kb_label(k) for k in self.grid['keys'])

    def _get_centers(self) -> Tuple[Tensor, Tensor]:
        """
        Returns:
        --------
        centers: 
            Tensor of shape (vocab_size, 2)
        mask: 
            BoolTensor of shape (vocab_size,) — True where center is missing
        """
        max_token_id = max(self.token_ids)
        centers = torch.empty((max_token_id + 1, 2),
                             dtype=torch.float32,
                             device=self.device)

        present_tokens = set()

        for key in self.grid['keys']:
            label = get_kb_label(key)
            token = self.tokenizer.get_token(label)
            if token in self.token_ids:
                hb = key['hitbox']
                centers[token] = torch.tensor(
                    [hb['x'] + hb['w'] / 2, hb['y'] + hb['h'] / 2],
                    device=self.device
                )
                present_tokens.add(token)

        mask = torch.ones((max_token_id + 1,), dtype=torch.bool, device=self.device)
        mask[torch.tensor(list(present_tokens), device=self.device)] = False

        return centers, mask

    def __call__(self, coords: Tensor) -> Tensor:
        """
        Arguments:
        ----------
        coords: Tensor
            Coordinates tensor of shape (N, 2)

        Returns:
        --------
        distances: Tensor
            Tensor of shape (N, K), 
            where K is the (max token id + 1) among key_labels_of_interest.
        """
        coords = coords.to(dtype=torch.float32, device=self.device)
        dists = compute_pairwise_squared_distances(coords, self.centers)  # (N, K)
        dists[:, self.mask] = self.missing_distance_val
        return dists
