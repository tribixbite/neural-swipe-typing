from collections.abc import Callable
from typing import Optional

import torch
from torch import Tensor

_INTEGER_DTYPES = {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}

def is_integer_tensor(tensor: Tensor) -> bool:
    return tensor.dtype in _INTEGER_DTYPES

class GridLookup:
    def __init__(
        self,
        grid_width: int,
        grid_height: int,
        value_fn: Callable[[Tensor, Tensor], Tensor],
        feature_size: Optional[int] = None,
    ):
        """
        Arguments:
        ----------
        grid_width: int
            Keyboard grid width.
        grid_height: int
            Keyboard grid height.
        feature_size: int
            Number of features per coordinate (x, y).
            Defaults to value_fn([[0, 0]]).shape[1].
        value_fn: Callable[[Tensor, Tensoe], Tensor]
            Function accepting Tensor (N, 2) of (x, y)
            and returning (N, feature_size).
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.value_fn = value_fn
        self.feature_size = feature_size or self._get_feature_size()
        self.lookup_tensor = self._build_lookup_tensor()

    def _get_feature_size(self) -> int:
        sample_coords = torch.zeros((1, 2), dtype=torch.int32)
        out_shape = self.value_fn(sample_coords).shape
        return out_shape[1] if len(out_shape) == 2 else 1
    
    def _build_lookup_tensor(self) -> Tensor:
        coords = torch.stack(torch.meshgrid(
            torch.arange(self.grid_width),
            torch.arange(self.grid_height),
            indexing='ij'
        ), dim=-1).reshape(-1, 2)  # shape (W*H, 2)

        features = self.value_fn(coords)  # shape (W*H, F)
        assert features.shape == (coords.size(0), self.feature_size), \
            f"Expected ({coords.size(0)}, {self.feature_size}), got {features.shape}"

        return features.view(self.grid_width, self.grid_height, self.feature_size)

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        if not is_integer_tensor(x) or not is_integer_tensor(y):
            raise ValueError(f"x and y must be integer tensors, " \
                             "got types x: {x.dtype} and y: {y.dtype}")

        in_bounds = (x >= 0) & (x < self.grid_width) & (y >= 0) & (y < self.grid_height)
        output = torch.empty((x.size(0), self.feature_size), dtype=self.lookup_tensor.dtype)

        if in_bounds.any():
            x_ib = x[in_bounds]
            y_ib = y[in_bounds]
            output[in_bounds] = self.lookup_tensor[x_ib, y_ib]

        if (~in_bounds).any():
            coords_oob = torch.cat(x[~in_bounds], y[~in_bounds])
            output[~in_bounds] = self.value_fn(coords_oob)

        return output
