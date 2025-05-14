import torch
from torch import Tensor
from typing import Callable


class GridLookup:
    def __init__(
        self,
        grid_width: int,
        grid_height: int,
        feature_size: int,
        value_fn: Callable[[Tensor], Tensor],
        dtype: torch.dtype = torch.float32,
    ):
        """
        grid_width, grid_height: dimensions of the keyboard grid
        feature_size: number of features per coordinate (x, y)
        value_fn: function accepting Tensor (N, 2) of (x, y) and returning (N, feature_size)
        dtype: data type of cached values
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.feature_size = feature_size
        self.value_fn = value_fn
        self.lookup_tensor = torch.empty((grid_width, grid_height, feature_size), dtype=dtype)
        self._populate_cache()

    def _populate_cache(self):
        coords = torch.stack(torch.meshgrid(
            torch.arange(self.grid_width),
            torch.arange(self.grid_height),
            indexing='ij'
        ), dim=-1).reshape(-1, 2)  # (W*H, 2)

        features = self.value_fn(coords)  # (W*H, F)
        assert features.shape == (coords.size(0), self.feature_size), \
            f"Expected ({coords.size(0)}, {self.feature_size}), got {features.shape}"

        self.lookup_tensor = features.view(self.grid_width, self.grid_height, self.feature_size)

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        x, y = x.to(dtype=torch.int32), y.to(dtype=torch.int32)

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
