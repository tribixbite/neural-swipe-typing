import json

import torch
import torch.nn as nn

from modules.swipe_point_embedders import (NearestEmbeddingWithPos,
                                           SeparateTrajAndWEightedEmbeddingWithPos,
                                           SeparateTrajAndTrainableWeightedEmbeddingWithPos)


def swipe_point_embedder_factory(
        config: json) -> nn.Module:
    embedder_type = config.get("type", "nearest")
    if embedder_type == "nearest":
        return NearestEmbeddingWithPos(**config)
    elif embedder_type == "separate":
        return SeparateTrajAndWEightedEmbeddingWithPos(**config)
    elif embedder_type == "trainable":
        return SeparateTrajAndTrainableWeightedEmbeddingWithPos(**config)
    else:
        raise ValueError(f"Unknown embedder type: {embedder_type}")
