from typing import List, Dict

from .swipe_feature_extractors import (SwipeFeatureExtractor, 
                                       MultiFeatureExtractor,
                                       TrajectoryFeatureExtractor,
                                        NearestKeyFeatureExtractor,
                                        DistancesFeatureExtractor,
                                        KeyWeightsFeatureExtractor)
from .nearest_key_lookup import NearestKeyLookup
from .distances_lookup import DistancesLookup


def multi_swipe_feature_extractor_factory(
    extractor_params_list: List[dict]) -> MultiFeatureExtractor:
    """
    Arguments:
    ----------
    extractor_params_list : List[dict]
        List of dictionaries with parameters for each extractor.
        Each dictionary should contain the following keys:
        - 'type': type of the extractor (e.g. 'trajectory', 'nearest_key', etc.)
        - 'params': parameters for the extractor.
    """
    extractors = []
    for extractor_params in extractor_params_list:
        extractor_type = extractor_params['type']
        params = extractor_params['params']
        extractor = single_swipe_feature_extractor_factory(extractor_type, params)
        extractors.append(extractor)
    return MultiFeatureExtractor(extractors)


def single_swipe_feature_extractor_factory(
        type: str,
        params: dict) -> SwipeFeatureExtractor:
    """
    Arguments:
    ----------
    type : str
        Type of the extractor (e.g. 'trajectory', 'nearest_key', etc.)
    params : dict
        Parameters for the extractor.
    """
    if type == 'trajectory':
        coordinate_normalizer = get_coordinate_normalizer(params['coordinate_normalizer'])
        dt_normalizer = params.get('dt_normalizer', lambda x: x)
        velocities_normalizer = params.get('velocities_normalizer', lambda x: x)
        accelerations_normalizer = params.get('accelerations_normalizer', lambda x: x)
        return TrajectoryFeatureExtractor(
            params['include_dt'],
            params['include_velocities'],
            params['include_accelerations'],
            coordinate_normalizer,
            dt_normalizer,
            velocities_normalizer,
            accelerations_normalizer
        )

    elif type == 'nearest_key':
        get_keyboard_tokenizer = get_tokenizer(params.get('keyboard_tokenizer_name'))
        grid = get_grid(params.get('grid_name'))
        nearest_key_candidates = params.get('nearest_key_candidates')
        nearest_key_lookup = NearestKeyLookup(grid) 
        return NearestKeyFeatureExtractor(nearest_key_lookup, get_keyboard_tokenizer)
    
    elif type == 'distances':
        pass

    elif type == 'key_weights':
        pass

    else:
        raise ValueError(f"Unknown extractor type: {type}")
