from typing import Dict, List, Optional, Tuple
import json
import os
import argparse

from .swipe_validity import (
    monotoniacally_increases, 
    points_not_too_far, 
    over_two_points_in_each_segment)
from raw_keyboard_utils import (
    get_label_to_key_map,
    get_kb_key_center,
    distance
)

from tqdm import tqdm



def create_dataset_without_errors(dataset_path: str,
                                  out_path: str,
                                  max_dist: int,
                                  grids: Dict[str, dict] = None,
                                  total: Optional[int] = None) -> List[Tuple[int, str]]:
    """
    Creates a version of a given dataset with invalid data filtered out.

    Arguments:
    ----------
    grids: str
        Dict with `grid names` as keys and `grids` as values.
        If grids is None
            it's supposed that the dataset is in original
            format (curves in the dataset don't have
            grid_name attribute, but have grid attribute with
            full grid information).
        Else
            Curves have grid_name attribute and don't have 
            grid attribute
    """
    if os.path.exists(out_path):
        raise ValueError(f"File {out_path} already exists!")
    
    is_inplace = (os.path.abspath(dataset_path) == os.path.abspath(out_path))
    temp_out_path = out_path + '.tmp' if is_inplace else out_path
    
    error_idxs = []
    with open(dataset_path, encoding="utf-8") as f:
        for i, line in tqdm(enumerate(f), total = total):
            line_data = json.loads(line)

            c = line_data['curve']
            x, y, t = c['x'], c['y'], c['t']
            if grids is not None:
                kb_keys = grids[c['grid_name']]['keys']
            else:
                kb_keys = c['grid']['keys']

            has_error = (not monotoniacally_increases(t) or
                         not points_not_too_far(x, y, kb_keys, max_dist) or
                         not over_two_points_in_each_segment(
                             line_data['word'], x, y, get_label_to_key_map(kb_keys))
            )

            if has_error:
                error_idxs.append((i, line))
                continue

            with open(temp_out_path, 'a', encoding="utf-8") as out_f:
                out_f.write(line)
    
    os.replace(temp_out_path, out_path)
    
    return error_idxs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--grids_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    with open(args.grids_path, "r", encoding="utf-8") as f:
        grids = json.load(f)
    
    # Calculate max_dist for `default`` grid 
    # (expecting same maxdist for `extra` grid).
    grid_name = 'default'
    label2key = get_label_to_key_map(grids[grid_name]['keys'])
    max_dist = distance(
        *get_kb_key_center(label2key['ф']['hitbox']),
        *get_kb_key_center(label2key['ц']['hitbox'])
    )

    FULL_TRAIN_DATASET_LENGTH = 6_000_000

    create_dataset_without_errors(
        dataset_path=args.dataset_path,
        out_path=args.output_path,
        max_dist=max_dist,
        grids=grids,
        total=FULL_TRAIN_DATASET_LENGTH  
    )

if __name__ == "__main__":
    main()
