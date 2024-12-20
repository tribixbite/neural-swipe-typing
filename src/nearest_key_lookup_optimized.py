import pickle
from typing import Dict, Tuple, Iterable, List

from grid_processing_utils import get_kb_label


def _group_keys_by_y(keys: List[dict]) -> Dict[int, List[dict]]:
    y_to_keys = {}
    for key in keys:
        y = key['hitbox']['y']
        y_to_keys.setdefault(y, []).append(key)
    return y_to_keys


def clip(x, a_min, a_max):
    return min(max(x, a_min), a_max)


def get_rows(keys: List[dict], allowed_width_deviation: float
             ) -> Tuple[List[List[str]], List[int], List[float], int, int]:
    """
    Creates a keyboard grid from a list of keys and ensures it can be represented as a grid.
    A valid grid satisfies:
    - All keys within a row have approximately the same width.
    - All keys have the same height.
    - All keys within a row touch horizontally.
    - All rows touch vertically.

    Arguments:
    ----------
    keys: List[dict]
        List of keyboard keys represented as a dict with a label and a hitbox.
    allowed_width_difference: float 
        Maximum allowed difference in width relative to mean per row.

    Returns:
    --------
    Optional[Tuple]: 
        - rows (List[List[str]]): Key labels organized into rows.
        - x_offsets (List[int]): X-coordinates of the leftmost keys in each row.
        - mean_widths (List[float]): Mean widths of keys in each row.
        - key_height (int): Height of the keys.
        - keyboard_y_offset (int): Y-offset of the topmost row.
        Returns `None` if the grid constraints are not satisfied.
    """    
    y_to_keys = _group_keys_by_y(keys)
    ys = sorted(y_to_keys.keys())
    keyboard_y_offset = ys[0]
    key_dict_rows = [y_to_keys[y] for y in ys]
    
    mean_widths = [sum([key['hitbox']['w'] for key in row]) / len(row) 
                   for row in key_dict_rows]

    for i, (row, mean_width) in enumerate(zip(key_dict_rows, mean_widths)):
        for key in row:
            width = key['hitbox']['w']
            if abs(width - mean_width) > allowed_width_deviation:
                raise ValueError(
                    f"Key width deviation too high in row {i}. \n" \
                    f"Key width: {width}, mean width: {mean_width}")

    for row in key_dict_rows:
        row.sort(key=lambda key: key['hitbox']['x'])

    x_offsets = [row[0]['hitbox']['x'] for row in key_dict_rows]

    # Ensure keys within each row are touching
    for row in key_dict_rows:
        for i in range(len(row) - 1):
            if row[i]['hitbox']['x'] + row[i]['hitbox']['w'] < row[i + 1]['hitbox']['x']:
                raise ValueError("Keys are not touching in a row")

    key_heights = set(key['hitbox']['h'] for row in key_dict_rows for key in row)
    if len(key_heights) != 1:
        raise ValueError("Keys have different heights")
    key_height = key_heights.pop()

    # Ensure all rows are touching
    for i in range(len(key_dict_rows) - 1):
        cur_y = key_dict_rows[i][0]['hitbox']['y']
        next_y = key_dict_rows[i + 1][0]['hitbox']['y']
        if cur_y + key_height != next_y:
            raise ValueError(f"Rows are not touching: {cur_y + key_height} != {next_y}")
            


    rows = [[get_kb_label(key) for key in row] for row in key_dict_rows]

    return rows, x_offsets, mean_widths, key_height, keyboard_y_offset
    
    
    
class NearestKeyLookup:
    """
    Given a keyboard grid and a list of nearest_key_candidates
    returns the nearest key label for a given (x, y) coordinate.

    Stores a keyboard as a list of rows. Each row is a list of key labels (string).
    To find the nearest key label for a given (x, y) coordinate,
    we determine the row and column of the coordinate in the described structure.
    """

    def __init__(self, 
                 keys_list: dict, 
                 allowed_keys: Iterable[str],
                 allowed_width_difference: float = 1.1) -> None:
        filtered_keys = [key for key in keys_list if get_kb_label(key) in allowed_keys]
        (self.rows, 
         self.x_offsets, 
         self.mean_widths, 
         self.key_height,
         self.keyboard_y_offset) = get_rows(filtered_keys, allowed_width_difference)
    
    def __call__(self, x, y):
        return self.get_nearest_kb_label(x, y)
        
    def get_nearest_kb_label(self, x: int, y: int):
        """
        Returns the nearest key label for a given (x, y) coordinate.
        """
        row_idx_unclipped = (y - self.keyboard_y_offset) // self.key_height
        row_idx = int(clip(row_idx_unclipped, a_min = 0, a_max = len(self.rows)-1))
        row = self.rows[row_idx]

        x_offset = self.x_offsets[row_idx]
        key_width = self.mean_widths[row_idx]
        col_idx = int(clip((x - x_offset) // key_width, a_min = 0, a_max = len(row)-1))

        return row[col_idx]
        
    def _get_state_dict(self) -> dict:
        return {
            'rows': self.rows,
            'x_offsets': self.x_offsets,
            'mean_widths': self.mean_widths,
            'keyboard_y_offset': self.keyboard_y_offset,
            'key_height': self.key_height,
        }

    def save_state(self, path: str) -> None:
        """
        Saves the current state of the object to a file.
        """
        state = self._get_state_dict()
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def from_state_dict(cls, path: str):
        """
        Loads the object state from a file.
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)
        obj = cls.__new__(cls)
        obj.rows = state['rows']
        obj.x_offsets = state['x_offsets']
        obj.mean_widths = state['mean_widths']
        obj.keyboard_y_offset = state['keyboard_y_offset']
        obj.key_height = state['key_height']
        return obj
