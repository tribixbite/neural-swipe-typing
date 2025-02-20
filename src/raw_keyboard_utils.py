from typing import List, Tuple, Dict

def get_label_to_key_map(kb_keys: dict,
                         substitutions: dict = None) -> dict:
    """
    Arguments:
    ----------
    substitutions: dict
        Dict with keys being labels that are may be absent in grid
        and values being labels that should be used instead. 
        For example, {'ъ': 'ь', 'ё': 'е'}. If there is no 'ё' in grid,
        users swipes over 'е' instead.
    """
    if substitutions is None:
        substitutions = {'ъ': 'ь', 'ё': 'е'}

    label2key = {}
    for key in kb_keys:
        if 'label' not in key:
            continue
        # Since i don't plan editing keys, I don't do keys.copy().
        label2key[key['label']] = key
    for potentially_missing_label in substitutions.keys():
        if potentially_missing_label not in label2key:
            label2key[potentially_missing_label] = (
                label2key[substitutions[potentially_missing_label]])
    return label2key


def get_key_centers(target_word: str,
                    label_to_key: Dict[str, dict],
                    absent_chars_on_keyboard: Tuple[str] = ()) -> List[Tuple[int, int]]:
    """
    Arguments:
    ----------
    target_word: str
        The word being typed.
    label_to_key: Dict[str, dict]
        Mapping from character labels to keyboard key metadata.
    absent_chars_on_keyboard: Tuple[str]
        Characters that are not expected to be found on the keyboard.
    """
    key_centers = []
    for char in target_word:
        if char not in label_to_key:
            if char not in absent_chars_on_keyboard:
                print(f"Warning: Character '{char}' not found in label2key")
            continue
        key_centers.append(get_kb_key_center(label_to_key[char]['hitbox']))
    return key_centers



def get_kb_key_center(hitbox: Dict[str, int]) -> Tuple[float, float]:
    x = hitbox['x'] + hitbox['w'] / 2
    y = hitbox['y'] + hitbox['h'] / 2
    return x, y

def distance(x1, y1, x2, y2) -> float:
    return ((x2 - x1)**2 + (y2 - y1)**2)**0.5
