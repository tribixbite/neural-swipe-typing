SPE = Swipe Point Embedding

## Nearest keyboard key lookup

Some SPE methods utilize nearest key embeddings and require a fast lookup for the nearest keyboard key's label given a point on the keyboard. There are two implementations of a NearestKeyLookup class. `nearest_key_lookup.NearestKeyLookup` and `nearest_key_lookup_optimized.NearestKeyLookup`. 

**It's recommended to use the optimized version if the assumptions are met (see below), and the regular version otherwise.**

1. `nearest_key_lookup.NearestKeyLookup` 
    * Uses a numpy array to store key labels for all integer coordinates on the keyboard according to a grid json file. The array is of shape (keyboard_width, keyboard_height). In the given dataset it's 1080x667 = 720360 elements (chars)
    * Pros: Has no assumptions about the keyboard organization.
    * Cons: each point on the keyboard is stored in the array, making it O(1) lookup time but O(m) memory usage, where m is NOT number of keys but number of points on the keyboard.    
2. `nearest_key_lookup_optimized.NearestKeyLookup`
    * **Makes a number of assumptions**
        * the keyboard is keys are organized in rows
        * there are no gaps between the rows
        * all keys have the same height
        * all keys within a row have the same width
        * there are no gaps between the keys within a row
        * Note that this implementations allows rows to have left and right offsets
    * Stores a list of rows where each row is a 1d array of key labels (chars)
    * Algorithm:
        * Finds the row index as `row_idx = (y - keyboard_top_offset) // key_height` and the key index as `key_idx = (x - left_offsets[row_idx]) // key_widths[row_idx]`
        * returns `rows_list[row_idx][key_idx]`
    * In the given dataset is stores just 30 chars
    * Pros: O(1) lookup time and O(n) memory usage, where n is the number of keys.

> [!NOTE]
> The bounds of some keys in the two implementations of nearest_key_lookup don't match.
> Reason:
> I am convinced that in reality all "letter keys" have the same width within a row, even though in the dataset it is not the case. For example, I suppose that in the first row (I note that it has no left and right offsets) the width of each key is actually equal to `keyboard_width/keys_number = 1080/11 = 98.(18)`. In general, I believe that the real width is the arithmetic mean of all the widths of all the keys in the row. It seems like key coordinates are discretized by rounding in the Yandex dataset. So, the width of the i-th key in a row is `round(left_offset+key_width*key_index)`. Thus, the width of the first key was rounded to 98, and it turned out that the first key occupies coordinates from 0 to 97, and 98 already belongs to the second key. However, if we had not made the key coordinates in the dataset integers, the coordinate 98 would still have belonged to the first key, as well as all coordinates strictly less than 98.(18). Thus, at the boundaries of the keys, these two implementations of nearest_key_lookup give different results. **Note that if the key widths for each row are defined correctly, then `nearest_key_lookup_optimized` not only uses less memory but also gives accurate results, while the regular version of `nearest_key_lookup` (when used with Yandex dataset) has an error at the boundaries of the keys.** (It's the problem of the dataset as far as I can see)
