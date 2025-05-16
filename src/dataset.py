import json
from collections.abc import Callable
from typing import Optional, List, Tuple, Dict
import array
from multiprocessing import Pool

import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

from feature_extraction.swipe_feature_extractors import SwipeFeatureExtractor
from ns_tokenizers import CharLevelTokenizerv2


RawDatasetEl = Tuple[array.array, array.array, 
                     array.array, str, Optional[str]]


def _get_data_from_json_line(line) -> RawDatasetEl:
    data = json.loads(line)

    X = array.array('h', data['curve']['x'])
    Y = array.array('h', data['curve']['y'])
    T = array.array('h', data['curve']['t'])

    grid_name = data['curve']['grid_name']   

    tgt_word = data['word'] if 'word' in data else None

    return X, Y, T, grid_name, tgt_word


class SwipeDataset(Dataset):
    """
    Dataset class for NeuroSwipe jsonl dataset
    """

    def __init__(self,
                 data_path: str,
                 store_gnames: bool,
                 word_tokenizer: CharLevelTokenizerv2,
                 grid_name_to_swipe_feature_extractor: Dict[str, SwipeFeatureExtractor],
                 total: Optional[int] = None):
        """
        Arguments:
        ----------
        data_path: str
            Path to the NeuroSwipe dataset in JSON format.
            A custom version of the dataset is used: "grid" property
            is replaced with "grid_name". The grid itself is stored in
            a separate gridname_to_grid dictionary.
            Dataset is a list of JSON lines. Each line is a dictionary
            with the following properties:
            - word (str): word that was typed. 
                Is abscent in test and val datasets.
            - curve (dict): dictionary that contains the following properties:
                - x (List[int]): x coordinates of the swipe trajectory.
                - y (List[int]): y coordinates of the swipe trajectory.
                - t (List[int]): time (in ms) from the beginning of the swipe.
                - grid_name (str): name of the keyboard grid.
        store_gnames: bool
            If True, stores grid names in self.grid_name_list.
        total: Optional[int]
            Number of dataset elements. Is used only for progress bar.
        """
        self.data_list = self._get_data(
            data_path, store_gnames, total=total)
        self.word_tokenizer = word_tokenizer
        self.grid_name_to_swipe_feature_extractor = grid_name_to_swipe_feature_extractor
        
    def _get_data(self,
                  data_path: str,
                  set_gnames: bool,
                  transform: Optional[Callable] = None,
                  total: Optional[int] = None) -> List[RawDatasetEl]:
        data_list = []
        if set_gnames:
            self.grid_name_list = []
        with open(data_path, "r", encoding="utf-8") as json_file:
            for line in tqdm(json_file, total = total):
                data_el = self._get_data_from_json_line(line)
                if set_gnames:
                    self.grid_name_list.append(data_el[3])
                if transform is not None:
                    data_el = transform(data_el)
                data_list.append(data_el)
        return data_list

    def _get_data_from_json_line(self,
                                 line
                                 ) -> RawDatasetEl:
        return _get_data_from_json_line(line)
    
    def _get_decoder_in_and_out(self, tgt_word: str
                                ) -> Tuple[Tensor, Tensor]:
        tgt_token_seq: List[int] = self.word_tokenizer.encode(tgt_word)
        tgt_token_seq = torch.tensor(tgt_token_seq, dtype=self.dtype)
        decoder_in = tgt_token_seq[:-1]
        decoder_out = tgt_token_seq[1:]
        return decoder_in, decoder_out
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx: int
                    ) -> Tuple[Tuple[List[Tensor], Tensor], Tensor]:
        x, y, t, grid_name, tgt_word = self.data_list[idx]
        x, y, t = map(
            lambda x: torch.tensor(x, dtype=torch.float32), 
            (x, y, t))
        swipe_feature_extractor = self.grid_name_to_swipe_feature_extractor[grid_name]
        swipe_features = swipe_feature_extractor(x, y, t)
        decoder_in, decoder_out = self._get_decoder_in_and_out(tgt_word)
        
        return ((swipe_features, decoder_in), decoder_out)
    
    @classmethod
    def from_data_list(cls, 
                       data_list: list, 
                       word_tokenizer: CharLevelTokenizerv2,
                       grid_name_to_swipe_feature_extractor: Dict[str, SwipeFeatureExtractor],
                       grid_name_list: Optional[List[str]] = None,
                       ):
        if grid_name_list:
            if len(grid_name_list) != len(data_list):
                raise ValueError(
                    f"grid_name_list length {len(grid_name_list)} " \
                    f"doesn't match data_list length {len(data_list)}")
        
        obj = cls.__new__(cls)

        obj.data_list = data_list
        obj.grid_name_to_swipe_feature_extractor = grid_name_to_swipe_feature_extractor
        obj.word_tokenizer = word_tokenizer

        if grid_name_list:
            obj.grid_name_list = grid_name_list

        return obj



class SwipeDatasetSubset:
    def __init__(self, dataset: SwipeDataset, grid_name: str):
        assert hasattr(dataset, 'grid_name_list'), \
            "Dataset doesn't have grid_name_list property. " \
            "To fix this create the dataset with store_gnames=True"
        # ! Maybe check dataset.grid_name_list is Iterable
        assert dataset.grid_name_list is not None
        assert len(dataset) == len(dataset.grid_name_list)
        
        self.dataset = dataset
        self.grid_name = grid_name
        self.grid_idxs = self._get_grid_idxs()
    
    def _get_grid_idxs(self):
        return [i for i, gname in enumerate(self.dataset.grid_name_list)
                if gname == self.grid_name]
    
    def __len__(self):
        return len(self.grid_idxs)
    
    def __getitem__(self, idx):
        return self.dataset[self.grid_idxs[idx]]



class CollateFn:
    def __init__(self, batch_first: bool, word_pad_idx: int, 
                 swipe_pad_idx: int = 0) -> None:
        self.word_pad_idx = word_pad_idx
        self.batch_first = batch_first
        self.swipe_pad_idx = swipe_pad_idx


    def __call__(self, batch: list):
        """
        Given a List where each row is 
        ((encoder_in_sample, decoder_in_sample), decoder_out_sample) 
        returns a tuple of two elements:
        1. (encoder_in, decoder_in, swipe_pad_mask, word_pad_mask)
        2. decoder_out

        Arguments:
        ----------
        batch: list of tuples:
            ((encoder_in, dec_in_char_seq), dec_out_char_seq),
            where encoder_in may be a tuple of torch tensors
            (ex. ```(traj_feats, nearest_kb_tokens)```)
            or a single tensor (ex. ```nearest_kb_tokens```)


        Returns:
        --------
        1. transformer_in: tuple of torch tensors:
            (enc_in, dec_in, swipe_pad_mask, word_pad_mask),
            where enc_in can be either a single tensor or a tuple
            of two tensors (depends on type of input)
            Each element is a torch tensor of shape:
            - enc_in: list of tuples of tensors with shapes:
                [(curve_len, batch_size, n_feats1), (curve_len, batch_size, n_feats2), ...]
            - dec_in: (chars_seq_len - 1, batch_size)
            - swipe_pad_mask: (batch_size, curve_len)
            - word_pad_mask: (batch_size, chars_seq_len - 1)
        2. dec_out: torch tensor of shape (chars_seq_len - 1, batch_size)
        """
        decoder_inputs, decoder_outputs = [], []

        num_encoder_features = len(batch[0][0])
        encoder_inputs = [[] for _ in range(num_encoder_features)]

        for row in batch:
            (enc_in, dec_in), dec_out = row

            for feature, features_list in zip(enc_in, encoder_inputs):
                features_list.append(feature)

            decoder_inputs.append(dec_in)
            decoder_outputs.append(dec_out)

        encoder_inputs_padded = [
            pad_sequence(
                encoder_in_el, batch_first=self.batch_first,
                padding_value=self.swipe_pad_idx)
            for encoder_in_el in encoder_inputs]

        decoder_inputs_padded = pad_sequence(
            decoder_inputs, batch_first=self.batch_first,
            padding_value=self.word_pad_idx)
        
        decoder_outputs_padded = pad_sequence(
            decoder_outputs, batch_first=self.batch_first,
            padding_value=self.word_pad_idx)
        

        word_pad_mask = decoder_inputs_padded == self.word_pad_idx
        if not self.batch_first:
            word_pad_mask = word_pad_mask.T  # word_pad_mask is always batch first


        encoder_in_el = encoder_inputs_padded[0]
        max_curve_len = encoder_in_el.shape[1] if self.batch_first else encoder_in_el.shape[0]
        encoder_inputs_single_feature_no_pad = encoder_inputs[0]
        encoder_lens = torch.tensor([len(x) for x in encoder_inputs_single_feature_no_pad])

        # Берем матрицу c len(encoder_lens) строками вида
        # [0, 1, ... , max_curve_len - 1].  Каждый элемент i-ой строки
        # сравниваем с длиной i-ой траектории.  Получится матрица, где True
        # только на позициях, больших, чем длина соответствующей траектории.
        # (batch_size, max_curve_len)
        swipe_pad_mask = torch.arange(max_curve_len).expand(
            len(encoder_lens), max_curve_len) >= encoder_lens.unsqueeze(1)
        

        transformer_in = (encoder_inputs_padded, 
                          decoder_inputs_padded, 
                          swipe_pad_mask, 
                          word_pad_mask)
        
        return transformer_in, decoder_outputs_padded
