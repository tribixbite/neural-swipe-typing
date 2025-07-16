"""
Currently all models (except for the legacy ones) are obhects of the class EncoderDecoderTransformerLike.
This class has 4 components:
* swipe point embedder
* word token embedder
* encoder (that must have interface of nn.TransformerEncoder)
* decoder (that must have interface of nn.TransformerDecoder)

The primary difference between models is in the swipe point embedder.
"""


from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.positional_encodings import SinusoidalPositionalEncoding
from modules.swipe_point_embedders import (NearestEmbeddingWithPos, 
                                           SeparateTrajAndWEightedEmbeddingWithPos, 
                                           SeparateTrajAndNearestEmbeddingWithPos,
                                           SeparateTrajAndTrainableWeightedEmbeddingWithPos)


################################################################################
##################            Transformer Interface           ##################
################################################################################

def _get_mask(max_seq_len: int):
    """
    Returns a mask for the decoder transformer.
    """
    mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


"""
encode() and decode() methods are extremely useful in decoding algorithms
like beamsearch where we do encoding once and decoding multimple times.
Rhis reduces computations up to two times.
"""

class EncoderDecoderTransformerLike(nn.Module):
    def _get_mask(self, max_seq_len: int):
        """
        Returns a mask for the decoder transformer.
        """
        return _get_mask(max_seq_len)

    def __init__(self, 
                 enc_in_emb_model: nn.Module, 
                 dec_in_emb_model: nn.Module, 
                 encoder: nn.Module, 
                 decoder: nn.Module, 
                 out: nn.Module,
                 device: Optional[str] = None):
        super().__init__()
        self.enc_in_emb_model = enc_in_emb_model
        self.dec_in_emb_model = dec_in_emb_model
        self.encoder = encoder
        self.decoder = decoder
        self.out = out  # linear
        self.device = torch.device(
            device or 'cuda' if torch.cuda.is_available() else 'cpu')

    # x can be a tuple (ex. traj_feats, kb_tokens) or a single tensor
    # (ex. just kb_tokens).
    def encode(self, x, x_pad_mask):
        x = self.enc_in_emb_model(x)
        return self.encoder(x, src_key_padding_mask = x_pad_mask)
    
    def decode(self, y, x_encoded, memory_key_padding_mask, tgt_key_padding_mask):
        y = self.dec_in_emb_model(y)
        tgt_mask = self._get_mask(len(y)).to(device=self.device)
        dec_out = self.decoder(y, x_encoded, tgt_mask=tgt_mask, 
                               memory_key_padding_mask=memory_key_padding_mask, 
                               tgt_key_padding_mask=tgt_key_padding_mask)
        return self.out(dec_out)

    def forward(self, x, y, x_pad_mask, y_pad_mask):
        x_encoded = self.encode(x, x_pad_mask)
        return self.decode(y, x_encoded, x_pad_mask, y_pad_mask)



################################################################################
#################                Model Getters                 #################
################################################################################


def get_transformer_encoder_backbone_bigger__v3() -> nn.TransformerEncoder:
    d_model = 128
    num_encoder_layers = 4
    num_heads_encoder = 4
    dim_feedforward = 128
    dropout = 0.1
    activation = F.relu

    encoder_norm = nn.LayerNorm(d_model, eps=1e-5, bias=True)

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=num_heads_encoder,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
    )
    
    encoder = nn.TransformerEncoder(
        encoder_layer,
        num_layers=num_encoder_layers,
        norm=encoder_norm,
    )

    return encoder


def get_transformer_decoder_backbone_bigger__v3() -> nn.TransformerDecoder:
    d_model = 128
    num_decoder_layers = 4
    num_heads_decoder = 4
    dim_feedforward = 128
    dropout = 0.1
    activation = F.relu

    decoder_norm = nn.LayerNorm(d_model, eps=1e-5, bias=True)

    decoder_layer = nn.TransformerDecoderLayer(
        d_model=d_model,
        nhead=num_heads_decoder,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
    )
    
    decoder = nn.TransformerDecoder(
        decoder_layer,
        num_layers=num_decoder_layers,
        norm=decoder_norm,
    )

    return decoder
                                                


def get_word_char_embedding_model_bigger__v3(d_model: int, n_word_chars: int, 
                                             max_out_seq_len: int=35, 
                                             dropout: float=0.1,
                                             device=None) -> nn.Module:
    word_char_embedding = nn.Embedding(n_word_chars, d_model)
    dropout = 0.1
    word_char_emb_dropout = nn.Dropout(dropout)
    word_char_pos_encoder = SinusoidalPositionalEncoding(d_model, max_out_seq_len, device=device)

    word_char_embedding_model = nn.Sequential(
        word_char_embedding,
        word_char_emb_dropout,
        word_char_pos_encoder
    )

    return word_char_embedding_model



def _get_transformer_bigger__v3(input_embedding: nn.Module,
                                device = None,):
    CHAR_VOCAB_SIZE = 37  # = len(word_char_tokenizer.char_to_idx)
    MAX_OUT_SEQ_LEN = 35  # word_char_tokenizer.max_word_len - 1

    n_word_chars = CHAR_VOCAB_SIZE


    n_classes = CHAR_VOCAB_SIZE - 2  # <sos> and <pad> are not predicted


    d_model = 128

    device = torch.device(
        device
        or 'cuda' if torch.cuda.is_available() else 'cpu')


    word_char_embedding_model = get_word_char_embedding_model_bigger__v3(
        d_model, n_word_chars, max_out_seq_len=MAX_OUT_SEQ_LEN,
        dropout=0.1, device=device)


    out = nn.Linear(d_model, n_classes, device = device)


    encoder = get_transformer_encoder_backbone_bigger__v3()
    decoder = get_transformer_decoder_backbone_bigger__v3()


    return EncoderDecoderTransformerLike(
        input_embedding, word_char_embedding_model, encoder, decoder, out
    )



def _set_state(model, weights_path, device):
    if weights_path:
        model.load_state_dict(
            torch.load(weights_path, map_location = device))
    model = model.to(device)
    model = model.eval()
    return model



def get_transformer_bigger_weighted_and_traj__v3(device = None, 
                                                 weights_path = None,
                                                 n_coord_feats = 6) -> EncoderDecoderTransformerLike:
    CHAR_VOCAB_SIZE = 37  # = len(word_char_tokenizer.char_to_idx)
    MAX_CURVES_SEQ_LEN = 299
    # Actually, n_keys != n_word_chars. n_keys = 36.
    # It's legacy. It should not affect the model performace though.
    n_keys = CHAR_VOCAB_SIZE

    d_model = 128
    key_emb_size = d_model - n_coord_feats

    device = torch.device(
        device 
        or 'cuda' if torch.cuda.is_available() else 'cpu')

    input_embedding = SeparateTrajAndWEightedEmbeddingWithPos(
        n_keys=n_keys, key_emb_size=key_emb_size, 
        max_len=MAX_CURVES_SEQ_LEN, device = device, dropout=0.1)
    
    model = _get_transformer_bigger__v3(input_embedding, device)

    model = _set_state(model, weights_path, device)

    return model




def get_transformer_bigger_nearest_and_traj__v3(device = None,
                                                weights_path = None,
                                                n_coord_feats = 6) -> EncoderDecoderTransformerLike:
    device = torch.device(
        device 
        or 'cuda' if torch.cuda.is_available() else 'cpu')

    d_model = 128
    key_emb_size = d_model - n_coord_feats

    input_embedding = SeparateTrajAndNearestEmbeddingWithPos(
        n_keys=37, key_emb_size=key_emb_size, 
        max_len=299, device = device, dropout=0.1)
    
    model = _get_transformer_bigger__v3(input_embedding, device)

    model = _set_state(model, weights_path, device)

    return model





def get_transformer_bigger_nearest_only__v3(device = None,
                                            weights_path = None,
                                            n_coord_feats = 0) -> EncoderDecoderTransformerLike:
    assert n_coord_feats == 0, f"n_coord_feats is {n_coord_feats}, but should be 0"
    device = torch.device(
        device 
        or 'cuda' if torch.cuda.is_available() else 'cpu')

    d_model = 128

    input_embedding = NearestEmbeddingWithPos(
        n_elements=37, dim=d_model, max_len=299, device=device, dropout=0.1)
    
    model = _get_transformer_bigger__v3(input_embedding, device)

    model = _set_state(model, weights_path, device)

    return model



def get_transformer_bigger_trainable_gaussian_weights_and_traj__v3(
                                                                device = None,
                                                                weights_path = None,
                                                                n_coord_feats = 6,
                                                                key_centers: Optional[torch.Tensor] = None
                                                                ) -> EncoderDecoderTransformerLike:
    device = torch.device(
        device 
        or 'cuda' if torch.cuda.is_available() else 'cpu')
    
    d_model = 128
    key_emb_size = d_model - n_coord_feats

    input_embedding = SeparateTrajAndTrainableWeightedEmbeddingWithPos(
        n_keys=37, key_emb_size=key_emb_size,
        max_len=299, device=device, dropout=0.1,
        key_centers=key_centers)
    
    model = _get_transformer_bigger__v3(input_embedding, device)

    model = _set_state(model, weights_path, device)

    return model





MODEL_GETTERS_DICT = {
    "v3_weighted_and_traj_transformer_bigger": get_transformer_bigger_weighted_and_traj__v3,  # has layer norm
    "v3_nearest_and_traj_transformer_bigger": get_transformer_bigger_nearest_and_traj__v3,  # has layer norm
    "v3_nearest_only_transformer_bigger": get_transformer_bigger_nearest_only__v3,  # has layer norm
    
    "v3_trainable_gaussian_weights_and_traj_transformer_bigger": get_transformer_bigger_trainable_gaussian_weights_and_traj__v3,  # has layer norm
}
