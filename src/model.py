"""
Currently all models (except for the legacy ones) are obhects of the class EncoderDecoderTransformerLike.
This class has 4 components:
* swipe point embedder
* word token embedder
* encoder (that must have interface of nn.TransformerEncoder)
* decoder (that must have interface of nn.TransformerDecoder)

The primary difference between models is in the swipe point embedder.
"""


from typing import Callable, Optional

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





################################################################################
#################       Legacy model classes and getters       #################
################################################################################

# Legacy models are kept to ensure Yandex Cup submission is reproducible.
# The models defined above are superior.
# This section would have been deleted if new submission creation was not broken.

class SwipeCurveTransformerEncoderv1(nn.Module):
    """
    Transformer-based Curve encoder takes in a sequence of vectors and creates a representation
    of a swipe gesture on a samrtphone keyboard.
    Each vector contains information about finger trajectory at a time step.
    It contains:
    * x coordinate
    * y coordinate
    * Optionally: t
    * Optionally: dx/dt
    * Optionally: dy/dt
    * Optionally: keyboard key that has x and y coordinates within its boundaries
    """

    def __init__(self,
                 input_size: int,
                 d_model: int,
                 dim_feedforward: int,
                 num_layers: int,
                 num_heads_first: int,
                 num_heads_other: int,
                 dropout: float = 0.1,
                 device = None):
        """
        Arguments:
        ----------
        input_size: int
            Size of input vectors.
        d_model: int
            Size of the embeddings (output vectors).
            Should be equal to char embedding size of the decoder.
        dim_feedforward: int
        num_layers: int
            Number of encoder layers including the first layer.

        """
        super().__init__()

        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.first_encoder_layer = nn.TransformerEncoderLayer(
            input_size, num_heads_first, dim_feedforward, dropout, device=device)
        self.liner = nn.Linear(input_size, d_model, device=device)  # to convert embedding to d_model size
        num_layer_after_first = num_layers - 1
        if num_layer_after_first > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model, num_heads_other, dim_feedforward, dropout, device=device)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layer_after_first)
        else:
            self.transformer_encoder = None
    

    def forward(self, x, pad_mask: torch.Tensor):
        x = self.first_encoder_layer(x, src_key_padding_mask=pad_mask)
        x = self.liner(x)
        if self.transformer_encoder is not None:
            x = self.transformer_encoder(x, src_key_padding_mask=pad_mask)
        return x



class SwipeCurveTransformerDecoderv1(nn.Module):
    """
    Decodes a swipe gesture representation into a sequence of characters.

    Uses decoder transformer with masked attention to prevent the model from cheating.
    """

    def __init__(self,
                 char_emb_size,
                 n_classes,
                 nhead,
                 num_decoder_layers,
                 dim_feedforward,
                 dropout,
                 activation = F.relu,
                 device = None):
        super().__init__()

        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.decoder_layer = nn.TransformerDecoderLayer(
            char_emb_size, nhead, dim_feedforward, dropout, activation, device = device)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_decoder_layers)
        self.out = nn.Linear(char_emb_size, n_classes, device = device)
    
    def forward(self, x, memory, tgt_mask, memory_key_padding_mask, tgt_key_padding_mask):
        x = self.transformer_decoder(x,
                                     memory,
                                     tgt_mask=tgt_mask,
                                     memory_key_padding_mask=memory_key_padding_mask,
                                     tgt_key_padding_mask=tgt_key_padding_mask)
        x = self.out(x)
        return x




class SwipeCurveTransformer(nn.Module):
    """
    Seq2seq model. Encodes a sequence of points of a
    swipe-keyboard-typing gesture into a sequence of characters.

    n_output_classes = char_vocab_size - 2 because <pad> and <sos>
    tokens are never predicted.
    """

    def _get_mask(self, max_seq_len: int):
        """
        Returns a mask for the decoder transformer.
        """
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def __init__(self,
                 n_coord_feats: int,
                 char_emb_size: int,
                 char_vocab_size: int,
                 key_emb_size: int,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 dim_feedforward: int,
                 num_heads_encoder_1: int,
                 num_heads_encoder_2: int,
                 num_heads_decoder: int,
                 dropout:float,
                 char_embedding_dropout: float,
                 key_embedding_dropout: float,
                 max_out_seq_len: int,
                 max_curves_seq_len: int,
                 activation: Callable = F.relu,
                 device: Optional[str] = None):
        super().__init__()

        self.device = torch.device(
            device 
            or 'cuda' if torch.cuda.is_available() else 'cpu')

        input_feats_size = n_coord_feats + key_emb_size

        d_model = char_emb_size

        self.char_embedding_dropout = nn.Dropout(char_embedding_dropout)
        self.key_embedding_dropout = nn.Dropout(key_embedding_dropout)
        
        self.char_embedding = nn.Embedding(char_vocab_size, char_emb_size, device=device)
        self.key_embedding = nn.Embedding(char_vocab_size, key_emb_size, device=device)

        self.encoder = SwipeCurveTransformerEncoderv1(
            input_feats_size, d_model, dim_feedforward,
            num_encoder_layers, num_heads_encoder_1,
            num_heads_encoder_2, dropout, device=device)
        
        self.char_pos_encoder = SinusoidalPositionalEncoding(
            char_emb_size, max_out_seq_len, device=device)
        
        self.key_pos_encoder = SinusoidalPositionalEncoding(
            key_emb_size, max_curves_seq_len, device=device)
        
        n_classes = char_vocab_size - 2  # <sos> and <pad> are not predicted
        self.decoder = SwipeCurveTransformerDecoderv1(
            char_emb_size, n_classes, num_heads_decoder,
            num_decoder_layers, dim_feedforward, dropout, activation, device=device)

        self.mask = self._get_mask(max_out_seq_len).to(device=device)

    # def forward_old(self, x, kb_tokens, y, x_pad_mask, y_pad_mask):
    #     # Differs from forward(): uses self.mask instead of generating it.
    #     kb_k_emb = self.key_embedding(kb_tokens)  # keyboard key
    #     kb_k_emb = self.key_embedding_dropout(kb_k_emb)
    #     kb_k_emb = self.key_pos_encoder(kb_k_emb)
    #     x = torch.cat((x, kb_k_emb), dim = -1)
    #     x = self.encoder(x, x_pad_mask)
    #     y = self.char_embedding(y)
    #     y = self.char_embedding_dropout(y)
    #     y = self.char_pos_encoder(y)
    #     y = self.decoder(y, x, self.mask, x_pad_mask, y_pad_mask)
    #     return y
    
    def encode(self, x, kb_tokens, x_pad_mask):
        kb_k_emb = self.key_embedding(kb_tokens)  # keyboard key
        kb_k_emb = self.key_embedding_dropout(kb_k_emb)
        kb_k_emb = self.key_pos_encoder(kb_k_emb)
        x = torch.cat((x, kb_k_emb), dim = -1)
        x = self.encoder(x, x_pad_mask)
        return x
    
    def decode(self, x_encoded, y, x_pad_mask, y_pad_mask):
        y = self.char_embedding(y)
        y = self.char_embedding_dropout(y)
        y = self.char_pos_encoder(y)
        mask = self._get_mask(len(y)).to(device=self.device)
        y = self.decoder(y, x_encoded, mask, x_pad_mask, y_pad_mask)
        return y

    def forward(self, x, kb_tokens, y, x_pad_mask, y_pad_mask):
        x_encoded = self.encode(x, kb_tokens, x_pad_mask)
        return self.decode(x_encoded, y, x_pad_mask, y_pad_mask)




def get_m1_model(device = None, weights_path = None):
    CHAR_VOCAB_SIZE = 37  # = len(word_char_tokenizer.char_to_idx)
    MAX_CURVES_SEQ_LEN = 299
    MAX_OUT_SEQ_LEN = 35  # word_char_tokenizer.max_word_len - 1

    model = SwipeCurveTransformer(
        n_coord_feats=6,
        char_emb_size=128,
        char_vocab_size=CHAR_VOCAB_SIZE,
        key_emb_size=54,
        num_encoder_layers=4,
        num_decoder_layers=3,
        dim_feedforward=128,
        num_heads_encoder_1=4,
        num_heads_encoder_2=4,
        num_heads_decoder=4,
        dropout=0.1,
        char_embedding_dropout=0.1,
        key_embedding_dropout=0.1,
        max_out_seq_len=MAX_OUT_SEQ_LEN,
        max_curves_seq_len=MAX_CURVES_SEQ_LEN,
    device = device)

    if weights_path:
        model.load_state_dict(
            torch.load(weights_path,
                    map_location = device))
    
    model = model.to(device)
        
    model = model.eval()

    return model


def get_m1_bigger_model(device = None, weights_path = None):
    CHAR_VOCAB_SIZE = 37  # = len(word_char_tokenizer.char_to_idx)
    MAX_CURVES_SEQ_LEN = 299
    MAX_OUT_SEQ_LEN = 35  # word_char_tokenizer.max_word_len - 1

    model = SwipeCurveTransformer(
        n_coord_feats=6,
        char_emb_size=128,
        char_vocab_size=CHAR_VOCAB_SIZE,
        key_emb_size=66,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=128,
        num_heads_encoder_1=4,
        num_heads_encoder_2=4,
        num_heads_decoder=4,
        dropout=0.1,
        char_embedding_dropout=0.1,
        key_embedding_dropout=0.1,
        max_out_seq_len=MAX_OUT_SEQ_LEN,
        max_curves_seq_len=MAX_CURVES_SEQ_LEN,
        device = device)

    if weights_path:
        model.load_state_dict(
            torch.load(weights_path,
                    map_location = device))
    
    model = model.to(device)
        
    model = model.eval()

    return model


def get_m1_smaller_model(device = None, weights_path = None):
    CHAR_VOCAB_SIZE = 37  # = len(word_char_tokenizer.char_to_idx)
    MAX_CURVES_SEQ_LEN = 299
    MAX_OUT_SEQ_LEN = 35  # word_char_tokenizer.max_word_len - 1

    model = SwipeCurveTransformer(
        n_coord_feats=6,
        char_emb_size=128,
        char_vocab_size=CHAR_VOCAB_SIZE,
        key_emb_size=54,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=128,
        num_heads_encoder_1=4,
        num_heads_encoder_2=4,
        num_heads_decoder=4,
        dropout=0.1,
        char_embedding_dropout=0.1,
        key_embedding_dropout=0.1,
        max_out_seq_len=MAX_OUT_SEQ_LEN,
        max_curves_seq_len=MAX_CURVES_SEQ_LEN,
        device = device)

    if weights_path:
        model.load_state_dict(
            torch.load(weights_path,
                    map_location = device))
    
    model = model.to(device)
        
    model = model.eval()

    return model




###############################################################################
###############################################################################





MODEL_GETTERS_DICT = {
    "v3_weighted_and_traj_transformer_bigger": get_transformer_bigger_weighted_and_traj__v3,  # has layer norm
    "v3_nearest_and_traj_transformer_bigger": get_transformer_bigger_nearest_and_traj__v3,  # has layer norm
    "v3_nearest_only_transformer_bigger": get_transformer_bigger_nearest_only__v3,  # has layer norm
    
    "v3_trainable_gaussian_weights_and_traj_transformer_bigger": get_transformer_bigger_trainable_gaussian_weights_and_traj__v3,  # has layer norm



    ########## Legacy models. They have an old interface; they don't have layer norm; They are transformer with a smalller dim of first layer
    "m1": get_m1_model,
    "m1_bigger": get_m1_bigger_model,
    "m1_smaller": get_m1_smaller_model,
}
