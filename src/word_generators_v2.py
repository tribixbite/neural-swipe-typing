from typing import List, Tuple, Set, Dict, Optional, Union
from abc import ABC, abstractmethod
import heapq
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import Tensor

from ns_tokenizers import CharLevelTokenizerv2
from model import EncoderDecoderTransformerLike


def _prepare_encoder_input(encoder_in: Union[Tensor, Tuple[Tensor, Tensor]], 
                           device: str, batch_first: bool
                           ) -> Tuple[Tensor, Tensor]:
    is_tensor = None
    if isinstance(encoder_in, Tensor):
        is_tensor = True
        encoder_in = [encoder_in]
    else:
        is_tensor = False

    encoder_in = [el.unsqueeze(0 if batch_first else 1) for el in encoder_in]
    encoder_in = [el.to(device) for el in encoder_in]

    return encoder_in[0] if is_tensor else encoder_in


def move_encoder_in_to_device(encoder_in: Union[Tensor, Tuple[Tensor, Tensor]], 
                              device: str) -> Tuple[Tensor, Tensor]:
    if isinstance(encoder_in, Tensor):
        return encoder_in.to(device)
    return tuple(el.to(device) for el in encoder_in)


class WordGenerator(ABC):
    def __init__(self, model: EncoderDecoderTransformerLike, 
                 tokenizer: CharLevelTokenizerv2, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.model.to(self.device)
        self.eos_token_id = tokenizer.char_to_idx['<eos>']
    
    def switch_model(self, model: EncoderDecoderTransformerLike):
        self.model = model

    @abstractmethod
    def __call__(self, xyt, kb_tokens, max_steps_n, 
                 *args, **kwargs) -> List[Tuple[float, str]]:
        pass



class WordGeneratorWithVocab(WordGenerator):
    def __init__(self, model: EncoderDecoderTransformerLike, 
                 tokenizer: CharLevelTokenizerv2, device,
                 vocab: Optional[List[str]] = None,
                 max_token_id = None) -> None:
        """
        Arguments:
        ----------
        vocab: Optional[List[str]]
            List of all possible words.
            It's used to mask out the tokens that can't follow
            generated prefix.
            If vocab is provided, max_token_id must be provided too.
        max_token_id: Optional[int]
            The maximum token id that can be generated.
            A model might never generate some tokens. For example,
            we never need to generate <pad> or <sos> tokens.
            max_token_id == n_out_neurons - 1 == n_classes - 1.
            It's supposed that if model doesn't generate some tokens,
            the unallowed tokens correspond to the last n_tokens - n_out_neurons
            tokens in the tokenizer.
        """
        if max_token_id is None and vocab is not None:
            raise ValueError(
                "If vocab is provided max_token_id must be provided too")
        
        super().__init__(model, tokenizer, device)

        self.max_token_id = max_token_id
        self.vocab = vocab
        self.prefix_to_allowed_ids = None
        if vocab is not None:
            self.prefix_to_allowed_ids = self._create_prefix_to_allowed_ids(vocab)

    
    def _create_prefix_to_allowed_ids(self, vocab: List[str]
                                      ) -> Dict[Tuple[int, ...], Set[int]]:
        # ! When switching to another type of tokenizer where tokens are not just characters
        # but can be a sequence of characters, we need to change the implementation of this method. 
        prefix_to_allowed_ids = defaultdict(set)

        for word in vocab:
            tokenized_word = self.tokenizer.encode(word)
            for i in range(1, len(tokenized_word)):
                prefix = tuple(tokenized_word[:i])
                prefix_to_allowed_ids[prefix].add(tokenized_word[i])
        return prefix_to_allowed_ids
    
    def _get_unallowed_token_ids(self, prefix_ids: List[int]) -> Set[int]:
        if self.prefix_to_allowed_ids is None:
            return set()        
        
        allowed_ids = self.prefix_to_allowed_ids[tuple(prefix_ids)]
        all_ids = set(self.tokenizer.idx_to_char.keys())
        impossible_ids = set(range(self.max_token_id + 1, len(self.tokenizer.char_to_idx)))
        unallowed_ids = all_ids - allowed_ids - impossible_ids

        # print([self.tokenizer.idx_to_char[idx] for idx in prefix_ids])
        # print([self.tokenizer.idx_to_char[idx] for idx in allowed_ids])
        # print([self.tokenizer.idx_to_char[idx] for idx in all_ids])
        # print([self.tokenizer.idx_to_char[idx] for idx in impossible_ids])
        # print([self.tokenizer.idx_to_char[idx] for idx in unallowed_ids])

        return unallowed_ids
    
    def _mask_out_unallowed_ids(self, prefix_ids: List[int], logits: Tensor
                                ) -> Tensor:
        if self.prefix_to_allowed_ids is None:
            return logits
        unallowed_ids = self._get_unallowed_token_ids(prefix_ids)
        logits[torch.tensor(list(unallowed_ids), dtype = torch.int)] = float('-inf')
        return logits



class GreedyGenerator(WordGeneratorWithVocab):
    @torch.inference_mode()
    def _generate(self, encoder_in, max_steps_n=35) -> List[Tuple[float, str]]:
        BATCH_SIZE_DIM = 1
        tokens = [self.tokenizer.char_to_idx['<sos>']]
        log_prob = 0.0
        
        encoder_in = _prepare_encoder_input(encoder_in, self.device, False)
        encoded = self.model.encode(encoder_in, None)

        for _ in range(max_steps_n):
            dec_in_char_seq = torch.tensor(tokens).unsqueeze_(BATCH_SIZE_DIM)
            next_tokens_logits: torch.Tensor = self.model.decode(
                dec_in_char_seq, encoded, None, None).squeeze_(BATCH_SIZE_DIM)[-1]
            next_tokens_logits = self._mask_out_unallowed_ids(tokens, next_tokens_logits)
            next_tokens_logproba = F.log_softmax(next_tokens_logits)
            best_next_token = int(next_tokens_logproba.argmax())
            log_prob += float(next_tokens_logproba[best_next_token])
            
            tokens.append(best_next_token)
            if best_next_token == self.eos_token_id:
                break

        return [(-log_prob, self.tokenizer.decode(tokens[1:-1]))]

    def __call__(self, encoder_in, max_steps_n=35) -> List[Tuple[float, str]]:
        return self._generate(encoder_in, max_steps_n)
    
    def generate_word_only(self, encoder_in, max_steps_n=35) -> str:
        return self._generate(encoder_in, max_steps_n)[0][1]



class BeamGenerator(WordGeneratorWithVocab):
    @torch.inference_mode()
    def __call__(self,
                 encoder_in,
                 max_steps_n=35,  # max tokens in a seq
                 return_hypotheses_n: Optional[int] = None,  # n best hypothesis to return
                 beamsize=6,  # n best solutions we store in intermidiate comuptations
                 normalization_factor=0.5,
                 ) -> List[Tuple[float, str]]:
        """
        Arguments:
        ----------
        return_hypotheses_n: Optional[int]
            return_hypotheses_n: Number of best hypotheses to return. If None,
            returns all found hypotheses.

        Returns:
        --------
        List of tuples (score, text), where:
        - score is the normalized **negative** log probability of the hypothesis
        - text is the decoded word
        The list is sorted by score (best hypotheses first) and contains
        min(return_hypotheses_n, total_hypotheses_found) items if
        return_hypotheses_n is specified, or all found hypotheses otherwise.
        """
        tokens = [self.tokenizer.char_to_idx['<sos>']]
        initial_length = len(tokens)

        # Partial hypotheses is a heap (stored as a list) of tuples.
        # Each tuple consists of a partial (unfinished aka intermidiate)
        # hypothesis and it's weight.
        # Weight is a measure of likelihood of the hypothesis.
        # [(w1, hypothesis1), (w2, hypothesis2), ...] 
        partial_hypotheses = [(0, tokens)]
        final_hypotheses = []


        encoder_in = _prepare_encoder_input(encoder_in, self.device, False)

        encoded = self.model.encode(encoder_in, None)

        while len(partial_hypotheses) > 0:
            cur_partial_score, cur_partial_hypothesis = heapq.heappop(partial_hypotheses)


            dec_in_char_seq = torch.tensor(cur_partial_hypothesis).reshape(-1, 1).to(self.device)  # (chars_seq_len, batch_size)
            # word_pad_mask = torch.zeros_like(dec_in_char_seq, dtype=torch.bool, device=self.device).transpose_(0,1)
            word_pad_mask = None
            curve_pad_mask = None

            
            next_tokens_logits = self.model.decode(
                dec_in_char_seq, encoded, curve_pad_mask, word_pad_mask).transpose_(0, 1)[0, -1]
            next_tokens_logits = self._mask_out_unallowed_ids(cur_partial_hypothesis, next_tokens_logits)
            next_tokens_logproba = F.log_softmax(next_tokens_logits)
            topk_continuations = next_tokens_logproba.topk(beamsize)

            for token_score, token_idx in zip(topk_continuations.values, topk_continuations.indices):
                # Convert tesors to loat and int to avoid memory leakage.
                token_score = float(token_score)
                token_idx = int(token_idx)

                # Skipping tokens with prob = 0 (log_prob = -inf).
                # Theese tokens apper because even if there's less 
                # then `beamsize` tokens with non-zero probs
                # topk()  will still return exactly `beamsize` tokens. 
                # There are two sourses of zero prob: 
                # 1. Model is extremely confident (maybe overconfident) 
                #    that a certain token is impossible with a given prefix.
                # 2. Masking out unallowed tokens makes their prob = 0.
                if token_score == float('-inf'):
                    continue

                # score - нормализованная разность log_softmax всех токенов.
                # Разность, а не сумма, потому что heapq - мин-куча. 
                old_denorm_score = cur_partial_score * len(cur_partial_hypothesis)**normalization_factor
                new_score = (old_denorm_score - token_score) / (len(cur_partial_hypothesis) + 1)**normalization_factor

                new_hypothesis = cur_partial_hypothesis + [token_idx]
                new_item = (new_score, new_hypothesis)

                if token_idx == self.eos_token_id or len(new_hypothesis) - initial_length >= max_steps_n:
                    final_hypotheses.append(new_item)
                else:
                    heapq.heappush(partial_hypotheses, new_item)

            if len(partial_hypotheses) > beamsize:
                partial_hypotheses = heapq.nsmallest(beamsize, partial_hypotheses)
                heapq.heapify(partial_hypotheses)

        final_scores, final_token_lists = zip(*final_hypotheses)
        final_texts = [self.tokenizer.decode(final_token_list[1:-1]) for final_token_list in final_token_lists]
        result = list(zip(final_scores, final_texts))
        result.sort()

        return result if return_hypotheses_n is None else result[:return_hypotheses_n]




# ! Note ! 
# GreedyGeneratorBatched Pros:
# * produces exactly the same results as GreedyGenerator
# * is more efficient (it's batched :) )
# GreedyGeneratorBatched Cons:
# * Doesn't support vocab masking yet
# * Has a different interface
class GreedyGeneratorBatched(WordGenerator):    
    @torch.inference_mode()
    def _generate(self, encoder_in, encoder_in_pad_mask: torch.Tensor, 
                  max_steps_n=35) -> List[Tuple[float, str]]:
        # We suppose that BATCH_FIRST is False. Note that setting `BATCH_FIRST`
        # to True won't be the only step to make the code 
        # work with a model that expects batch_first data.
        BATCH_FIRST = False 
        batch_size_dim = 0 if BATCH_FIRST else 1
        char_seq_len_dim = 1 if BATCH_FIRST else 0
        batch_size = encoder_in_pad_mask.size(0)  # mask is always batch_first
        dec_in_token_ids = torch.full((1, batch_size,), self.tokenizer.char_to_idx['<sos>'],
                                         device=self.device, dtype=torch.int32)
        sequence_finish_statuses = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        log_probs = torch.zeros(batch_size, device=self.device)
        eos_token_id = self.tokenizer.char_to_idx['<eos>']
        pad_token_id = self.tokenizer.char_to_idx['<pad>']
      
        encoder_in = move_encoder_in_to_device(encoder_in, self.device)
        encoder_in_pad_mask.to(self.device)
        encoded = self.model.encode(encoder_in, encoder_in_pad_mask)

        for _ in range(max_steps_n):
            # decoder_output.shape = char_seq_len x batch_size x n_tokens

            tgt_pad_mask = (dec_in_token_ids == pad_token_id).T

            next_tokens_logits = self.model.decode(
                dec_in_token_ids, encoded, encoder_in_pad_mask, tgt_pad_mask)[-1]  # shape = batch_size x n_tokens 
            
            # next_tokens_logits = self._mask_out_unallowed_ids(
            #     dec_in_char_seq.squeeze(BATCH_SIZE_DIM).tolist(),
            #     next_tokens_logits)

            # ! Note !  We don't really need to perform softmax since argmax would be the same.
            # It's only needed to return proper log probabilities (that can be used to output probabilities).
            next_tokens_logproba = F.log_softmax(next_tokens_logits, dim=-1) # shape = batch_size x n_tokens
            next_tokens = next_tokens_logproba.argmax(dim=-1)  # shape = batch_size
            
            # Finished sequences should only be extended by pad tokens.
            next_tokens = next_tokens * ~sequence_finish_statuses + pad_token_id * sequence_finish_statuses

            # Our model never predict <pad> and actually, pad_oken_id is out of range of model's output.
            # Thus we need to add log_probs only for unfinishedsequences
            unfinished_indices = torch.nonzero(~sequence_finish_statuses, as_tuple=True)[0]
            log_probs[unfinished_indices] += next_tokens_logproba[unfinished_indices, next_tokens[unfinished_indices]]

            dec_in_token_ids = torch.cat([dec_in_token_ids, next_tokens.unsqueeze(char_seq_len_dim)], dim=char_seq_len_dim)
          
            sequence_finish_statuses |= next_tokens == eos_token_id
            if sequence_finish_statuses.all():
                break

        return dec_in_token_ids, log_probs

    def __call__(self, encoder_in, encoder_in_pad_mask, max_steps_n=35) -> List[Tuple[float, str]]:
        return self._generate(encoder_in, encoder_in_pad_mask, max_steps_n)



GENERATOR_CTORS_DICT = {
    "greedy": GreedyGenerator,
    "beam": BeamGenerator
}
