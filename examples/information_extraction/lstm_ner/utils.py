import collections
import copy
import os

from datasets import load_dataset


class GPUTokenizer():
    def __init__(self, vocab_path, do_lower_case=True, max_seq_len=512):
        self.vocab = self.load_vocab(vocab_path)
        self.do_lower_case = do_lower_case
        self.max_seq_len = max_seq_len

    def load_vocab(self, vocab_file):
        """Loads a vocabulary file into a dictionary."""
        vocab = collections.OrderedDict()
        index = 0
        with open(vocab_file, "r", encoding="utf-8") as reader:
            while True:
                token = reader.readline()

                if not token:
                    break
                token = token.strip()
                vocab[token] = index
                index += 1
        return vocab

    def token(self, text, **kwargs):
        tokens = [self.vocab["[CLS]"]]
        for char in text:
            if len(tokens) >= self.max_seq_len:
                raise Exception(f"sequence is too long! {text}")
            tokens.append(self.vocab.get(
                char.lower() if self.do_lower_case and len(char) == 1 else char,
                self.vocab["[UNK]"]))

        token_type_ids = [0 for _ in  range(len(tokens))]
        seq_len = len(tokens)
        return tokens,seq_len,token_type_ids

    def __call__(self, text, **kwargs):
        is_batched = isinstance(text, (list, tuple)) and text and isinstance(text[0], (list, tuple))
        input_ids_list = []
        seq_len_list = []
        token_type_ids_list = []
        if is_batched:
            for line in text:
                input_ids,seq_len,token_type_ids = self.token(line)
                input_ids_list.append(input_ids)
                seq_len_list.append(seq_len)
                token_type_ids_list.append(token_type_ids)
        else:
            input_ids,seq_len,token_type_ids = self.token(text)
            input_ids_list.append(input_ids)
            seq_len_list.append(seq_len)
            token_type_ids_list.append(token_type_ids)
        return {"input_ids":input_ids_list,"seq_len":seq_len_list,"token_type_ids":token_type_ids_list}

    def prepare_for_model(self, tokens, token_types=None, token_padding=0, max_length=1024, token_max_len=None):
        tokens = copy.deepcopy(tokens)
        if token_types is not None:
            segment_ids = copy.deepcopy(token_types)
        else:
            segment_ids = [[0] * min(len(each), max_length) for each in tokens]
        attn_masks = [[1] * min(len(each), max_length) for each in tokens]
        if token_max_len is None:
            token_max_len = max([len(each) for each in tokens])
        for token, segment_id, attn_mask in zip(tokens, segment_ids, attn_masks):
            token.extend([token_padding] * (token_max_len - len(token)))
            segment_id.extend([0] * (token_max_len - len(segment_id)))
            attn_mask.extend([0] * (token_max_len - len(attn_mask)))
        return tokens, segment_ids, attn_masks

