import collections
import copy
import os

from datasets import load_dataset

import paddlenlp


# from paddlenlp.transformers import BertTokenizer,BertFasterTokenizer


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

        token_type_ids = [0 for _ in range(len(tokens))]
        seq_len = len(tokens)
        return tokens, seq_len, token_type_ids

    def __call__(self, text, **kwargs):
        is_batched = isinstance(text, (list, tuple)) and text and isinstance(text[0], (list, tuple))
        input_ids_list = []
        seq_len_list = []
        token_type_ids_list = []
        if is_batched:
            for line in text:
                input_ids, seq_len, token_type_ids = self.token(line)
                input_ids_list.append(input_ids)
                seq_len_list.append(seq_len)
                token_type_ids_list.append(token_type_ids)
        else:
            input_ids, seq_len, token_type_ids = self.token(text)
            input_ids_list.append(input_ids)
            seq_len_list.append(seq_len)
            token_type_ids_list.append(token_type_ids)
        return {"input_ids": input_ids_list, "seq_len": seq_len_list, "token_type_ids": token_type_ids_list}

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


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples)
    ner_tags = [[0]+ner_tag for ner_tag in examples['ner_tags']]
    tokenized_inputs["labels"] = ner_tags

    input_ids = tokenized_inputs['input_ids']
    labels = tokenized_inputs["labels"]

    nums = len(labels)
    for i in range(nums):
        tmp_label = labels[i]
        tmp_token = examples['tokens'][i]
        for j in range(len(tmp_label)):
            if tmp_label[j]!=0:
                print(tmp_token[j],end="")
                if j+1<len(tmp_label) and tmp_label[j+1]==0:
                    print()

    return tokenized_inputs


if __name__ == '__main__':
    raw_datasets = load_dataset(os.path.abspath(paddlenlp.datasets.business_license_ner.__file__))
    voc_path = "/home/richinfo/.paddlenlp/models/bert-base-uncased/bert-base-uncased-vocab.txt"
    tokenizer = GPUTokenizer(voc_path)

    train_ds = raw_datasets['train']
    label_list = train_ds.features['ner_tags'].feature.names
    label_num = len(label_list)
    no_entity_id = 0
    max_seq_length = 512

    train_ds = train_ds.select(range(len(train_ds) - 1))
    column_names = train_ds.column_names
    train_ds = train_ds.map(tokenize_and_align_labels,
                            batched=True,
                            remove_columns=column_names)

    for tokenized_inputs in train_ds:
        print(tokenized_inputs)

        # pass
