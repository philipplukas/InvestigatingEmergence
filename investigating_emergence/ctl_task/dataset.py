import os
import torch
from torch.utils.data import IterableDataset
import io
from torchtext.vocab import build_vocab_from_iterator
from torchtext.transforms import VocabTransformer, ToTensor

"""
Currently only handling fixed target_len
"""
class CTLDataset(IterableDataset):

    def __init__(self, dataset_dir, split, target_len):

        # Load dataset 
        full_path = os.path.join(dataset_dir, split + '.txt')
        with open(full_path, 'r') as fp:
            lines = fp.readlines()
            self.segments = map(str.rstrip, lines)

        # Load vocabulary
        full_path = os.path.join(dataset_dir, split + '.vocab')
        self.vocab = self.load_vocab(full_path)

        self.vocab_transform = VocabTransformer(self.vocab)
        self.to_tensor_transform = ToTensor()

        self.curr_idx = -1
        self.target_len = target_len

    def load_vocab(self, file_path):
        # generating vocab from text file
        # Taken from pytorch examples in documentation
        def yield_tokens(file_path):
            with io.open(file_path, encoding = 'utf-8') as f:
                for line in f:

                    # Each line should contain one character, 
                    # therefore we don't need to further split the line into chunks
                    yield line.strip()

        # One special character for unknown tokens.
        unk_token = "<unk>"
        vocab = build_vocab_from_iterator(yield_tokens(file_path), specials=[unk_token])
        vocab.set_default_index(vocab[unk_token])

        return vocab
        
    def __len__(self):
        return len(self.segments)

    # Retrieve next element at idx, but also perform tokenization
    # Returns a long tensor
    def __getitem__(self, idx):
        return self.segments[idx]

    def preprocess(self, raw):

        tokenized =  self.vocab_transform(raw)
        tensor = self.to_tensor_transform(tokenized)
        return tensor

    def __iter__(self):
        self.curr_idx = 0
        self.current_len = 0
        self.remaining = ""
        return self

    def internal_next(self):
        remaining = ""
        while self.curr_idx < len(self):
            next_segment = self[self.curr_idx]
            len_segment = len(next_segment)

            next_segment = remaining + next_segment
            self.remaining = ""

            # Not enough, more data is neeeded for full length, don't yield yet.
            if len_segment < self.target_len:
                self.curr_idx += 1
                current_text += next_segment
                self.current_len = len_segment
                continue

            # Exactly the same length, just return and increase the state counter
            elif len_segment == self.target_len:
                self.curr_idx += 1
                yield self.preprocess(next_segment)
                continue  

            # Enough data, remaining data needs to be saved for next sample
            else:

                self.curr_idx += 1
                current_text += next_segment[:len_segment - self.target_len]
                self.remaining = next_segment[len_segment - self.target_len:]
                yield self.preprocess(current_text)

                current_text = ""
                continue

        # Make sure that there is no remaining data
        if len(self.remaining) > 0:
            yield self.preprocess(self.remaining)

        # No more data samples available
        raise StopIteration

    """
        Returs a triple of the form (train_input, train_correct, _)
        The last element is set to -1, just to be compatible with
        the expected length of the iterator elements.
    """
    def __next__(self):

        next_sample = self.internal_next()

        # In this case, there is a following data line
        if self.curr_idx < (len(self) - 1):
            train_input = next_sample

            # Simply append first element from next data line
            train_output = next_sample[1:] + self[self.curr_idx+1][0]

            yield (train_input, train_output, -1)

        # No following data line, this is the last one.
        elif self.curr_idx == len(self) - 1:
            train_input = next_sample[:-1]
            train_output = next_sample[1:]

            yield (train_input, train_output, -1)


