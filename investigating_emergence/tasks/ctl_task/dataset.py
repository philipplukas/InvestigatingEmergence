import os
import torch
from torch.utils.data import IterableDataset
import io
from torchtext.vocab import Vocab
from torchtext.vocab import build_vocab_from_iterator
from torchtext.transforms import VocabTransform, ToTensor
from typing import List

"""
Currently only handling fixed target_len
"""
class CTLVocabBuilder(object):

    def __init__(self, vocab_file) -> None:
        
        #Load vocabulary
        self.full_path = vocab_file
        self.vocab = self.load_vocab(self.full_path)

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
    
    def get_vocab(self):
        return self.vocab


class CTLDataset(IterableDataset):

    def __init__(self, dataset_dir, split, target_len, device, eval_mode=False):

        super(CTLDataset).__init__()

        # Load dataset 
        full_path = os.path.join(dataset_dir, split + '.txt')
        full_path_mask = os.path.join(dataset_dir, split + '_' + 'mask' + '.txt')

        self.segments: List[str]
        self.masks: List[str]
        with open(full_path, 'r') as fp:
            lines = fp.readlines()
            self.segments = list(map(self.tokenize, lines))

        with open(full_path_mask, 'r') as fp:
            lines = fp.readlines()
            self.masks = list(map(self.tokenize_mask, lines))

        full_path = os.path.join(dataset_dir, 'vocab.txt')
        self.vocab = CTLVocabBuilder(full_path).get_vocab()

        self.device = device
        self.vocab_transform = VocabTransform(self.vocab)
        self.to_tensor_transform = ToTensor()
        #self.to_device_transform =  transforms.Lambda(lambda x: x.to(self.device))

        self.curr_idx = -1
        self.target_len = target_len
        self.eval_mode = eval_mode

    def get_vocab(self):
        return self.vocab
        
    def __len__(self):
       return len(self.segments)

     #Retrieve next element at idx, but also perform tokenization
     #Returns a long tensor
    def __getitem__(self, idx):
        return self.segments[idx]
    
    def tokenize(self, raw: str) -> List[str]:
        return list(raw.strip())
    
    def tokenize_mask(self, raw: str) -> List[int]:
        return list(map(int,list(raw.strip())))

    def preprocess_mask(self, tokenized: List[int]) -> torch.Tensor:

        tensor = self.to_tensor_transform(tokenized)
        device_tensor = tensor.to(device=self.device)
        return device_tensor

    def preprocess(self, tokenized: str) -> torch.Tensor:

        encoded =  self.vocab_transform(tokenized)
        tensor = self.to_tensor_transform(encoded)
        device_tensor = tensor.to(device=self.device)
        return device_tensor

    def __iter__(self):
        self.curr_idx = 0
        self.current_len = 0
        self.remaining : List[str] = []
        self.internal_generator = self.internal_next()
        return self
    
    def internal_next(self):
        remaining : List[str] = []
        remaining_mask : List[str] = []
        current_text : List[str] = []
        current_mask : List[str] = []

        while self.curr_idx < len(self):
            next_segment = self[self.curr_idx]
            if self.eval_mode:
                next_segment_mask = self.masks[self.curr_idx]

            next_segment = remaining + next_segment
            if self.eval_mode:
                next_segment_mask = remaining_mask + next_segment_mask

            len_segment = len(next_segment)

            remaining = []
            remaining_mask = []

            # Not enough, more data is neeeded for full length, don't yield yet.
            if self.current_len + len_segment < self.target_len:
                self.curr_idx += 1
                current_text += next_segment
                if self.eval_mode:
                    current_mask += next_segment_mask
                self.current_len += len_segment
                continue

            # Exactly the same length, just return and increase the state counter
            elif self.current_len + len_segment == self.target_len:
                self.curr_idx += 1
                if self.eval_mode:
                    yield self.preprocess(current_text + next_segment), self.preprocess_mask(current_mask + next_segment_mask)
                else:
                    yield self.preprocess(current_text + next_segment)

                current_text = []
                current_mask = []
                self.current_len = 0
                continue  

            # Enough data, remaining data needs to be saved for next sample
            else:
                assert self.current_len + len_segment > self.target_len

                current_text += next_segment[:self.target_len-self.current_len]
                if self.eval_mode:
                    current_mask += next_segment_mask[:self.target_len-self.current_len]

                remaining = next_segment[self.target_len-self.current_len:]
                if self.eval_mode:
                    remaining_mask = next_segment_mask[self.target_len-self.current_len:]

                assert len(current_text) == self.target_len
                if self.eval_mode:
                    yield self.preprocess(current_text), self.preprocess_mask(current_mask)
                else:
                    yield self.preprocess(current_text)
                current_text = []
                current_mask = []
                self.current_len = 0
                self.curr_idx += 1
                continue

        # Make sure that there is no remaining data
        if len(remaining) > 0:
            if self.eval_mode:
                yield self.preprocess(remaining), self.preprocess_mask(remaining_mask)
            else:
                yield self.preprocess(remaining)

        # No more data samples available
        #raise StopIteration

    """
        Returs a triple of the form (train_input, train_correct, _)
        The last element is set to -1, just to be compatible with
        the expected length of the iterator elements.
    """
    def __next__(self):

        try:
            if self.eval_mode:
                next_sample, next_sample_mask = next(self.internal_generator)
            else:
                next_sample = next(self.internal_generator)
        except StopIteration:
            #print("raising stop")
            raise StopIteration()

        # In this case, there is a following data line
        if self.curr_idx < (len(self)-1):
            train_input = next_sample

            # Simply append first element from next data line
            train_output = torch.cat([next_sample[1:],
                                    self.preprocess([self[self.curr_idx+1][0]])])
            
            if self.eval_mode:
                eval_mask = torch.cat([next_sample_mask[1:], self.preprocess_mask([self.masks[self.curr_idx+1][0]])])
            
            assert len(train_input) == len(train_output)
            assert len(train_input) == self.target_len
            if self.eval_mode:
                assert len(eval_mask) == len(train_output)

            if self.eval_mode:
                return train_input, train_output, -1, eval_mask
            
            # Return -1 instead of none, since pytorch doesn't recognize None
            else:
                return train_input, train_output, -1, -1
            
        # No following data line, this is the last one.
         
        # For now ignore such a situation
        """
        elif self.curr_idx == len(self)-1:
            padding_tensor = torch.tensor([float('-Inf')*(self.target_len-(next_sample[:-1].size()[0]))]).to(device=self.device)
            train_input = torch.cat([next_sample[:-1], padding_tensor], 0).to(device=self.device)
            train_output = torch.cat([next_sample[1:], padding_tensor], 0).to(device=self.device)

            return train_input, train_output, -1
        
        if next_sample.nelement() > 0:
            padding_tensor = torch.tensor([float('-Inf')*(self.target_len-(next_sample[:-1].size()[0]))]).to(device=self.device)
            train_input = torch.cat([next_sample[:-1], padding_tensor], 0)
            train_output = torch.cat([next_sample[1:], padding_tensor], 0)

            return train_input, train_output, -1
        
        """
        
        raise StopIteration()


