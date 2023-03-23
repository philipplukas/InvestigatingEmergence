import os
import torch
from ..ctl_task.dataset import CTLDataset
import io
from torchtext.vocab import build_vocab_from_iterator
from torchtext.transforms import VocabTransform, ToTensor

from typing import List

"""
Currently only handling fixed target_len
"""
class EnwikDataset(CTLDataset):

    def __init__(self, dataset_dir, split, target_len, device):

        # super(EnwikDataset, self).__init__(dataset_dir, split, target_len)

        # Load dataset 
        full_path = os.path.join(dataset_dir, split + '.txt')
        with open(full_path, 'r') as fp:
            lines = fp.readlines()
        
        self.segments = [self.tokenize(line) for line in lines]

        # Only keep segments which are not empty
        self.segments = list(filter(lambda segment: len(segment) > 0, self.segments))

        # No need for vocabulary, dataset is alrady preprocessed and contains integer tokens.

        self.to_tensor_transform = ToTensor()

        self.curr_idx = -1
        self.target_len = target_len

        self.device = device

    def tokenize(self, raw: str) -> List[str]:
        return raw.strip().split()

    """ Takes string as arugment and turns it into a tensor containing numerical tokens"""
    def preprocess(self, tokenized: List[str]):

        encoded =  list(map(int, tokenized))
        #tensor = torch.tensor(tokenized).to(device=self.device)
        tensor = self.to_tensor_transform(encoded)
        return tensor.to(device=self.device)
    
    def get_vocab(self):
        return range(256)

    


