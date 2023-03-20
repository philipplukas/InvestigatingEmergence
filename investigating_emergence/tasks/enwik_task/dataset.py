import os
import torch
from ..ctl_task.dataset import CTLDataset
import io
from torchtext.vocab import build_vocab_from_iterator
from torchtext.transforms import VocabTransform, ToTensor

"""
Currently only handling fixed target_len
"""
class EnwikDataset(CTLDataset):

    def __init__(self, dataset_dir, split, target_len):

        super(EnwikDataset).__init__(dataset_dir, split, target_len)

        # Load dataset 
        full_path = os.path.join(dataset_dir, split + '.txt')
        with open(full_path, 'r') as fp:
            lines = fp.readlines()
            self.segments = list(map(str.rstrip, lines))

        # No need for vocabulary, dataset is alrady preprocessed and contains integer tokens.

        self.to_tensor_transform = ToTensor()

        self.curr_idx = -1
        self.target_len = target_len

    """ Takes string as arugment and turns it into a tensor containing numerical tokens"""
    def preprocess(self, raw: str):

        tokenized =  map(int, raw.split())
        tensor = self.to_tensor_transform(tokenized)
        return tensor

    


