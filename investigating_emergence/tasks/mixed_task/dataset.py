import random
import torch
from torch.utils.data import IterableDataset, DataLoader

from itertools import cycle



"""
Currently only handling fixed target_len
"""
class MixedDataset(IterableDataset):

    def __init__(self, data_task1: IterableDataset, data_task2: IterableDataset, batch_size: int, task_ratio: float, device: torch.device):

        super(MixedDataset).__init__()

        self.vocab = data_task1.vocab
        self.data_laoder1 = iter(cycle(DataLoader(data_task1, batch_size)))
        self.data_laoder2 = iter(cycle(DataLoader(data_task2, batch_size)))
        self.task_ratio = task_ratio
        self.device = device

        self.total_batches  = 0
        self.ctl_batches = 0
        self.enwik_batches = 0

    def get_vocab(self):
        return self.vocab

    def __iter__(self):
        return self
    
    def __next__(self):
        if random.random() < self.task_ratio:
            try:
                self.total_batches += 1
                self.ctl_batches += 1 
                return next(self.data_laoder1)
            except StopIteration:
                # In this case try to get data from the other data stream
                pass

        self.total_batches += 1
        self.enwik_batches += 1
        
        in_tensor, out_tensor, _, _ = next(self.data_laoder2)
        mask = torch.zeros(out_tensor.shape, device=self.device)
        return in_tensor, out_tensor, -1, mask
        