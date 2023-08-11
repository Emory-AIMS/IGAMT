import os
import io
import json
import torch
import random
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset

from utils.utils import OrderedCounter

class ED(Dataset):

    def __init__(self, dataset):

        super().__init__()

        self.data = self.load_data(dataset)

            
    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        
        return {
            "src_tempo": np.asarray(self.data[idx]["src_tempo"]).astype(float),
            "src_mask": np.asarray(self.data[idx]["src_mask"]).astype(float)
        }

    def load_data(self, dataset):
        examples = []
        for idx in range(len(dataset["data"])):
            # append
            example = {
                #
                "src_tempo": dataset["data"][idx],
                "src_mask": dataset["mask"][idx]
            }
            examples.append(example)

        return examples

    