import os
import io
import json
import torch
import random
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset

from utils.utils import OrderedCounter

class MIMIC3(Dataset):

    def __init__(self, data_dir, split, split_ratio=0.8, **kwargs):

        super().__init__()
        self.data_dir = data_dir
        self.split_ratio = split_ratio
        self.max_length = kwargs.get('max_length', None)

        # check whether has been splitted or not
        assert split in ["train", "valid", "test"]
        for split in ["train", "valid", "test"]:
            if not os.path.exists(os.path.join(data_dir, split)):
                data_file = os.path.join(data_dir, '{}_data.npy'.format(split))
                data = np.load(data_file, allow_pickle=True).item()
                #import pdb; pdb.set_trace()
                data_length = data["X"].shape[1]
                if self.max_length == None:
                    self.max_length = data_length
                
                processed_data = self.load_data(data)
                # save
                path = os.path.join(data_dir, split); os.mkdir(path)
                np.save(os.path.join(path, "data.npy"), processed_data)

            path = os.path.join(data_dir, split)
            self.data = np.load(os.path.join(path, "data.npy"), allow_pickle=True).item()
            #import pdb; pdb.set_trace()

            
    def __len__(self):
        return self.data["label"].shape[0]


    def __getitem__(self, idx):
        
        return {
            "src_tempo": np.asarray(self.data["src_tempo"][idx]).astype(float),
            "src_mask": np.asarray(self.data["src_mask"][idx]).astype(float),
            "tgt_tempo": np.asarray(self.data["tgt_tempo"][idx]).astype(float),
            "tgt_mask": np.asarray(self.data["tgt_mask"][idx]).astype(float),
            "label": np.asarray(self.data["label"][idx]).astype(float),
        }

 
    def load_data(self, data):
        X = data["X"][:, :, 10:16] # temporal data
        y = data["y"] # label
        
        batch, length, fea_size = X.shape
        X_m = np.ones(X.shape)
        one_padding = np.ones((batch, 1, fea_size))
        pre_padding = np.random.uniform(size=(batch, 1, fea_size))*0.2 + 0.4
        if length < self.max_length:
            zero_padding = np.zeros((batch, self.max_length-length, fea_size))

            whole = np.concatenate([pre_padding, X, zero_padding], axis=1)
            whole_mask = np.concatenate([one_padding, X_m, zero_padding], axis=1)

        else:
            whole = np.concatenate([pre_padding, X[:, :self.max_length, :]], axis=1)
            whole_mask = np.concatenate([one_padding, X_m[:, :self.max_length, :]], axis=1)

        #import pdb; pdb.set_trace()
        # append
        examples = {
            "src_tempo": whole[:, :-1, :],
            "src_mask": whole_mask[:, :-1, :],
            "tgt_tempo": whole[:, 1:, :],
            "tgt_mask": whole_mask[:, 1:, :],
            "label": y
        }

        return examples
