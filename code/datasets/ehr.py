import os
import io
import json
import torch
import random
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset

from utils.utils import OrderedCounter

class EHR(Dataset):

    def __init__(self, data_dir, split, split_ratio=0.8, **kwargs):

        super().__init__()
        self.data_dir = data_dir
        self.split_ratio = split_ratio
        self.max_length = kwargs.get('max_length', None)

        data_file = os.path.join(data_dir, 'data.npy')
        data_len_file = os.path.join(data_dir, 'data_len.npy')

        # check whether has been splitted or not
        assert split in ["train", "valid", "test"]
        if not os.path.exists(os.path.join(data_dir, split)):
            data = np.load(data_file, allow_pickle=True)
            data_length = np.load(data_len_file, allow_pickle=True)

            if self.max_length == None:
                self.max_length = max(data_length)
            
            total_size = len(data_length)
            train_size = int(total_size * split_ratio)
            valid_size = (total_size - train_size) // 2
            test_size = total_size - train_size - valid_size

            total_idx = list(range(len(data)))
            random.shuffle(total_idx)
            train_idx = total_idx[:train_size]
            valid_idx = total_idx[train_size:train_size+valid_size]
            test_idx = total_idx[train_size+valid_size:]

            train_data = self.load_data(data, train_idx)
            valid_data = self.load_data(data, valid_idx)
            test_data = self.load_data(data, test_idx)

            # save
            train_path = os.path.join(data_dir, "train"); os.mkdir(train_path)
            np.save(os.path.join(train_path, "data.npy"), train_data)
            valid_path = os.path.join(data_dir, "valid"); os.mkdir(valid_path)
            np.save(os.path.join(valid_path, "data.npy"), valid_data)
            test_path = os.path.join(data_dir, "test"); os.mkdir(test_path)
            np.save(os.path.join(test_path, "data.npy"), test_data)

        path = os.path.join(data_dir, split)
        self.data = np.load(os.path.join(path, "data.npy").format(split), allow_pickle=True)

            
    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        
        return {
            "gender": np.asarray(self.data[idx]["gender"]).astype(float),
            "race": np.asarray(self.data[idx]["race"]).astype(float),
            "src_tempo": np.asarray(self.data[idx]["src_tempo"]).astype(float),
            "src_mask": np.asarray(self.data[idx]["src_mask"]).astype(float),
            "src_time": np.asarray(self.data[idx]["src_time"]).astype(float),
            "src_ava": np.asarray(self.data[idx]["src_ava"]).astype(float),
            "tgt_tempo": np.asarray(self.data[idx]["tgt_tempo"]).astype(float),
            "tgt_mask": np.asarray(self.data[idx]["tgt_mask"]).astype(float),
            "tgt_time": np.asarray(self.data[idx]["tgt_time"]).astype(float),
            "tgt_ava": np.asarray(self.data[idx]["tgt_ava"]).astype(float)
        }

 
    def load_data(self, data, index):
        examples = []
        for idx in index:
            ex = data[idx]
            #
            gender = ex["gender"] # scalar
            race = ex["race"] # scalar
            start_age_year = ex["start_age_year"] # [1, 2]
            mask = ex["mask"] # [time_len, 9]
            tempo = ex["tempo"] # [time_len, 9]
            time = np.floor(ex["time"])#.astype(int) # [time_en]
            # add struc into tempo
            padding = np.array([[0.0]*1])
            gender_pos = np.zeros((1, 2)); gender_pos[:, gender] = 1
            race_pos = np.zeros((1, 3)); race_pos[:, race] = 1
            scale = 0.2; shift = 0.8
            gender_pos = gender_pos * (np.random.random((1, 2)) * scale + shift)
            race_pos = race_pos * (np.random.random((1, 3)) * scale + shift)
            gender_neg = (1-gender_pos) * (np.random.random((1, 2)) * scale)
            race_neg = (1-race_pos) * (np.random.random((1, 3)) * scale)
            gender_ = gender_pos + gender_neg
            race_ = race_pos + race_neg
            extra =  np.concatenate([start_age_year, gender_, race_, start_age_year], 1) # [1, 9]
            extra_mask = np.array([[1]*9])
            # add extra info at start feature
            tempo = np.concatenate([extra, tempo], 0)
            mask = np.concatenate([extra_mask, mask], 0)
            time = np.concatenate([[time[0]], time], 0)
            # cut or padding
            padding = np.array([[0.0]*9]) # [1, 9]
            
            if tempo.shape[0] >= self.max_length+1:
                # src
                src_tempo = tempo[:self.max_length, :] 
                src_mask = mask[:self.max_length, :]
                src_time = time[:self.max_length]
                src_ava = np.ones((self.max_length, 1), dtype=int) # available time points
                # tgt
                tgt_tempo = tempo[1:self.max_length+1, :] 
                tgt_mask = mask[1:self.max_length+1, :]
                tgt_time = time[1:self.max_length+1]
                tgt_ava = np.ones((self.max_length, 1), dtype=int) # available time points

                try:
                    assert tgt_tempo.shape[0] == self.max_length
                    assert src_tempo.shape[0] == self.max_length
                except:
                    import pdb; pdb.set_trace()
            else:
                add = self.max_length - tempo.shape[0] +1
                # src
                src_tempo = np.concatenate([tempo[:-1, :]]+[padding]*add, 0)
                src_mask = np.concatenate([mask[:-1, :]]+[padding]*add, 0)
                src_time = np.concatenate([time[:-1], [0.0]*add], 0)
                src_ava = np.concatenate([np.ones((tempo.shape[0]-1, 1), dtype=int), np.zeros((add, 1), dtype=int)], 0)# available time points
                # tgt
                tgt_tempo = np.concatenate([tempo[1:, :]]+[padding]*add, 0)
                tgt_mask = np.concatenate([mask[1:, :]]+[padding]*add, 0)
                tgt_time = np.concatenate([time[1:], [0.0]*add], 0)
                tgt_ava = np.concatenate([np.ones((tempo.shape[0]-1, 1), dtype=int), np.zeros((add, 1), dtype=int)], 0)# available time points

                try:
                    assert tgt_tempo.shape[0] == self.max_length
                    assert src_tempo.shape[0] == self.max_length
                except:
                    import pdb; pdb.set_trace()
                
            
                


            # append
            example = {
                "gender": np.expand_dims(gender, 0),
                "race": np.expand_dims(race, 0),
                #
                "src_tempo": src_tempo,
                "src_mask": src_mask,
                "src_time": src_time,
                "src_ava": src_ava,
                "tgt_tempo": tgt_tempo,
                "tgt_mask": tgt_mask,
                "tgt_time": tgt_time,
                "tgt_ava": tgt_ava
            }
            examples.append(example)

        return examples

    
    '''
    @property
    def train_size(self):
        return self.train_size
    
    @property
    def valid_size(self):
        return self.valid_size
    
    @property
    def test_size(self):
        return self.test_size

    
    def shuffle(self):
        self.train_shuffle()
        self.valid_shuffle()
        self.test_shuffle()


    def train_shuffle(self):
        index = list(range(self.train_size))
        random.shuffle(index)
        self.train_data = self.train_data[index]
        self.train_mask = self.train_mask[index]
        self._train_batch_idx = 0
    
 
    def valid_shuffle(self):
        index = list(range(self.valid_size))
        random.shuffle(index)
        self.valid_data = self.valid_data[index]
        self.valid_mask = self.valid_mask[index]
        self._valid_batch_idx = 0
    
    
    def test_shuffle(self):
        index = list(range(self.test_size))
        random.shuffle(index)
        self.test_data = self.test_data[index]
        self.test_mask = self.test_mask[index]
        self._test_batch_idx = 0

    
    def next_train_batch(self, batch_size): 
        if self._train_batch_idx+batch_size > self.train_size:
            self.train_shuffle()
        batch_data = self.train_data[self._train_batch_idx: self._train_batch_idx+batch_size]
        batch_mask = self.train_mask[self._train_batch_idx: self._train_batch_idx+batch_size]
        self._train_batch_idx += batch_size
        return batch_data, batch_mask


    def next_valid_batch(self, batch_size): 
        if self._valid_batch_idx+batch_size > self.valid_size:
            self.valid_shuffle()
        batch_data = self.valid_data[self._valid_batch_idx: self._valid_batch_idx+batch_size]
        batch_mask = self.valid_mask[self._valid_batch_idx: self._valid_batch_idx+batch_size]
        self._valid_batch_idx += batch_size
        return batch_data, batch_mask


    def next_test_batch(self, batch_size): 
        if self._test_batch_idx+batch_size > self.test_size:
            self.test_shuffle()
        batch_data = self.test_data[self._test_batch_idx: self._test_batch_idx+batch_size]
        batch_mask = self.test_mask[self._test_batch_idx: self._test_batch_idx+batch_size]
        self._test_batch_idx += batch_size
        return batch_data, batch_mask
    '''