
# coding: utf-8

# In[1]:

from PIL import Image
import os
import io
import json
import torch
import random
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from tensorflow.python.platform import gfile
from utils.utils import OrderedCounter
import glob, gzip

# ## Training images

# In[2]:

class MNIST(object):
    """
    This data class is designed for mnist dataset, which has three different kinds of data --
    train, val and test. I split training dataset (60000) into train (54000) and valid (6000)
    datasets by the split_ratio
    
    """
    def __init__(self, data_dir, image_dir="images", split_ratio=0.9, slice_w = 4, slice_h=4,
                 onehot=True, normalize=True, biased=False):
        print("Dataset here")
        self.data_dir = data_dir
        self.train_dir = self.path("train")
        self.test_dir = self.path("test")
        self.image_dir = image_dir
        self.split_ratio = split_ratio
        self.slice_w = slice_w
        self.slice_h = slice_h
        self.onehot = onehot
        self.normalize = normalize
        self.biased = biased

        train_path = os.path.join("./", "train")
        valid_path = os.path.join("./", "valid")
        test_path = os.path.join("./", "test")
        if not os.path.exists(train_path) or not os.path.exists(valid_path) or not os.path.exists(test_path):
            # load images and labels
            _file = self.path("train-images-idx3-ubyte.gz")
            with gfile.Open(_file, 'rb') as f:
                train_images = self._extract_images(f)
            _file = self.path("train-labels-idx1-ubyte.gz")
            with gfile.Open(_file, 'rb') as f:
                train_labels = self._extract_labels(f, one_hot=onehot)
            _file = self.path("t10k-images-idx3-ubyte.gz")
            with gfile.Open(_file, 'rb') as f:
                self.test_images = self._extract_images(f)
            _file = self.path("t10k-labels-idx1-ubyte.gz")
            with gfile.Open(_file, 'rb') as f:
                self.test_labels = self._extract_labels(f, one_hot=onehot)
            
            validation_size = int(len(train_images) * (1-split_ratio))
            train_idx = list(range(len(train_images)))
            random.shuffle(train_idx)
            #
            train_idx = train_idx[:len(train_images)//5]
            validation_size = validation_size // 5
            #
            val_idx = train_idx[:validation_size]
            tra_idx = train_idx[validation_size:]
            self.valid_images = train_images[val_idx]
            self.valid_labels = train_labels[val_idx]
            self.train_images = train_images[tra_idx]
            self.train_labels = train_labels[tra_idx]
        
            #
            # save
            if not os.path.exists(train_path): os.mkdir(train_path)
            np.save(os.path.join(train_path, "images.npy"), self.train_images)
            np.save(os.path.join(train_path, "labels.npy"), self.train_labels)

            if not os.path.exists(valid_path): os.mkdir(valid_path)
            np.save(os.path.join(valid_path, "images.npy"), self.valid_images)
            np.save(os.path.join(valid_path, "labels.npy"), self.valid_labels)

            if not os.path.exists(test_path): os.mkdir(test_path)
            np.save(os.path.join(test_path, "images.npy"), self.test_images)
            np.save(os.path.join(test_path, "labels.npy"), self.test_labels)
        else:
            self.train_images = np.load(os.path.join(train_path, "images.npy"), allow_pickle=True)
            self.train_labels = np.load(os.path.join(train_path, "labels.npy"), allow_pickle=True)

            self.valid_images = np.load(os.path.join(valid_path, "images.npy"), allow_pickle=True)
            self.valid_labels = np.load(os.path.join(valid_path, "labels.npy"), allow_pickle=True)

            self.test_images = np.load(os.path.join(test_path, "images.npy"), allow_pickle=True)
            self.test_labels = np.load(os.path.join(test_path, "labels.npy"), allow_pickle=True)
            
        self.shuffle() # shuffle the lists
       
    
    def path(self, *path):
        return os.path.join(self.data_dir, *path)
    
    def _read32(self, bytestream):
        dt = np.dtype(np.uint32).newbyteorder('>')
        return np.frombuffer(bytestream.read(4), dtype=dt)[0]
        
    def _extract_images(self, f):
        """Extract the images into a 4D uint8 np array [index, y, x, depth].

        Args:
            f: A file object that can be passed into a gzip reader.

        Returns:
            data: A 4D uint8 np array [index, y, x, depth].

        Raises:
            ValueError: If the bytestream does not start with 2051.

        """
        print('Extracting', f.name)
        with gzip.GzipFile(fileobj=f) as bytestream:
            magic = self._read32(bytestream)
            if magic != 2051:
                raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                            (magic, f.name))
            num_images = self._read32(bytestream)
            rows = self._read32(bytestream)
            cols = self._read32(bytestream)
            buf = bytestream.read(rows * cols * num_images)
            data = np.frombuffer(buf, dtype=np.uint8)
            data = data.reshape(num_images, rows, cols, 1)
            return data

    # One hot encoding
    def _dense_to_one_hot(self, labels_dense, num_classes):
        """Convert class labels from scalars to one-hot vectors."""
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    def _extract_labels(self, f, one_hot=False, num_classes=10):
        """Extract the labels into a 1D uint8 np array [index].

        Args:
            f: A file object that can be passed into a gzip reader.
            one_hot: Does one hot encoding for the result.
            num_classes: Number of classes for the one hot encoding.

        Returns:
            labels: a 1D uint8 np array.

        Raises:
            ValueError: If the bystream doesn't start with 2049.
        """
        print('Extracting', f.name)
        with gzip.GzipFile(fileobj=f) as bytestream:
            magic = self._read32(bytestream)
            if magic != 2049:
                raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                            (magic, f.name))
            num_items = self._read32(bytestream)
            buf = bytestream.read(num_items)
            labels = np.frombuffer(buf, dtype=np.uint8)
            if one_hot:
                return self._dense_to_one_hot(labels, num_classes)
            return labels
    

    @property
    def train_size(self): return len(self.train_labels)
    
    @property
    def valid_size(self): return len(self.valid_labels)
    
    @property
    def test_size(self): return len(self.test_labels)


    def shuffle(self): 
        self.train_shuffle()
        self.valid_shuffle()
        self.test_shuffle()
        
    def train_shuffle(self): 
        train_idx = list(range(self.train_size))
        random.shuffle(train_idx)
        self.train_images = self.train_images[train_idx]
        self.train_labels = self.train_labels[train_idx]
        self._train_batch_idx = 0
    
    def valid_shuffle(self):
        valid_idx = list(range(self.valid_size))
        random.shuffle(valid_idx)
        self.valid_images = self.valid_images[valid_idx]
        self.valid_labels = self.valid_labels[valid_idx]
        self._valid_batch_idx = 0
        
    def test_shuffle(self): 
        test_idx = list(range(self.test_size))
        random.shuffle(test_idx)
        self.test_images = self.test_images[test_idx]
        self.test_labels = self.test_labels[test_idx]
        self._test_batch_idx = 0

    
    def image_to_seq(self, images):
        batch, img_w, img_h, _ = images.shape
        slice_w = self.slice_w
        slice_h = self.slice_h
        w_len = img_w//slice_w
        h_len = img_h//slice_h
        seq_len = w_len*h_len
        fea_size = slice_w * slice_h
        slices = []
        for h_idx in range(h_len):
            for w_idx in range(w_len):
                slices.append(np.reshape(images[:, w_idx*slice_w:(w_idx+1)*slice_w, h_idx*slice_h:(h_idx+1)*slice_h], [batch, -1]))
        content = np.stack(slices, axis=1)
        padding = np.zeros((batch, 1, fea_size))
        in_ = np.concatenate([padding, content], axis=1)
        tgt = np.concatenate([content, padding], axis=1)
        return in_, tgt



    def next_train_batch(self, batch_size):
        if self._train_batch_idx+batch_size >= self.train_size:
            self.train_shuffle()
        images = self.train_images[self._train_batch_idx:self._train_batch_idx+batch_size]
        if self.normalize:
            # Images for inception classifier are un-normalized to be in [0, 255] interval.
            images = images / 255.0 
        if self.biased:
            # scale to [-1, 1]
            images = images * 2.0 - 1.0
        
        labels = self.train_labels[self._train_batch_idx:self._train_batch_idx+batch_size]
        self._train_batch_idx += batch_size
        
        example = {}
        in_, tgt = self.image_to_seq(images)
        example["src_tempo"] = in_
        example["tgt_tempo"] = tgt
        example["label"] = labels
        return example
    
    def next_valid_batch(self, batch_size):
        if self._valid_batch_idx+batch_size >= self.valid_size:
            self.valid_shuffle()
        images = self.valid_images[self._valid_batch_idx:self._valid_batch_idx+batch_size]
        if self.normalize:
            # Images for inception classifier are un-normalized to be in [0, 255] interval.
            images = images / 255.0 
        if self.biased:
            # scale to [-1, 1]
            images = images * 2.0 - 1.0
        
        labels = self.valid_labels[self._valid_batch_idx:self._valid_batch_idx+batch_size]
        self._valid_batch_idx += batch_size
        
        example = {}
        in_, tgt = self.image_to_seq(images)
        example["src_tempo"] = in_
        example["tgt_tempo"] = tgt
        example["label"] = labels
        return example
    
    def next_test_batch(self, batch_size):
        if self._test_batch_idx+batch_size >= self.test_size:
            self.test_shuffle()
        images = self.test_images[self._test_batch_idx:self._test_batch_idx+batch_size]
        if self.normalize:
            # Images for inception classifier are un-normalized to be in [0, 255] interval.
            images = images / 255.0 
        if self.biased:
            # scale to [-1, 1]
            images = images * 2.0 - 1.0
        
        labels = self.test_labels[self._test_batch_idx:self._test_batch_idx+batch_size]
        self._test_batch_idx += batch_size
        
        example = {}
        in_, tgt = self.image_to_seq(images)
        example["src_tempo"] = in_
        example["tgt_tempo"] = tgt
        example["label"] = labels
        return example

