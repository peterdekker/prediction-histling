#-------------------------------------------------------------------------------
# Copyright (C) 2018 Peter Dekker
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#-------------------------------------------------------------------------------
from dataset import data
from dataset.Subset import Subset
import numpy as np  # use numpy for real random numbers
import pandas as pd
import random  # use python random lib for pseudo-random numbers



# Dataset object contains the whole dataset
class Dataset():

    def __init__(self, batch_size, matrix_x, matrix_x_unnormalized, matrix_y, mask_x, max_length_x,
                    max_length_y, matrix_x_unbounded, matrix_y_unbounded, datafile_path, datafile_ids, word_lengths):
        self.batch_size = batch_size
        self.matrix_x = matrix_x
        self.matrix_x_unnormalized = matrix_x_unnormalized
        self.matrix_y = matrix_y
        self.mask_x = mask_x
        self.max_length_x = max_length_x
        self.max_length_y = max_length_y
        self.datafile_path = datafile_path
        self.datafile_ids = np.array(datafile_ids)
        self.set_size = matrix_x.shape[0]
        
        # Needed for SeqModel
        self.matrix_x_unbounded = matrix_x_unbounded
        self.matrix_y_unbounded = matrix_y_unbounded
        self.word_lengths = word_lengths
    
    def compute_subset_sizes(self, set_size, only_train=False, only_valtest=False):
        # Set can be used for only training or val/test, then other set is used
        # for the rest.
        if only_train:
            n_train = set_size
            n_val = 0
            n_test = 0
        else:
            if only_valtest:
                n_train = 0
                n_test = set_size // 2
            else:
                # In all other cases, take +- 10% of this set as test and val batch,
                # rest as train batch
                n_test = 0.2 * set_size
                if n_test < 40:
                    n_test = 40
            
            # Validation and test sets should be a full number of batches
            batch_residue = n_test % self.batch_size
            n_test -= batch_residue
            
            n_val = n_test
            
            if not only_valtest:
                n_train = set_size - n_val - n_test
            
            if n_val % self.batch_size > 0 or n_test % self.batch_size > 0:
                raise ValueError("Validation and test set sizes must be a multiple of the batch size.")
        
        # Sanity check
        if set_size < (n_train + n_val + n_test):
            raise ValueError("Total dataset size too small to create subsets: " + str(self.set_size))
        return int(n_train), int(n_val), int(n_test)
    
    def divide_subsets(self, n_train, n_val, n_test):
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test
        print("Train/val/test sizes: " + str(self.n_train) + "|" + str(self.n_val) + "|" + str(self.n_test))
        # We need to randomize the words, because they have been sorted alphabetically
        # Use pseudo-random numbers for division of subsets,
        # so subsets are the same every time.
        random.seed(10)
        indices = list(range(0, self.set_size))
        random.shuffle(indices)
        # Take training items from front
        train_indices = indices[0:self.n_train]
        # Take validation and test items from back
        val_indices = indices[-self.n_val - self.n_test:-self.n_test]
        test_indices = indices[-self.n_test:]

        trainset = Subset(self.batch_size, self.matrix_x[train_indices],
                            self.matrix_x_unnormalized[train_indices],
                            self.matrix_y[train_indices],
                            self.mask_x[train_indices],
                            self.matrix_x_unbounded[train_indices],
                            self.matrix_y_unbounded[train_indices],
                            self.datafile_path,
                            self.datafile_ids[train_indices],
                            self.word_lengths[train_indices])

        valset = Subset(self.batch_size, self.matrix_x[val_indices],
                            self.matrix_x_unnormalized[val_indices],
                            self.matrix_y[val_indices],
                            self.mask_x[val_indices],
                            self.matrix_x_unbounded[val_indices],
                            self.matrix_y_unbounded[val_indices],
                            self.datafile_path,
                            self.datafile_ids[val_indices],
                            self.word_lengths[val_indices])
        
        testset = Subset(self.batch_size, self.matrix_x[test_indices],
                            self.matrix_x_unnormalized[test_indices],
                            self.matrix_y[test_indices],
                            self.mask_x[test_indices],
                            self.matrix_x_unbounded[test_indices],
                            self.matrix_y_unbounded[test_indices],
                            self.datafile_path,
                            self.datafile_ids[test_indices],
                            self.word_lengths[test_indices])
        
        return trainset, valset, testset
    
    def get_max_length_x(self):
        return self.max_length_x
    
    def get_max_length_y(self):
        return self.max_length_y
    
    def get_dimensions(self):
        return self.matrix_x.shape
    
    def get_size(self):
        return self.matrix_x.shape[0]
