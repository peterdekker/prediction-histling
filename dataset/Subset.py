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
import numpy as np  # use numpy for real random numbers
import pandas as pd
import random  # use python random lib for pseudo-random numbers


# Subset object contains one portion of the dataset: train, validation or test
class Subset():

    def __init__(self, batch_size, matrix_x, matrix_x_unnormalized, matrix_y, mask_x, matrix_x_unbounded, matrix_y_unbounded, datafile_path, datafile_ids, word_lengths):
        self.batch_size = batch_size
        self.matrix_x = matrix_x
        self.matrix_x_unnormalized = matrix_x_unnormalized
        self.matrix_y = matrix_y
        self.mask_x = mask_x
        # Read in TSV file
        self.datafile_path = datafile_path
        self.datafile = pd.read_csv(self.datafile_path, sep="\t", engine="python", skipfooter=3, index_col=False)
        self.datafile_ids = datafile_ids
        
        # Needed for SeqModel
        self.matrix_x_unbounded = matrix_x_unbounded
        self.matrix_y_unbounded = matrix_y_unbounded
        self.word_lengths = word_lengths
        
        assert matrix_x.shape[0] == matrix_y.shape[0] == mask_x.shape[0]
        self.subset_size = matrix_x.shape[0]
        self.sample_pos = 0
        self.epochs_completed = 0
    
    # Return whole set, used form SeqModel word prediction
    def get_set(self):
        # Flatten array: all training samples are put after each other
        # to become one long line of characters
        X = np.concatenate(self.matrix_x_unbounded)
        Y = np.concatenate(self.matrix_y_unbounded)
        
        assert X.shape[0] == Y.shape[0] == np.sum(self.word_lengths)
        
        return X, Y, self.word_lengths
    
    def get_size(self):
        return self.subset_size
    
    def get_dimensions(self):
        return self.matrix_x.shape
    
    def get_epochs_completed(self):
        return self.epochs_completed
    
    def reset_sample_pos(self):
        self.sample_pos = 0
        
    def get_batch(self, val=False):
        start = self.sample_pos
        self.sample_pos += self.batch_size
        if self.sample_pos > self.subset_size:
            self.epochs_completed += 1
            perm = np.arange(self.subset_size)
            if not val:  # Validation or test set has fixed order
                np.random.shuffle(perm)
            self.matrix_x = self.matrix_x[perm]
            self.matrix_x_unnormalized = self.matrix_x_unnormalized[perm]
            self.matrix_y = self.matrix_y[perm]
            self.mask_x = self.mask_x[perm]

            start = 0
            self.sample_pos = self.batch_size
            assert self.batch_size <= self.subset_size

        end = self.sample_pos
        return self.matrix_x[start:end], self.matrix_x_unnormalized[start:end], self.matrix_y[start:end], self.mask_x[start:end]
    
    def get_dataframe(self, conversion_key, input_encoding, output_encoding):
        input_words = []
        target_words = []
        for ex in np.arange(self.subset_size):
            input_word, _ = data.word_surface(self.matrix_x[ex], conversion_key[0], input_encoding)
            target_word, _ = data.word_surface(self.matrix_y[ex], conversion_key[1], output_encoding)
            
            input_cut = [t for t in input_word if t != "."]
            target_cut = [t for t in target_word if t != "."]
            
            # Save lists of input and target words for baseline calculation
            input_words.append("".join(input_cut))
            target_words.append("".join(target_cut))
        return pd.DataFrame([input_words, target_words]).T
    
    def get_datafile_record(self, index):
        # Get data file row IDs from datafile_ids list
        id1, id2 = self.datafile_ids[index]
        
        df1 = self.datafile[self.datafile["ID"] == id1]
        df2 = self.datafile[self.datafile["ID"] == id2]
        return df1, df2
    
    def filter_cognates(self):
        indices = np.arange(self.subset_size)
        cognate_indices = []
        # Iterate over indices of word pairs
        for i in indices:
            # Get datafile record for word1,word2 of this word pair
            rec1, rec2 = self.get_datafile_record(i)
            # First try to use IElex
            if "COGNATES_IELEX" in rec1 and pd.notnull(rec1["COGNATES_IELEX"].iloc[0]):
                 if rec1["COGNATES_IELEX"].iloc[0] == rec2["COGNATES_IELEX"].iloc[0]:
                     cognate_indices.append(i)
            # Only use Lexstat if Ielex is unavailable
            elif "COGNATES_LEXSTAT" in rec1 and pd.notnull(rec1["COGNATES_LEXSTAT"].iloc[0]):
                if rec1["COGNATES_LEXSTAT"].iloc[0] == rec2["COGNATES_LEXSTAT"].iloc[0]:
                    cognate_indices.append(i)
        
        return Subset(self.batch_size, self.matrix_x[cognate_indices],
                            self.matrix_x_unnormalized[cognate_indices],
                            self.matrix_y[cognate_indices],
                            self.mask_x[cognate_indices],
                            self.matrix_x_unbounded[cognate_indices],
                            self.matrix_y_unbounded[cognate_indices],
                            self.datafile_path,
                            self.datafile_ids[cognate_indices],
                            self.word_lengths[cognate_indices])