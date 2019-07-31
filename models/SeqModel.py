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
from models import baseline
from util import utility
from dataset import data

from collections import defaultdict
import numpy as np
import pandas as pd
from seqlearn.perceptron import StructuredPerceptron
import time


class SeqModel():

    def __init__(self, input_encoding, conversion_key, n_iter_seq):
        self.input_encoding = input_encoding
        self.conversion_key = conversion_key
        
        # self.model = MultinomialHMM()
        self.model = StructuredPerceptron(max_iter=n_iter_seq)

    def train(self, trainset):
        print("Training ...")
        start_time = time.time()
        plot_losses = []
        plot_distances = []
        
        X_train, Y_train, lengths_train = trainset.get_set()
        
        # Perform training
        self.model.fit(X_train, Y_train, lengths=lengths_train)
        
        duration = time.time() - start_time
        print("Duration = {0:.2f}".format(duration))
        
        return plot_losses, plot_distances
    
    def _compute_distance(self, X, Y, predictions=None, word_lengths=None):
        distances_t_p = []
        distances_s_t = []
        input_words = []
        target_words = []
        predicted_words = []
        # X, Y and predictions are long lists of characters. Split into words again.
        if predictions is not None:
            predictions = data.split_predictions(predictions, word_lengths)
        X = data.split_predictions(X, word_lengths)
        Y = data.split_predictions(Y, word_lengths)
        for ex in np.arange(len(X)):
            # X is encoded and has to be decoded
            _, input_tokens = data.word_surface(X[ex], self.conversion_key[0], self.input_encoding)
            # Y is already in surface form
            #target_word = "".join(Y[ex])
            target_tokens = Y[ex]
            
            input_cut = [t for t in input_tokens if t != "."]
            target_cut = [t for t in target_tokens if t != "."]
            if predictions is not None:
                # Predictions are already in surface form
                #predicted_word = "".join(predictions[ex])
                predicted_tokens = predictions[ex]
                predicted_cut = [t for t in predicted_tokens if t != "."]
                dist_t_p = utility.calculate_levenshtein(target_cut, predicted_cut)
                distances_t_p.append(dist_t_p)
                predicted_words.append(predicted_cut)
            dist_s_t = utility.calculate_levenshtein(input_cut, target_cut)
            distances_s_t.append(dist_s_t)
            input_words.append(input_cut)
            target_words.append(target_cut)
        if predictions is not None:
            return input_words, target_words, predicted_words, distances_t_p, distances_s_t
        else:
            return np.mean(distances_s_t)

    def predict(self, testset, print_output=True):
        all_distances_t_p = []
        all_distances_s_t = []
        all_input_words = []
        all_target_words = []
        all_predicted_words = []
        
        if print_output:
            text_output = ""
            header_template = "{0:20} {1:20} {2:20} {3:8}"
            template = "{0:20} {1:20} {2:20} {3:.2f}"
            text_output += header_template.format("INPUT", "TARGET", "PREDICTION", "DISTANCE") + "\n"

        # Fetch whole test set in format suitable for seqmodel
        X_test, Y_test, lengths_test = testset.get_set()
        predictions = self.model.predict(X_test, lengths=lengths_test)
        input_words, target_words, predicted_words, distances_t_p, distances_s_t = self._compute_distance(X_test, Y_test, predictions, lengths_test)
        all_distances_t_p += distances_t_p
        all_distances_s_t += distances_s_t
        all_input_words += input_words
        all_target_words += target_words
        all_predicted_words += predicted_words
            
        row_dict = defaultdict(list)
        for i in np.arange(len(all_input_words)):
            input_word = all_input_words[i]
            target_word = all_target_words[i]
            predicted_word = all_predicted_words[i]
            dist_t_p = all_distances_t_p[i]
            dist_s_t = all_distances_s_t[i]
            if print_output:
                text_output += template.format("".join(input_word), "".join(target_word), "".join(predicted_word), dist_t_p) + "\n"
            row_dict["INPUT"].append(" ".join(input_word))
            row_dict["TARGET"].append(" ".join(target_word))
            row_dict["PREDICTION"].append(" ".join(predicted_word))
            row_dict["DISTANCE_T_P"].append(dist_t_p)
            row_dict["DISTANCE_S_T"].append(dist_s_t)
            
            # Get information from datafile
            records = testset.get_datafile_record(i)
            row_dict["CONCEPT"].append(records[0].iloc[0]["CONCEPT"])
            # Add columns for cognate judgments of both word1 and word2
            for ix in [0, 1]:
                if "COGNATES_LEXSTAT" in records[ix]:
                    row_dict["COGNATES_LEXSTAT" + str(ix)].append(records[ix].iloc[0]["COGNATES_LEXSTAT"])
                if "COGNATES_IELEX" in records[ix]:
                    row_dict["COGNATES_IELEX" + str(ix)].append(records[ix].iloc[0]["COGNATES_IELEX"])

        avg_distance = np.average(all_distances_t_p)
        if print_output:
            text_output += "Average distance: " + str(avg_distance) + "\n"
            print(text_output)
        results_table = pd.DataFrame(row_dict)
        return avg_distance, results_table

