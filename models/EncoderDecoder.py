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
from models.GatedLayer import gated_layer
from models import baseline
from util import utility
from dataset import data

# Import library
import lasagne
from lasagne.layers import InputLayer, DropoutLayer, DenseLayer, ReshapeLayer, SliceLayer, DimshuffleLayer, ConcatLayer
from lasagne.regularization import regularize_layer_params, regularize_network_params, l2
import numpy as np
import pandas as pd
import theano
import theano.tensor as T
import time
from collections import defaultdict


class EncoderDecoder():

    def __init__(self, batch_size, max_len, voc_size, n_hidden, n_layers_encoder, n_layers_decoder, n_layers_dense, bidirectional_encoder, bidirectional_decoder, encoder_only_final, dropout, learning_rate, learning_rate_decay, adaptive_learning_rate, reg_weight, grad_clip, initialization, gated_layer_type, cognacy_prior, input_encoding, output_encoding, conversion_key, export_weights, optimizer):
        self.batch_size = batch_size
        self.max_len = max_len
        self.voc_size = voc_size
        self.n_hidden = n_hidden
        self.n_layers_encoder = n_layers_encoder
        self.n_layers_decoder = n_layers_decoder
        self.n_layers_dense = n_layers_dense
        self.bidirectional_encoder = bidirectional_encoder
        self.bidirectional_decoder = bidirectional_decoder
        self.encoder_only_final = encoder_only_final
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.adaptive_learning_rate = adaptive_learning_rate
        self.reg_weight = reg_weight
        self.grad_clip = grad_clip
        self.gated_layer_type = gated_layer_type
        self.cognacy_prior = cognacy_prior
        self.input_encoding = input_encoding
        self.output_encoding = output_encoding
        self.conversion_key = conversion_key
        self.export_weights = export_weights
        self.optimizer = optimizer
        
        if initialization == "constant":
            self.init = lasagne.init.Constant(0.)
        elif initialization == "xavier_normal":
            self.init = lasagne.init.GlorotNormal()
        elif initialization == "xavier_uniform":
            self.init = lasagne.init.GlorotUniform()
        
        print("Building network ...")
        self.l_in_X = InputLayer(shape=(self.batch_size, self.max_len[0], self.voc_size[0]))
        self.l_mask_X = InputLayer(shape=(self.batch_size, self.max_len[0]))
        self.l_target_Y = InputLayer(shape=(self.batch_size, self.max_len[1], self.voc_size[1]))
        
        # Encoder-decoder
        self.l_encoder = self._create_encoder(encoder_input=self.l_in_X, mask=self.l_mask_X)
        
        self.l_decoder = self._create_decoder(decoder_input=self.l_encoder, decoder_input_hid=self.l_encoder, max_len=self.max_len[1], n_features_context=self.n_hidden)
        
        #### Reshape before dense layers
        self.l_reshape = ReshapeLayer(self.l_decoder, (self.batch_size * self.max_len[1], 2 * self.n_hidden if self.bidirectional_decoder else self.n_hidden))
        
        self.l_dense_input = self.l_reshape
        
        # Extra dense layers:
        for _ in np.arange(self.n_layers_dense - 1):
            self.l_dense_input = DenseLayer(self.l_dense_input, num_units=self.voc_size[1], nonlinearity=lasagne.nonlinearities.rectify)
            
            if self.dropout > 0.0:
                self.l_dense_input = DropoutLayer(self.l_dense_input, p=self.dropout, rescale=False)
        
        # Output layer
        # Phonetic: sigmoid, multi-label classification
        if self.output_encoding == "phonetic" or self.output_encoding == "embedding":
            output_nonlinearity = lasagne.nonlinearities.sigmoid
        # Character: softmax, single-label classification
        elif self.output_encoding == "character":
            output_nonlinearity = lasagne.nonlinearities.softmax
        self.l_out = DenseLayer(self.l_dense_input, num_units=self.voc_size[1], nonlinearity=output_nonlinearity)
        
        # Fetch network output
        # Deterministic (without dropout) for prediction
        self.predicted_values = lasagne.layers.get_output(self.l_out)
        self.predicted_values_deterministic = lasagne.layers.get_output(self.l_out, deterministic=True)
        self.context_vector = lasagne.layers.get_output(self.l_encoder, deterministic=True)
        # Do reshape on target values (actually build small network), and get output
        self.target_values = lasagne.layers.get_output(ReshapeLayer(self.l_target_Y, (self.batch_size * self.max_len[1], self.voc_size[1])))
        
        # Array storing the prediction errors, initialized with 10 as first error
        self.prediction_errors = [10]
        self.error_threshold = T.scalar(name='error_threshold')
        
        print("Creating loss function...")
        self.loss = self._create_loss(output_layer=self.l_out, predicted=self.predicted_values, target=self.target_values, error_threshold=self.error_threshold)
        
        # Compute SGD updates for training
        print("Computing updates ...")
        self.updates = self._compute_updates(output_layer=self.l_out, loss=self.loss[0])
        
        # Theano functions for training and computing loss
        print("Compiling functions ...")
        self.train_func, self.loss_func, self.predict_func, self.vector_func = self._compile_functions(X_input=self.l_in_X.input_var, Y_input=self.l_target_Y.input_var, mask=self.l_mask_X.input_var, error_threshold=self.error_threshold, loss=self.loss, updates=self.updates, pred_values_determ=self.predicted_values_deterministic, context_vector=self.context_vector)
    
    def _create_encoder(self, encoder_input, mask, resetgate=None, updategate=None, hiddengate=None, resetgate_fw=None, updategate_fw=None, hiddengate_fw=None, w_comb=None):
        # Use standard gates, if no weight sharing gates are supplied
        if resetgate is None:
            resetgate = lasagne.layers.Gate(W_cell=None)
        if updategate is None:
            updategate = lasagne.layers.Gate(W_cell=None)
        if hiddengate is None:
            hiddengate = lasagne.layers.Gate(W_cell=None, nonlinearity=lasagne.nonlinearities.tanh)
        if resetgate_fw is None:
            resetgate_fw = lasagne.layers.Gate(W_cell=None)
        if updategate_fw is None:
            updategate_fw = lasagne.layers.Gate(W_cell=None)
        if hiddengate_fw is None:
            hiddengate_fw = lasagne.layers.Gate(W_cell=None, nonlinearity=lasagne.nonlinearities.tanh)
        
        # n-1 encoder layers return all sequence values
        encoder_input_orig = encoder_input
        # for _ in np.arange(self.n_layers_encoder-1):
            # encoder_input = gated_layer(encoder_input, self.n_hidden,
                                            # mask_input=mask,
                                            # grad_clipping=self.grad_clip,
                                            # only_return_final=False, backwards=True,
                                            # cell_init = self.init,
                                            # hid_init=self.init,
                                            # gated_layer_type=self.gated_layer_type,
                                            # resetgate=resetgate, updategate=updategate, hidden_update=hiddengate)
            # if self.dropout > 0.0:
                # encoder_input = DropoutLayer(encoder_input, p=self.dropout, rescale=False)

        # Final encoder layer only returns final value
        l_encoder = gated_layer(encoder_input, self.n_hidden,
                                    mask_input=mask,
                                    grad_clipping=self.grad_clip,
                                    only_return_final=self.encoder_only_final, backwards=True,
                                    cell_init=self.init,
                                    hid_init=self.init,
                                    gated_layer_type=self.gated_layer_type,
                                    resetgate=resetgate, updategate=updategate, hidden_update=hiddengate, name="enc_bw")
        
        if not self.encoder_only_final:
            l_encoder = self._combine_encoder_steps(l_encoder)
        
        if self.dropout > 0.0:
            l_encoder = DropoutLayer(l_encoder, p=self.dropout, rescale=False)
        
        # # If bidirectional encoder: create separate forward encoder stack,
        # and combine with backward encoder at the end
        if self.bidirectional_encoder:
            encoder_input_fw = encoder_input_orig
            # # n-1 encoder layers return all sequence values
            # for _ in np.arange(self.n_layers_encoder-1):
                # encoder_input_fw = gated_layer(encoder_input_fw, self.n_hidden,
                                                # mask_input=mask,
                                                # grad_clipping=self.grad_clip,
                                                # only_return_final=False, backwards=False,
                                                # cell_init = self.init,
                                                # hid_init=self.init,
                                                # gated_layer_type=self.gated_layer_type,
                                                # resetgate=resetgate, updategate=updategate, hidden_update=hiddengate)
                # if self.dropout > 0.0:
                    # encoder_input_fw = DropoutLayer(encoder_input_fw, p=self.dropout, rescale=False)

            # Final encoder layer only returns final value
            l_encoder_fw = gated_layer(encoder_input_fw, self.n_hidden,
                                        mask_input=mask,
                                        grad_clipping=self.grad_clip,
                                        only_return_final=self.encoder_only_final, backwards=False,
                                        cell_init=self.init,
                                        hid_init=self.init,
                                        gated_layer_type=self.gated_layer_type,
                                        resetgate=resetgate_fw, updategate=updategate_fw, hidden_update=hiddengate_fw, name="enc_fw")
            
            if not self.encoder_only_final:
                l_encoder_fw = self._combine_encoder_steps(l_encoder_fw)
            
            if self.dropout > 0.0:
                l_encoder_fw = DropoutLayer(l_encoder_fw, p=self.dropout, rescale=False)
            
            # Combine backward (original) and forward encoder
            l_concat = ConcatLayer([l_encoder, l_encoder_fw])
            
            # If no shared weight available, initialize with new weight
            if w_comb is None:
                w_comb = self.init
            l_encoder = DenseLayer(l_concat, num_units=self.n_hidden, nonlinearity=lasagne.nonlinearities.rectify, W=w_comb, name="dense_comb")
        return l_encoder
    
    def _create_decoder(self, decoder_input, decoder_input_hid, max_len, n_features_context, resetgate=None, updategate=None, hiddengate=None):
        #### Decoder (forwards)
        # decoder_input_orig keeps the value of l_encoder, which is given to this method
        decoder_input_orig = decoder_input
        # decoder_input then becomes a repeated l_encoder
        for _ in np.arange(max_len - 1):
            decoder_input = ConcatLayer([decoder_input, decoder_input_orig])
        
        decoder_input = ReshapeLayer(decoder_input, (self.batch_size, n_features_context, max_len), name="reshape_enc_dec")
        decoder_input = DimshuffleLayer(decoder_input, (0, 2, 1), name="dimshuf_enc_dec")
        
        # Use standard gates, if no weight sharing gates are supplied
        if resetgate is None:
            resetgate = lasagne.layers.Gate(W_cell=None)
        if updategate is None:
            updategate = lasagne.layers.Gate(W_cell=None)
        if hiddengate is None:
            hiddengate = lasagne.layers.Gate(W_cell=None, nonlinearity=lasagne.nonlinearities.tanh)
        
        decoder_output = gated_layer(decoder_input, self.n_hidden,
                                      grad_clipping=self.grad_clip,
                                      only_return_final=False, backwards=False,
                                      cell_init=self.init,
                                      hid_init=decoder_input_hid,
                                      gated_layer_type=self.gated_layer_type,
                                      resetgate=resetgate, updategate=updategate, hidden_update=hiddengate,
                                      name="dec")
        if self.dropout > 0.0:
            decoder_output = DropoutLayer(decoder_output, p=self.dropout, rescale=False)
        l_decoder = decoder_output
        
        return l_decoder
        
    def _compute_updates(self, output_layer, loss, params=None):
        if params is None:
            # Retrieve all parameters from the network
            all_params = lasagne.layers.get_all_params(output_layer)
            print("All params:")
            print(all_params)
        else:
            # Use given list of parameters, a subset of all parameters
            all_params = params
        self.lr_var = T.scalar(name='learning_rate')
        if self.optimizer == "adagrad":
            return lasagne.updates.adagrad(loss_or_grads=loss, params=all_params, learning_rate=self.lr_var)
        elif self.optimizer == "adam":
            return lasagne.updates.adam(loss_or_grads=loss, params=all_params, learning_rate=self.lr_var)
        elif self.optimizer == "sgd":
            return lasagne.updates.sgd(loss_or_grads=loss, params=all_params, learning_rate=self.lr_var)
    
    def _compile_functions(self, X_input, Y_input, mask, error_threshold, loss, updates, pred_values_determ, context_vector):
        if Y_input == None:  # for proto-languages
            train_func = theano.function([X_input, mask, error_threshold, self.lr_var],
                                    [loss[0], loss[1], loss[2]], updates=updates, on_unused_input="warn")
            loss_func = theano.function(
                                    [X_input, mask, error_threshold],
                                    [loss[0], loss[1], loss[2], loss[3], pred_values_determ], on_unused_input="warn")
        else:
            train_func = theano.function([X_input, Y_input, mask, error_threshold, self.lr_var],
                                    [loss[0], loss[1], loss[2]], updates=updates, on_unused_input="warn")
            loss_func = theano.function(
                                    [X_input, Y_input, mask, error_threshold],
                                    [loss[0], loss[1], loss[2], loss[3], pred_values_determ], on_unused_input="warn")
        predict_func = theano.function(
                                [X_input, mask],
                                pred_values_determ, on_unused_input="warn")
        vector_func = None
        if self.export_weights:
            vector_func = theano.function(
                                    [X_input, mask],
                                    context_vector, on_unused_input="warn")
        return train_func, loss_func, predict_func, vector_func
    
    def _create_loss(self, output_layer, predicted, target, error_threshold, proto_loss_multiplier=1.0):
        # Regularization term
        reg_term = self.reg_weight * regularize_network_params(output_layer, l2)
        
        # Source-target penalty term:
        # distance between source and target is subtracted from loss
        # So less is learned from non-cognates

        # Phonetic: binary crossentropy, multi-label classifcation
        if self.output_encoding == "phonetic" or self.output_encoding == "embedding":
            loss = T.sum(lasagne.objectives.binary_crossentropy(predicted, target)) / self.batch_size + reg_term
        # Character: categorical crossentropy, single label classification
        elif self.output_encoding == "character":
            loss = T.sum(lasagne.objectives.categorical_crossentropy(predicted, target)) / self.batch_size + reg_term
            
        # Multiply loss with cognacy prior:
        # more should be learned from probable cognate examples
        if self.cognacy_prior > 0.0:
            target_prediction_error = T.sum(lasagne.objectives.squared_error(predicted, target)) / self.batch_size
            # sigmoid(-error+mean_error_history)
            # Cognacy prior is high for low error, but declines steeply
            # when error above mean_error_history
            cognacy_prior_factor = utility.sigmoid(-target_prediction_error + error_threshold)
            loss *= cognacy_prior_factor
        else:
            cognacy_prior_factor = T.constant(1)
            target_prediction_error = T.constant(0)
            
        loss *= proto_loss_multiplier
        return loss, cognacy_prior_factor, target_prediction_error, error_threshold
    
    def _combine_encoder_steps(self, encoder):
        print(lasagne.layers.get_output_shape(encoder))
        shuffled_encoder = DimshuffleLayer(encoder, (0, 2, 1))
        print(lasagne.layers.get_output_shape(shuffled_encoder))
        encoder = lasagne.layers.DenseLayer(shuffled_encoder, num_units=1, num_leading_axes=2)
        print(lasagne.layers.get_output_shape(encoder))
        encoder = ReshapeLayer(encoder, shape=(self.batch_size, self.n_hidden))
        print(lasagne.layers.get_output_shape(encoder))
        return encoder

    def train(self, trainset, valset, n_epochs):
        n_val_batches = valset.get_size() // self.batch_size
        
        # To compute mean error, look back at last processed 10% of training set
        n_batches_history = trainset.get_size() // self.batch_size
        print("Training ...")
        epochs_completed = 0
        start_time = time.time()
        plot_losses = []
        plot_distances = []
        while (epochs_completed < n_epochs):
            X, X_unnorm, Y, m_X = trainset.get_batch()
            lr = self.learning_rate * (self.learning_rate_decay ** epochs_completed)
            if self.adaptive_learning_rate > 0.0:
                # Adapt learning rate to X,Y edit distance:
                # learn more from probable cognates
                dist = self._compute_distance_batch_encoded(X_unnorm, Y, max_len_tar=self.max_len[1], voc_size_tar=self.voc_size[1], conversion_key=self.conversion_key)
                lr = lr - self.adaptive_learning_rate * dist
            
            # Prediction error threshold is mean+1 standard deviation
            error_threshold = np.mean(self.prediction_errors) + self.cognacy_prior * np.std(self.prediction_errors)
            
            # Perform training
            loss, cognacy_prior, prediction_error = self.train_func(X, Y, m_X, error_threshold, lr)
            # Now update the prediction_errors variable, to compute mean for next round
            self.prediction_errors.append(prediction_error)

            # Calculate validation loss when new epoch is entered
            prev_epochs_completed = epochs_completed
            epochs_completed = trainset.get_epochs_completed()
            if prev_epochs_completed < epochs_completed:
                loss_batches = []
                distance_batches = []
                for _ in np.arange(n_val_batches):
                    X_val, X_val_unnorm, Y_val, mask_X_val = valset.get_batch(val=True)
                    # Calculate and store loss
                    # Prediction error threshold is mean+1 standard deviation
                    error_threshold = np.mean(self.prediction_errors) + self.cognacy_prior * np.std(self.prediction_errors)
                    loss, cognacy_prior, prediction_error, mean_error, predictions = self.loss_func(X_val, Y_val, mask_X_val, error_threshold)
                    if self.cognacy_prior > 0.0:
                        print(cognacy_prior, prediction_error, mean_error, np.mean(self.prediction_errors))
                    loss_batches.append(loss)
                    # Calculate and store distance
                    _, _, _, distances, _ = self._compute_distance_batch_encoded(X_val_unnorm, Y_val, max_len_tar=self.max_len[1], voc_size_tar=self.voc_size[1], conversion_key=self.conversion_key, predictions=predictions)
                    distance_batches += distances
                    
                duration = time.time() - start_time
                avg_loss = np.average(loss_batches)
                avg_distance = np.average(distance_batches)
                plot_losses.append((epochs_completed, avg_loss))
                plot_distances.append((epochs_completed, avg_distance))
                print("Epoch {0:2} validation loss = {1:.2f}, distance = {2:.2f}, duration = {3:.2f}".format(epochs_completed, avg_loss, avg_distance, duration))
                start_time = time.time()
        
        return plot_losses, plot_distances
    
    def _compute_distance_batch_encoded(self, X, Y, max_len_tar, voc_size_tar, conversion_key, predictions=None):
        distances_t_p = []
        distances_s_t = []
        input_words = []
        target_words = []
        predicted_words = []
        if predictions is not None:
            predictions = np.reshape(predictions, (self.batch_size, max_len_tar, voc_size_tar))
        for ex in np.arange(self.batch_size):
            _, input_tokens = data.word_surface(X[ex], conversion_key[0], self.input_encoding)
            _, target_tokens = data.word_surface(Y[ex], conversion_key[1], self.output_encoding)
            
            input_cut = [t for t in input_tokens if t != "."]
            target_cut = [t for t in target_tokens if t != "."]
            if predictions is not None:
                predicted_word, predicted_tokens = data.word_surface(predictions[ex], conversion_key[1], self.output_encoding)
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

    def predict(self, testset, max_len_tar, voc_size_tar, conversion_key, predict_func, print_output=True):
        n_batches = testset.get_size() // self.batch_size
        all_distances_t_p = []
        all_distances_s_t = []
        all_input_words = []
        all_target_words = []
        all_predicted_words = []
        all_input_raw = []
        all_target_raw = []
        context_vectors = []
        
        # Set sample position of set we predict on to 0:
        # we want to process the whole set only once, in its original order.
        # The sample position can be different, when batches have been
        # drawn from set before
        testset.reset_sample_pos()
        
        if print_output:
            text_output = ""
            header_template = "{0:20} {1:20} {2:20} {3:8}"
            template = "{0:20} {1:20} {2:20} {3:.2f}"
            text_output += header_template.format("INPUT", "TARGET", "PREDICTION", "DISTANCE") + "\n"

        for _ in np.arange(n_batches):
            X_test, X_test_unnorm, Y_test, mask_X_test = testset.get_batch(val=True)
            predictions = predict_func(X_test, mask_X_test)
            input_words, target_words, predicted_words, distances_t_p, distances_s_t = self._compute_distance_batch_encoded(X_test_unnorm, Y_test, max_len_tar=max_len_tar, voc_size_tar=voc_size_tar, conversion_key=conversion_key, predictions=predictions)
            all_distances_t_p += distances_t_p
            all_distances_s_t += distances_s_t
            all_input_words += input_words
            all_input_raw.append(X_test_unnorm)
            all_target_raw.append(Y_test)
            all_target_words += target_words
            all_predicted_words += predicted_words
            if self.export_weights:
                context_vector = self.vector_func(X_test, mask_X_test)
                context_vectors.append(context_vector)
            
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
        return avg_distance, results_table, (context_vectors, all_input_words, all_target_words, all_input_raw, all_target_raw)

