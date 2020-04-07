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
from models.EncoderDecoder import EncoderDecoder
from models.GatedLayer import gated_layer
from models import baseline

from collections import defaultdict
import lasagne
from lasagne.layers import InputLayer, DropoutLayer, DenseLayer, ReshapeLayer, SliceLayer, DimshuffleLayer, ConcatLayer
from lasagne.regularization import regularize_layer_params, regularize_network_params, l2
import newick
import numpy as np
import pandas as pd
import theano
import theano.tensor as T
import time
from util import utility


class PhylNet(EncoderDecoder):

    def __init__(self, langs, lang_pairs, tree_string, batch_size, max_len, voc_size, n_hidden, n_layers_encoder, n_layers_decoder, n_layers_dense, bidirectional_encoder, bidirectional_decoder, encoder_only_final, dropout, learning_rate, learning_rate_decay, adaptive_learning_rate, reg_weight, grad_clip, initialization, gated_layer_type, cognacy_prior, input_encoding, output_encoding, conversion_key, export_weights, optimizer, units_phyl, train_proto):
        self.langs = langs
        self.lang_pairs = lang_pairs
        self.tree = newick.loads(tree_string)
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
        self.n_units_phyl = units_phyl
        self.train_proto = train_proto
        
        if initialization == "constant":
            self.init = lasagne.init.Constant(0.)
        elif initialization == "xavier_normal":
            self.init = lasagne.init.GlorotNormal()
        elif initialization == "xavier_uniform":
            self.init = lasagne.init.GlorotUniform()
        
        # Build tree-shaped neural network from tree string
        # Recursive method
        # langs = self.build_network(self.tree[0])
        
        # Build network, one for every language
        # A network starts at that language and goes to all other languages
        print("Building network, for all languages...")
        self.l_in_X = {}
        self.l_mask_X = {}
        self.l_target_Y = {}
        self.l_encoder = {}
        self.l_out = defaultdict(dict)
        self.network = defaultdict(dict)
        self.input_values = {}
        self.predicted_values = defaultdict(dict)
        self.predicted_values_deterministic = defaultdict(dict)
        self.context_vector = defaultdict(dict)
        self.target_values = defaultdict(dict)
        self.prediction_errors = defaultdict(dict)
        self.error_threshold = defaultdict(dict)
        self.loss = defaultdict(dict)
        self.train_func = defaultdict(dict)
        self.loss_func = defaultdict(dict)
        self.predict_func = defaultdict(dict)
        self.vector_func = defaultdict(dict)
        self.proto_func = defaultdict(dict)
        self.updates = defaultdict(dict)
        
        # List to store parameters, which are the only updated paramters for
        # protolanguage (only decoder+layers around)
        self.proto_params = defaultdict(list)
        ######### Language-dependent part
        # For now, we assume a tree structure where lang0 and lang1 are more
        # closely related than lang32
        #  /\
        # /\ \
        # 0 1 2
        
        lang0 = self.langs[0]
        lang1 = self.langs[1]
        lang2 = self.langs[2]
        
        proto01 = lang0 + "_" + lang1
        self.proto_languages = [proto01, "root"]
        
        self.lang_pairs_proto = utility.generate_pairs(self.langs, self.langs + self.proto_languages, allow_permutations=True)
        
        # TODO: self.n_hidden or lower number
        self.gate_init = lasagne.init.Normal(0.1)
        
        # Create shared weights for dense phylogenetic layers
        self.w_dense = defaultdict(dict)
        # Create shared weights for encoders and decoders.
        # Shared between all encoders with same input language and
        # decoders with same output language
        self.w = defaultdict(lambda: defaultdict(lambda:defaultdict(lambda: defaultdict(dict))))
        self.gate = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        self.w_comb = {}
        self.w_out = {}
        self.w_enc_phyl = {}
        self.w_phyl_dec_hid = {}
        
        for lang in self.langs + self.proto_languages:
            for layer in ["enc", "dec"]:
                for dirn in ["fw", "bw"]:
                    for weight in ["reset", "update", "hidden"]:
                        if layer == "enc":
                            input_dim = self.voc_size[0]
                        elif layer == "dec":
                            input_dim = self.n_units_phyl
                        self.w[lang][layer][weight][dirn]["in"] = lasagne.utils.create_param(spec=self.gate_init, shape=(input_dim, self.n_hidden), name="w_" + lang + "_" + layer + "_" + dirn + "_" + weight + "_" + "in")
                        self.w[lang][layer][weight][dirn]["hid"] = lasagne.utils.create_param(spec=self.gate_init, shape=(self.n_hidden, self.n_hidden), name="w_" + lang + "_" + layer + "_" + dirn + "_" + weight + "_" + "hid")
                        # if layer == "dec":
                            # Store this parameter for all language pairs which have lang as output
                            # self.proto_params[lang].append(self.w[lang][layer][weight][dirn]["in"])
                            # self.proto_params[lang].append(self.w[lang][layer][weight][dirn]["hid"])
                        
                        # Create gates
                        if weight == "hidden":
                            nonlinearity = lasagne.nonlinearities.tanh
                        else:
                            nonlinearity = lasagne.nonlinearities.sigmoid
                        self.gate[lang][layer][weight][dirn] = lasagne.layers.Gate(W_cell=None, W_in=self.w[lang][layer][weight][dirn]["in"], W_hid=self.w[lang][layer][weight][dirn]["hid"], nonlinearity=nonlinearity)
                        
            # Create shared weight for language-independent output layer
            self.w_out[lang] = lasagne.utils.create_param(spec=self.init, shape=(self.n_hidden, self.voc_size[1]), name="w_out")
            # self.proto_params[lang].append(self.w_out[lang])
            
            # Shared variable for conversion layers
            self.w_phyl_dec_hid[lang] = lasagne.utils.create_param(spec=self.init, shape=(self.n_units_phyl, self.n_hidden), name="w_phyl_dec_hid")
            # self.proto_params[lang].append(self.w_phyl_dec_hid[lang])
        
        # This creates shared variables for every pair of nodes,
        # these are used as weights for the dense layers, some stay unused.
        for i in [lang0, lang1, lang2, lang0 + "_" + lang1, "root"]:
            for j in [lang0, lang1, lang2, lang0 + "_" + lang1, "root"]:
                if i is not j:
                    self.w_dense[i][j] = lasagne.utils.create_param(spec=self.init, shape=(self.n_units_phyl, self.n_units_phyl), name="w_dense-" + i + "-" + j)
        
        # Create input side for every language
        for lang in self.langs:
            # Create input layer and encoder for every language
            self.l_in_X[lang] = InputLayer(shape=(self.batch_size, self.max_len[lang], self.voc_size[0]), name="input_X")
            self.l_mask_X[lang] = InputLayer(shape=(self.batch_size, self.max_len[lang]), name="input_mask")
            self.l_target_Y[lang] = InputLayer(shape=(self.batch_size, self.max_len[lang], self.voc_size[1]), name="input_Y")
            
            # # Reshape input and get network output. To compare to predictions, for protoform loss function
            l_reshape_inp = ReshapeLayer(self.l_in_X[lang], (self.batch_size * self.max_len[lang], self.voc_size[0]), name="reshape_inp")
            self.input_values[lang] = lasagne.layers.get_output(l_reshape_inp)
            
            # Create shared weight for combining fw and bw encoder, language-independent
            self.w_comb[lang] = lasagne.utils.create_param(spec=self.init, shape=(2 * self.n_hidden, self.n_hidden), name="w_comb")
            # Shared variable for conversion layers
            self.w_enc_phyl[lang] = lasagne.utils.create_param(spec=self.init, shape=(self.n_hidden, self.n_units_phyl), name="w_enc_phyl")
            
            gate_enc = self.gate[lang]["enc"]
            self.l_encoder[lang] = self._create_encoder(encoder_input=self.l_in_X[lang], mask=self.l_mask_X[lang], resetgate=gate_enc["reset"]["bw"], updategate=gate_enc["update"]["bw"], hiddengate=gate_enc["hidden"]["bw"], resetgate_fw=gate_enc["reset"]["fw"], updategate_fw=gate_enc["update"]["fw"], hiddengate_fw=gate_enc["hidden"]["fw"], w_comb=self.w_comb[lang])
            # Conversion layer, which goes from encoder (n_hidden) to phylogenetic network (n_units_phyl)
            self.l_encoder[lang] = DenseLayer(self.l_encoder[lang], num_units=self.n_units_phyl, W=self.w_enc_phyl[lang])
        
        #### From l0
        self.network[lang0][lang0 + "_" + lang1] = self.create_phyl_layer(self.l_encoder[lang0], self.w_dense[lang0][lang0 + "_" + lang1])
        # l0-> l1
        self.network[lang0][lang1] = self.create_phyl_layer(self.network[lang0][lang0 + "_" + lang1], self.w_dense[lang0 + "_" + lang1][lang1])
        self.create_decoder_output(lang0, lang1)
        # l0-> l2
        self.network[lang0]["root"] = self.create_phyl_layer(self.network[lang0][lang0 + "_" + lang1], self.w_dense[lang0 + "_" + lang1]["root"])
        self.network[lang0][lang2] = self.create_phyl_layer(self.network[lang0]["root"], self.w_dense["root"][lang2])
        self.create_decoder_output(lang0, lang2)
        
        #### From l1
        self.network[lang1][lang0 + "_" + lang1] = self.create_phyl_layer(self.l_encoder[lang1], self.w_dense[lang1][lang0 + "_" + lang1])
        # l1-> l0
        self.network[lang1][lang0] = self.create_phyl_layer(self.network[lang1][lang0 + "_" + lang1], self.w_dense[lang0 + "_" + lang1][lang0])
        self.create_decoder_output(lang1, lang0)
        
        # l1-> l2
        self.network[lang1]["root"] = self.create_phyl_layer(self.network[lang1][lang0 + "_" + lang1], self.w_dense[lang0 + "_" + lang1]["root"])
        self.network[lang1][lang2] = self.create_phyl_layer(self.network[lang1]["root"], self.w_dense["root"][lang2])
        self.create_decoder_output(lang1, lang2)
        
        #### From l2
        self.network[lang2]["root"] = self.create_phyl_layer(self.l_encoder[lang2], self.w_dense[lang2]["root"])
        self.network[lang2][lang0 + "_" + lang1] = self.create_phyl_layer(self.network[lang2]["root"], self.w_dense["root"][lang0 + "_" + lang1])
        # l2-> l0
        self.network[lang2][lang0] = self.create_phyl_layer(self.network[lang2][lang0 + "_" + lang1], self.w_dense[lang0 + "_" + lang1][lang0])
        self.create_decoder_output(lang2, lang0)
        # l2-> l1
        self.network[lang2][lang1] = self.create_phyl_layer(self.network[lang2][lang0 + "_" + lang1], self.w_dense[lang0 + "_" + lang1][lang1])
        self.create_decoder_output(lang2, lang1)
        
        # ## Create decoder output and functions for protolanguages
        # From all languages to proto-languages proto01 and root
        for lang_in in self.langs:
            for lang_proto in self.proto_languages:
                print("Creating protolanguage decoder output and prediction functions...")
                # Use max len of input language
                self.create_decoder_output(lang_in, lang_proto, max_len=self.max_len[lang_in], create_target=False)
                # self.proto_func[lang_in][lang_proto] = theano.function([self.l_in_X[lang_in].input_var, self.l_mask_X[lang_in].input_var], self.predicted_values_deterministic[lang_in][lang_proto], on_unused_input="warn")
        
        # ## Create network loss and train functions
        # (regardless of network structure)
        for lang_a, lang_b in self.lang_pairs_proto:
            print("Creating loss functions and computing updates...")
            if lang_b in self.langs:
                self.loss[lang_a][lang_b] = self._create_loss(output_layer=self.l_out[lang_a][lang_b], predicted=self.predicted_values[lang_a][lang_b], target=self.target_values[lang_a][lang_b], error_threshold=self.error_threshold[lang_a][lang_b])
                self.updates[lang_a][lang_b] = self._compute_updates(output_layer=self.l_out[lang_a][lang_b], loss=self.loss[lang_a][lang_b][0])
            # TODO: train proto-languages in the right way.
            # Currently, this fails because of disconnected input, because not all parameters are trained
            if self.train_proto:
                if lang_b in self.proto_languages:
                    # Use loss function which compares to input language
                    self.loss[lang_a][lang_b] = self._create_loss(output_layer=self.l_out[lang_a][lang_b], predicted=self.predicted_values[lang_a][lang_b], target=self.input_values[lang_a], error_threshold=self.error_threshold[lang_a][lang_b], proto_loss_multiplier=0.0001)
                    print("Proto params")
                    print(self.proto_params[lang_b])
                    self.updates[lang_a][lang_b] = self._compute_updates(output_layer=self.l_out[lang_a][lang_b], loss=self.loss[lang_a][lang_b][0], params=self.proto_params[lang_b])
            
            # Theano functions for training and computing loss
            print("Compiling functions ...")
            if lang_b in self.langs:
                self.train_func[lang_a][lang_b], self.loss_func[lang_a][lang_b], self.predict_func[lang_a][lang_b], _ = self._compile_functions(X_input=self.l_in_X[lang_a].input_var, Y_input=self.l_target_Y[lang_b].input_var, mask=self.l_mask_X[lang_a].input_var, error_threshold=self.error_threshold[lang_a][lang_b], loss=self.loss[lang_a][lang_b], updates=self.updates[lang_a][lang_b], pred_values_determ=self.predicted_values_deterministic[lang_a][lang_b], context_vector=None)
            if self.train_proto:
                self.train_func[lang_a][lang_b], self.loss_func[lang_a][lang_b], self.predict_func[lang_a][lang_b], _ = self._compile_functions(X_input=self.l_in_X[lang_a].input_var, Y_input=None, mask=self.l_mask_X[lang_a].input_var, error_threshold=self.error_threshold[lang_a][lang_b], loss=self.loss[lang_a][lang_b], updates=self.updates[lang_a][lang_b], pred_values_determ=self.predicted_values_deterministic[lang_a][lang_b], context_vector=None)
    
    # This method takes a network that goes from lang A to lang B as input,
    # adds a decoder, fetches the output, and stores it in variables specific
    # for lang_a->lang_b
    def create_decoder_output(self, lang_a, lang_b, max_len=None, create_target=True):
        if max_len is None:
            max_len = self.max_len[lang_b]
        gate_dec = self.gate[lang_b]["dec"]
        # Conversion layer from n_dense to n_hidden nodes
        decoder_input_hid = DenseLayer(self.network[lang_a][lang_b], num_units=self.n_hidden, W=self.w_phyl_dec_hid[lang_b])
        self.proto_params[lang_b] += decoder_input_hid.get_params()
        
        # Decoder
        l_decoder = self._create_decoder(decoder_input=self.network[lang_a][lang_b], decoder_input_hid=decoder_input_hid, max_len=max_len, n_features_context=self.n_units_phyl, resetgate=gate_dec["reset"]["bw"], updategate=gate_dec["update"]["bw"], hiddengate=gate_dec["hidden"]["bw"])
        self.proto_params[lang_b] += l_decoder.get_params()
        #### Reshape before dense layers
        l_reshape = ReshapeLayer(l_decoder, (self.batch_size * max_len, self.n_hidden), name="reshape_dec_dense")
        
        # Output layer
        # Phonetic: sigmoid, multi-label classification
        if self.output_encoding == "phonetic" or self.output_encoding == "embedding":
            output_nonlinearity = lasagne.nonlinearities.sigmoid
        # Character: softmax, single-label classification
        elif self.output_encoding == "character":
            output_nonlinearity = lasagne.nonlinearities.softmax
        self.l_out[lang_a][lang_b] = DenseLayer(l_reshape, num_units=self.voc_size[1], nonlinearity=output_nonlinearity, W=self.w_out[lang_b], name="dense_out")
        self.proto_params[lang_b] += self.l_out[lang_a][lang_b].get_params()
        
        # Fetch network output
        # Deterministic (without dropout) for prediction
        self.predicted_values[lang_a][lang_b] = lasagne.layers.get_output(self.l_out[lang_a][lang_b])
        self.predicted_values_deterministic[lang_a][lang_b] = lasagne.layers.get_output(self.l_out[lang_a][lang_b], deterministic=True)
        # self.context_vector[lang_a] = lasagne.layers.get_output(self.l_encoder[lang_a], deterministic=True)
        if create_target is True:
            # Do reshape on target values (actually build small network), and get output
            self.target_values[lang_a][lang_b] = lasagne.layers.get_output(ReshapeLayer(self.l_target_Y[lang_b], (self.batch_size * max_len, self.voc_size[1])))
        
        # Array storing the prediction errors, initialized with 10 as first error
        self.prediction_errors[lang_a][lang_b] = [10]
        self.error_threshold[lang_a][lang_b] = T.scalar(name='error_threshold')
    
    def build_network(self, tree):
        all_langs = []
        descendants = tree.descendants
        print(descendants)
        # If tree has descendants
        if len(descendants) > 0:
            # Start merging something already??
            # l_encoder = DenseLayer(l_concat, num_units=self.n_hidden, nonlinearity=lasagne.nonlinearities.rectify)
            # Recursion: descendants are given to same function
            for child in descendants:
                all_langs += self.build_network(child)
            return all_langs
        # Else: leaf case
        else:
            return [tree.name]
    
    # This method trains all language pairs at once
    def train_all(self, trainset, valset, n_epochs):
        n_val_batches = defaultdict(lambda: defaultdict(int))
        # n_batches_history = defaultdict(lambda: defaultdict(int))
        epochs_completed = defaultdict(lambda: defaultdict(int))
        prev_epochs_completed = defaultdict(lambda: defaultdict(int))
        plot_losses = defaultdict(lambda : defaultdict(list))
        plot_distances = defaultdict(lambda : defaultdict(list))
        for lang_a, lang_b in self.lang_pairs_proto:
            n_val_batches[lang_a][lang_b] = valset[(lang_a, lang_b)].get_size() // self.batch_size
            # To compute mean error, look back at last processed 10% of training set
            # n_batches_history[lang_a][lang_b] = trainset[(lang_a,lang_b)].get_size() // self.batch_size
        print("Training ...")
        if self.train_proto:
            pairs = self.lang_pairs_proto
        else:
            pairs = self.lang_pairs
        start_time = time.time()
        while (np.any([epochs_completed[lang_a][lang_b] < n_epochs for lang_a, lang_b in pairs])):
            # Per epoch, go through every lang pair
            for lang_a, lang_b in pairs:
                X, X_unnorm, Y, m_X = trainset[(lang_a, lang_b)].get_batch()
                lr = self.learning_rate * (self.learning_rate_decay ** epochs_completed[lang_a][lang_b])
                if self.adaptive_learning_rate > 0.0:
                    # Adapt learning rate to X,Y edit distance:
                    # learn more from probable cognates
                    dist = self._compute_distance_batch_encoded(X_unnorm, Y, max_len_tar=self.max_len[lang_b], voc_size_tar=self.voc_size[1], conversion_key=self.conversion_key)
                    lr = lr - self.adaptive_learning_rate * dist
                
                # Prediction error threshold is mean+1 standard deviation
                error_threshold = np.mean(self.prediction_errors[lang_a][lang_b]) + self.cognacy_prior * np.std(self.prediction_errors[lang_a][lang_b])
                
                # Perform training for lang_a->lang_b
                if lang_b in self.proto_languages:
                    # Skip target values when in proto-languages
                    loss, cognacy_prior, prediction_error = self.train_func[lang_a][lang_b](X, m_X, error_threshold, lr)
                else:
                    loss, cognacy_prior, prediction_error = self.train_func[lang_a][lang_b](X, Y, m_X, error_threshold, lr)
                # Now update the prediction_errors variable, to compute mean for next round
                self.prediction_errors[lang_a][lang_b].append(prediction_error)

                # Calculate validation loss when new epoch is entered
                prev_epochs_completed[lang_a][lang_b] = epochs_completed[lang_a][lang_b]
                epochs_completed[lang_a][lang_b] = trainset[(lang_a, lang_b)].get_epochs_completed()
                if prev_epochs_completed[lang_a][lang_b] < epochs_completed[lang_a][lang_b]:
                    loss_batches = []
                    distance_batches = []
                    for _ in np.arange(n_val_batches[lang_a][lang_b]):
                        X_val, X_val_unnorm, Y_val, mask_X_val = valset[(lang_a, lang_b)].get_batch(val=True)
                        # Calculate and store loss
                        # Prediction error threshold is mean+1 standard deviation
                        error_threshold = np.mean(self.prediction_errors[lang_a][lang_b]) + self.cognacy_prior * np.std(self.prediction_errors[lang_a][lang_b])
                        if lang_b in self.proto_languages:
                            loss, cognacy_prior, prediction_error, mean_error, predictions = self.loss_func[lang_a][lang_b](X_val, mask_X_val, error_threshold)
                        else:
                            loss, cognacy_prior, prediction_error, mean_error, predictions = self.loss_func[lang_a][lang_b](X_val, Y_val, mask_X_val, error_threshold)
                        if self.cognacy_prior > 0.0:
                            print(cognacy_prior, prediction_error, mean_error, np.mean(self.prediction_errors[lang_a][lang_b]))
                        loss_batches.append(loss)
                        if lang_b not in self.proto_languages:
                            # Calculate and store distance
                            _, _, _, distances, _ = self._compute_distance_batch_encoded(X_val_unnorm, Y_val, max_len_tar=self.max_len[lang_b], voc_size_tar=self.voc_size[1], conversion_key=self.conversion_key, predictions=predictions)
                            distance_batches += distances
                        
                    duration = time.time() - start_time
                    avg_loss = np.average(loss_batches)
                    avg_distance = np.average(distance_batches)
                    plot_losses[lang_a][lang_b].append((epochs_completed[lang_a][lang_b], avg_loss))
                    plot_distances[lang_a][lang_b].append((epochs_completed[lang_a][lang_b], avg_distance))
                    print(lang_a + "-" + lang_b + ": epoch {0:2} validation loss = {1:.2f}, distance = {2:.2f}, duration = {3:.2f}".format(epochs_completed[lang_a][lang_b], avg_loss, avg_distance, duration))
                    start_time = time.time()
        
        return plot_losses, plot_distances
    
    def create_phyl_layer(self, input_layer, weight):
        layer = DenseLayer(input_layer, num_units=self.n_units_phyl, W=weight)
        if self.dropout > 0.0:
            layer = DropoutLayer(layer, p=self.dropout, rescale=False)
        return layer
    
    def predict_proto(self, testset, max_len_tar, voc_size_tar, conversion_key, predict_func, print_output=True):
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
            if print_output:
                text_output += template.format("".join(input_word), "".join(target_word), "".join(predicted_word), dist_t_p) + "\n"
            row_dict["INPUT"].append(" ".join(input_word))
            # row_dict["TARGET"].append(" ".join(target_word))
            row_dict["PREDICTION"].append(" ".join(predicted_word))
            # row_dict["DISTANCE_T_P"].append(dist_t_p)
            # row_dict["DISTANCE_S_T"].append(dist_s_t)
            
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
    
    def get_proto_languages(self):
        return self.proto_languages
    
    def get_predict_func(self, lang_a, lang_b):
        return self.predict_func[lang_a][lang_b]
