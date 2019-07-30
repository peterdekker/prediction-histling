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
import models.baseline as baseline
from prediction import prediction

from cognatedetection import cd
from tree import cluster

from dataset import data
from util import utility
from visualize import visualize


# External libs
import lingpy
import argparse
import numpy as np
import os
import copy
import pickle
import pandas as pd

# ## Command line arguments
# Number of units in the hidden (recurrent) layer
N_HIDDEN = 400
N_UNITS_PHYL = 400
N_LAYERS_ENCODER = 1
N_LAYERS_DECODER = 1
DROPOUT = 0.1
BIDIRECTIONAL_ENCODER = True
BIDIRECTIONAL_DECODER = False
ENCODER_ALL_STEPS = False
# Number of training sequences in each batch
BATCH_SIZE = 10
# Optimization learning rate
LEARNING_RATE = 0.01
REG_WEIGHT = 0.0
INITIALIZATION = "xavier_normal"
OPTIMIZER = "adagrad"
# All gradients above this will be clipped
GRAD_CLIP = 100
# Number of epochs to train the net
N_EPOCHS = 15
GATED_LAYER_TYPE = "gru"
N_LAYERS_DENSE = 1
PREDICTION = False
SEQ = False
PHYL = False
N_ITER_SEQ = 100
CLUSTER = False
VISUALIZE = False
VISUALIZE_WEIGHTS = False
VISUALIZE_ENCODING = False
BASELINE = False
BASELINE_CLUSTER = False
COGNATE_DETECTION = False
TUNE_CD = False
# TUNE_SOURCE_CD = False
SHOW_N_COG = False
INPUT_TYPE = "asjp"
INPUT_ENCODING = "character"
OUTPUT_ENCODING = "character"  # fixed, there is no cmd line option for this
ENCODER_DECODER_HID_INIT = False
VALIDATION = False
MEAN_SUBTRACTION = False
NO_STANDARDIZATION = False
LEARNING_RATE_DECAY = 1.0
ADAPTIVE_LR = 0.0
COGNACY_PRIOR = 1.0
FILTER_TRAIN = 1.0
EXPORT_WEIGHTS = False
TRAIN_CORPUS = "northeuralex"
VALTEST_CORPUS = "northeuralex"
TRAIN_PROTO = False

# ## Other arguments
RESULTS_DIR = "output"
FEATURE_FILE = "data/asjp_phonetic_features_new.tsv"
LANGUAGES = ["nld", "deu"]
LANG_FAMILY = "none"
SAMPLE_LANG_PAIRS = None

LANG_FAMILIES_DICT = {
"slav": ["ces", "bul", "rus", "bel", "ukr", "pol", "slk", "slv", "hrv"],
"ger": ["swe", "isl", "eng", "nld", "deu", "dan", "nor"]
}


def main():
    # Set LingPy input encoding (IPA or ASJP)
    lingpy.settings.rc(schema=FLAGS.input_type)
    
    if not os.path.exists(RESULTS_DIR):
        os.mkdir(RESULTS_DIR)
    options = utility.create_option_string(FLAGS)
    
    # Set variables for train corpus
    if FLAGS.train_corpus == "northeuralex":
        input_path_train = "data/northeuralex-0.9-lingpy.tsv"
    elif FLAGS.train_corpus == "ielex" or FLAGS.train_corpus == "ielex-corr":
        input_path_train = "data/ielex-4-26-2016.csv"
    tsv_path_train = FLAGS.train_corpus + "-" + FLAGS.input_type
    tsv_cognates_path_train = tsv_path_train + "-cognates"
    
    # Set variables for val/test corpus
    if FLAGS.valtest_corpus == "northeuralex":
        input_path_valtest = "data/northeuralex-0.9-lingpy.tsv"
    elif FLAGS.valtest_corpus == "ielex" or FLAGS.valtest_corpus == "ielex-corr":
        input_path_valtest = "data/ielex-4-26-2016.csv"
    tsv_path_valtest = FLAGS.valtest_corpus + "-" + FLAGS.input_type
    tsv_cognates_path_valtest = tsv_path_valtest + "-cognates"
    
    intersection_path = "data/ielex-northeuralex-0.9-intersection.tsv"
    
    distances_path = utility.get_distances_path(RESULTS_DIR, options)
    baselines_path = utility.get_baselines_path(RESULTS_DIR, options)
    
    if "all" in FLAGS.languages:
        print("Get all langs")
        FLAGS.languages = data.get_all_languages(input_path_train, FLAGS.train_corpus)
    
    print("Generating all language pairs...")
    feature_matrix_phon = None
    if FLAGS.input_encoding == "phonetic" or FLAGS.visualize_weights or FLAGS.visualize_encoding:
        feature_matrix_phon = data.load_feature_file(FEATURE_FILE)
    perms = False
    if len(FLAGS.languages) > 2:
        perms = True
    lang_pairs = utility.generate_pairs(FLAGS.languages, allow_permutations=perms, sample=SAMPLE_LANG_PAIRS)
    
    if FLAGS.prediction or FLAGS.visualize or FLAGS.baseline or FLAGS.show_n_cog:
        print("Training corpus:")
        print(" - Convert wordlists to tsv format, and tokenize words.")
        data.convert_wordlist_tsv(input_path_train, source=FLAGS.train_corpus, input_type=FLAGS.input_type, output_path=tsv_path_train + ".tsv", intersection_path=intersection_path)
        print(" - Detect cognates in entire dataset using LexStat.")
        cd.cognate_detection_lexstat(tsv_path_train, tsv_cognates_path_train, input_type=FLAGS.input_type)
        
        print("Validation/test corpus:")
        print(" - Convert wordlists to tsv format, and tokenize words.")
        data.convert_wordlist_tsv(input_path_valtest, source=FLAGS.valtest_corpus, input_type=FLAGS.input_type, output_path=tsv_path_valtest + ".tsv", intersection_path=intersection_path)
        print(" - Fetch list of concepts (only for valtest corpus)")
        concepts_valtest = data.fetch_concepts(input_path_valtest, source=FLAGS.valtest_corpus)
        print(" - Detect cognates in entire dataset using LexStat.")
        cd.cognate_detection_lexstat(tsv_path_valtest, tsv_cognates_path_valtest, input_type=FLAGS.input_type)
        
        excluded_concepts_training = []
        if FLAGS.train_corpus != FLAGS.valtest_corpus:
            print("Loading IElex->NorthEuraLex concept mapping...")
            ielex_nelex_map = data.load_ielex_nelex_concept_mapping("data/ielex-nelex-mapping.csv")
            # All concepts in the validation/test corpus should be excluded from the training corpus
            for concept in concepts_valtest:
                if concept in ielex_nelex_map:
                    concept_nelex = ielex_nelex_map[concept]
                    excluded_concepts_training.append(concept_nelex)
        
        if FLAGS.show_n_cog:
            print("Show number of cognates per language")
            cog_per_lang, cliques = data.compute_n_cognates(lang_pairs, tsv_cognates_path_train, langs=FLAGS.languages, cognates_threshold=100)
            print("Cognates per language: " + str(cog_per_lang))
            print("Number of cliques: " + str(cliques))

    # ## Language pair-specific part
    if FLAGS.prediction or FLAGS.visualize or FLAGS.visualize_weights or FLAGS.baseline:
        # Language-pair specific variables: every dict entry is designated for a specific lang pair
        results_path = {}
        subs_sp_path = {}
        subs_st_path = {}
        context_vectors_path = {}
        
        train = {}
        val = {}
        test = {}
        conversion_key = {}
        features_lp = {}
        
        # Max_len saved per language, rather than per language pair
        max_len = {}

        features = [pd.DataFrame(), pd.DataFrame()]
        voc_size = [0, 0]
        voc_size_general = [0, 0]
        
        if FLAGS.phyl:
            # For phylogenetic word prediction, create one feature matrix for all languages
            print("Create feature matrix for all language pairs.")
            used_tokens = [[], []]
            tokens_set = [[], []]
            for lang_pair in lang_pairs:
                # For phylogenetic word prediction, create one feature matrix for all languages
                features_lp[lang_pair], max_len[lang_pair[0]], max_len[lang_pair[1]], _, _ = data.get_corpus_info([tsv_cognates_path_train + ".tsv", tsv_cognates_path_valtest + ".tsv"], lang_pair=lang_pair, input_encoding=FLAGS.input_encoding, output_encoding=OUTPUT_ENCODING, feature_matrix_phon=feature_matrix_phon)
                used_tokens[0] += list(features_lp[lang_pair][0].index)
                used_tokens[1] += list(features_lp[lang_pair][1].index)
            
            tokens_set[0] = list(set(used_tokens[0]))
            tokens_set[1] = list(set(used_tokens[1]))
            if FLAGS.input_encoding == "character":
                features[0] = data.create_one_hot_matrix(tokens_set[0])
            elif FLAGS.input_encoding == "phonetic":
                features[0] = feature_matrix_phon.loc[tokens_set[0]]
            else:
                print("Embedding encoding not possible in phylogenetic tree prediction.")
                return
            # Output encoding is always character
            features[1] = data.create_one_hot_matrix(tokens_set[1])
            voc_size_general[0] = features[0].shape[1]
            voc_size_general[1] = features[1].shape[1]
            conversion_key_general = data.create_conversion_key(features)
        
        # Set batch size to 1 for weight visualization:
        # we want to feed individual words through the network
        if FLAGS.export_weights:
            FLAGS.batch_size = 1
        
        plot_path_phyl = utility.create_path(RESULTS_DIR, options, prefix="plot_")
        for lang_pair in lang_pairs:
            lang_a, lang_b = lang_pair
            context_vectors_path[lang_pair] = utility.create_path(RESULTS_DIR, options, prefix="context_vectors_", lang_a=lang_a, lang_b=lang_b)
            # Create export path, containing all options
            # This is used to output a prediction results file, which can then be used for visualization and cognate detection
            results_path[lang_pair] = utility.get_results_path(lang_a, lang_b, RESULTS_DIR, options)
            subs_st_path[lang_pair] = utility.create_path(RESULTS_DIR, options, prefix="subs_st_", lang_a=lang_a, lang_b=lang_b)
            subs_sp_path[lang_pair] = utility.create_path(RESULTS_DIR, options, prefix="subs_sp_", lang_a=lang_a, lang_b=lang_b)
            
            if FLAGS.prediction or FLAGS.baseline:
                # If data in pickle, load pickle
                data_pickle = results_path[lang_pair] + "-data.p"
                if os.path.exists(data_pickle):
                    with open(data_pickle, "rb") as f:
                        print("Loading train/val/test sets from pickle, nothing generated.")
                        train[lang_pair], val[lang_pair], test[lang_pair], conversion_key[lang_pair], max_len[lang_pair[0]], max_len[lang_pair[1]], voc_size[0], voc_size[1] = pickle.load(f)
                else:
                    # For phylogenetic word prediction, we have a language-independent feature matrix
                    if not FLAGS.phyl:
                        print("Create feature matrix for this specific language pair.")
                        features, max_len[lang_pair[0]], max_len[lang_pair[1]], voc_size[0], voc_size[1] = data.get_corpus_info([tsv_cognates_path_train + ".tsv", tsv_cognates_path_valtest + ".tsv"], lang_pair=lang_pair, input_encoding=FLAGS.input_encoding, output_encoding=OUTPUT_ENCODING, feature_matrix_phon=feature_matrix_phon)
                        conversion_key[lang_pair] = data.create_conversion_key(features)
                    else:
                        conversion_key[lang_pair] = conversion_key_general
                        voc_size = voc_size_general
                    # In phylogenetic mode, we created one feature matrix for all languages
                        
                    print("Convert training corpus TSV file to data matrix")
                    dataset_train, train_mean, train_std = data.create_data_matrix(tsv_path=tsv_cognates_path_train + ".tsv", lang_pair=(lang_a, lang_b), features=features, max_len=(max_len[lang_pair[0]], max_len[lang_pair[1]]), voc_size=voc_size, batch_size=FLAGS.batch_size, mean_subtraction=FLAGS.mean_subtraction, feature_standardization=not FLAGS.no_standardization, excluded_concepts=excluded_concepts_training, cognate_detection=FLAGS.cognate_detection)
                    
                    print("Convert val/test corpus TSV file to data matrix")
                    dataset_valtest, _, _ = data.create_data_matrix(tsv_path=tsv_cognates_path_valtest + ".tsv", lang_pair=(lang_a, lang_b), features=features, max_len=(max_len[lang_pair[0]], max_len[lang_pair[1]]), voc_size=voc_size, batch_size=FLAGS.batch_size, mean_subtraction=FLAGS.mean_subtraction, feature_standardization=not FLAGS.no_standardization, cognate_detection=FLAGS.cognate_detection, valtest=True, train_mean=train_mean, train_std=train_std)
                    
                    t_set_size = dataset_train.get_size()
                    vt_set_size = dataset_valtest.get_size()
                    
                    if FLAGS.valtest_corpus == FLAGS.train_corpus:
                        # If train and valtest corpus the same, divide one corpus into parts
                        assert t_set_size == vt_set_size
                        n_train, n_val, n_test = dataset_train.compute_subset_sizes(t_set_size)
                    else:
                        # If train and valtest corpus different, use full train corpus as train and
                        # full valtest corpus for validation and testing
                        # TODO: In fact this is not needed, we can directly take set size.
                        n_train, _, _ = dataset_train.compute_subset_sizes(t_set_size, only_train=True)
                        _, n_val, n_test = dataset_valtest.compute_subset_sizes(vt_set_size, only_valtest=True)
                    
                    print("Divide into training, validation and test set.")
                    # Even if train and valtest corpus are the same, we do this separately,
                    # because valtest corpus is filtered on cognates and train corpus is not
                    # Use train corpus only for train set
                    train[lang_pair], _, _ = dataset_train.divide_subsets(n_train, 0, 0)
                    # Use val/test corpus for validation and test set
                    _, val[lang_pair], test[lang_pair] = dataset_valtest.divide_subsets(0, n_val, n_test)
                
                    if not FLAGS.cognate_detection:
                        print("Filter val/test sets on cognates.")
                        # Use only cognate pairs for validation and test
                        val[lang_pair] = val[lang_pair].filter_cognates()
                        test[lang_pair] = test[lang_pair].filter_cognates()
                        print("Val/test sizes after cognate filtering: " + str(val[lang_pair].get_size()) + "|" + str(test[lang_pair].get_size()))
                    
                    # Pickle train/val/test/sets
                    with open(data_pickle, "wb") as f:
                        pickle.dump((train[lang_pair], val[lang_pair], test[lang_pair], conversion_key[lang_pair], max_len[lang_pair[0]], max_len[lang_pair[1]], voc_size[0], voc_size[1]), f)
                
            if FLAGS.prediction and not FLAGS.seq and not FLAGS.phyl:
                print("Performing word prediction for pair (" + lang_a + ", " + lang_b + ")")
                prediction.word_prediction(lang_a, lang_b, (max_len[lang_pair[0]], max_len[lang_pair[1]]), train[lang_pair], val[lang_pair], test[lang_pair], conversion_key[lang_pair], voc_size, results_path[lang_pair], distances_path + ".txt", context_vectors_path[lang_pair] + ".p")
            if FLAGS.prediction and FLAGS.seq and not FLAGS.phyl:
                print("Performing SeqModel word prediction for pair (" + lang_a + ", " + lang_b + ")")
                prediction.word_prediction_seq(lang_a, lang_b, train[lang_pair], val[lang_pair], test[lang_pair], conversion_key[lang_pair], results_path[lang_pair], distances_path + ".txt")
            if FLAGS.baseline and FLAGS.input_type == "asjp":
                print("Performing baseline results for pair(" + lang_a + ", " + lang_b + ")")
                sounds = (list(features[0].index), list(features[1].index))
                training_frame = train[lang_pair].get_dataframe(conversion_key[lang_pair], FLAGS.input_encoding, OUTPUT_ENCODING)
                testing_frame = test[lang_pair].get_dataframe(conversion_key[lang_pair], FLAGS.input_encoding, OUTPUT_ENCODING)
                baseline.compute_baseline(lang_a, lang_b, sounds, training_frame, testing_frame, baselines_path + ".txt")
            if FLAGS.visualize:
                print("Inferring sound correspondences...")
                visualize.show_output_substitutions(results_path[lang_pair], subs_st_path[lang_pair], subs_sp_path[lang_pair])
            if FLAGS.visualize_weights:
                visualize.visualize_weights(context_vectors_path[lang_pair], lang_pair, FLAGS.input_encoding, OUTPUT_ENCODING, RESULTS_DIR, sample=None)
    for lang_pair in lang_pairs:
        if FLAGS.visualize_encoding:
            # Create embedding for first languages
            emb_matrix = data.create_embedding(lang_pair[0], [tsv_cognates_path_train + ".tsv", tsv_cognates_path_valtest + ".tsv"])
            visualize.visualize_encoding(emb_matrix, feature_matrix_phon, lang_pair, RESULTS_DIR)
    if FLAGS.cluster:
        # Cluster based on word prediction distances
        print("WP TREE:\n")
        cluster.cluster_languages(lang_pairs, distances_path, output_path=distances_path)
    if FLAGS.baseline_cluster:
        # Source prediction baseline
        print("\nSOURCE BASELINE TREE")
        cluster.cluster_languages(lang_pairs, baselines_path, output_path=baselines_path + "_source", distance_col=2)
        # PMI-based baseline
        print("\nPMI BASELINE TREE")
        cluster.cluster_languages(lang_pairs, baselines_path, output_path=baselines_path + "_pmi", distance_col=3)
    if FLAGS.cognate_detection:
        print("Performing WP cognate detection using clustering...")
        results_table = cd.cognate_detection_cluster(lang_pairs, RESULTS_DIR, options, use_distance="prediction")
        print(results_table)

    # Phylogenetic word prediction comes after datasets have been generated for
    # all language pairs. All language pairs are then taken into account at once
    # by phylogenetic word prediction
    if FLAGS.prediction and FLAGS.phyl and not FLAGS.seq:
        FLAGS.export_weights = False  # Turn off export of weights
        print("Performing phylogenetic word prediction")
        tree_string = "((nld,deu),eng)"  # unused at the moment
        if len(FLAGS.languages) >= 3:
            results_path_proto = utility.create_path(RESULTS_DIR, options, prefix="proto_")  # lang-pair independent path
            prediction.word_prediction_phyl(FLAGS.languages, lang_pairs, tree_string, max_len, train, val, test, conversion_key_general, voc_size, results_path, results_path_proto, distances_path + ".txt", context_vectors_path, plot_path_phyl)
        else:
            print("Please supply 3 languages, the first 2 being more closely related than the last.")
    
    
def print_flags(FLAGS):
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


if __name__ == "__main__":
    if __name__ == "__main__":
        # Command line arguments
        parser = argparse.ArgumentParser()
        # Mode
        parser.add_argument('--prediction', action='store_true', default=PREDICTION)
        parser.add_argument('--seq', action='store_true', default=SEQ)
        parser.add_argument('--phyl', action='store_true', default=PHYL)
        parser.add_argument('--baseline', action='store_true', default=BASELINE)
        parser.add_argument('--cluster', action='store_true', default=CLUSTER)
        parser.add_argument('--baseline_cluster', action='store_true', default=BASELINE_CLUSTER)
        parser.add_argument('--visualize', action='store_true', default=VISUALIZE)
        parser.add_argument('--visualize_weights', action='store_true', default=VISUALIZE_WEIGHTS)
        parser.add_argument('--visualize_encoding', action='store_true', default=VISUALIZE_ENCODING)
        parser.add_argument('--cognate_detection', action='store_true', default=COGNATE_DETECTION)
        # parser.add_argument('--tune_cd', action='store_true', default=TUNE_CD)
        # parser.add_argument('--tune_source_cd', action='store_true', default=TUNE_SOURCE_CD)
        parser.add_argument('--show_n_cog', action='store_true', default=SHOW_N_COG)
        
        # SeqModel options
        parser.add_argument('--n_iter_seq', type=int, default=N_ITER_SEQ)
        
        # Workflow options
        parser.add_argument('--languages', nargs="+", default=LANGUAGES)
        parser.add_argument('--lang_family', default=LANG_FAMILY, choices=["none", "slav", "ger"])
        parser.add_argument('--filter_train', type=float, default=FILTER_TRAIN, help="Filter train set for cognacy, based on prediction results, and rerun")
        parser.add_argument('--input_type', default=INPUT_TYPE, choices=["asjp", "ipa"])
        parser.add_argument('--train_corpus', default=TRAIN_CORPUS, choices=["northeuralex", "ielex", "ielex-corr"])
        parser.add_argument('--valtest_corpus', default=VALTEST_CORPUS, choices=["northeuralex", "ielex", "ielex-corr"])
        parser.add_argument('--input_encoding', default=INPUT_ENCODING, choices=["phonetic", "character", "embedding"])
        parser.add_argument('--validation', action='store_true', default=VALIDATION, help="test on validation set instead of test set")
        
        # Neural network options
        parser.add_argument('--hidden', type=int, default=N_HIDDEN)
        parser.add_argument('--units_phyl', type=int, default=N_UNITS_PHYL)
        parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
        parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE)
        parser.add_argument('--lr_decay', type=float, default=LEARNING_RATE_DECAY)
        parser.add_argument('--adaptive_lr', type=float, default=ADAPTIVE_LR)
        parser.add_argument('--reg_weight', type=float, default=REG_WEIGHT)
        parser.add_argument('--grad_clip', type=float, default=GRAD_CLIP)
        parser.add_argument('--n_epochs', type=int, default=N_EPOCHS)
        parser.add_argument('--layers_encoder', type=int, default=N_LAYERS_ENCODER)
        parser.add_argument('--layers_decoder', type=int, default=N_LAYERS_DECODER)
        parser.add_argument('--layers_dense', type=int, default=N_LAYERS_DENSE)
        parser.add_argument('--dropout', type=float, default=DROPOUT)
        parser.add_argument('--optimizer', default=OPTIMIZER, choices=["adagrad", "adam", "sgd"])
        parser.add_argument('--no_bidirectional_encoder', action='store_true', default=not BIDIRECTIONAL_ENCODER)
        parser.add_argument('--bidirectional_decoder', action='store_true', default=BIDIRECTIONAL_DECODER)
        parser.add_argument('--encoder_all_steps', action='store_true', default=ENCODER_ALL_STEPS)
        parser.add_argument('--init', default=INITIALIZATION, choices=["constant", "xavier_normal", "xavier_uniform"])
        parser.add_argument('--gated_layer_type', default=GATED_LAYER_TYPE, choices=["lstm", "gru"])
        parser.add_argument('--mean_subtraction', action='store_true', default=MEAN_SUBTRACTION)
        parser.add_argument('--no_standardization', action='store_true', default=NO_STANDARDIZATION)
        parser.add_argument('--cognacy_prior', type=float, default=COGNACY_PRIOR)
        parser.add_argument('--export_weights', action='store_true', default=EXPORT_WEIGHTS)
        parser.add_argument('--train_proto', action='store_true', default=TRAIN_PROTO)
        FLAGS, unparsed = parser.parse_known_args()
        
        if FLAGS.lang_family is not "none":
            FLAGS.languages = LANG_FAMILIES_DICT[FLAGS.lang_family]
        print_flags(FLAGS)
        main()
