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
import sys
sys.path.append(".")

from models import baseline
from prediction import prediction

from cognatedetection import cd
from tree import cluster
from dataset import data
from util import utility
from util.config import config
from visualize import visualize


# External libs
import pickle
import lingpy
import numpy as np
import os
import pandas as pd


def main():
    # Set LingPy input encoding (IPA or ASJP)
    lingpy.settings.rc(schema=config["input_type"])
    
    if not os.path.exists(config["results_dir"]):
        os.mkdir(config["results_dir"])
    options = utility.create_option_string(config)
    
    # Set variables for train corpus
    if config["train_corpus"] == "northeuralex":
        input_path_train = "data/northeuralex-0.9-lingpy.tsv"
    elif config["train_corpus"] == "ielex" or config["train_corpus"] == "ielex-corr":
        input_path_train = "data/ielex-4-26-2016.csv"
    tsv_path_train = config["train_corpus"] + "-" + config["input_type"]
    tsv_cognates_path_train = tsv_path_train + "-cognates"
    
    # Set variables for val/test corpus
    if config["valtest_corpus"] == "northeuralex":
        input_path_valtest = "data/northeuralex-0.9-lingpy.tsv"
    elif config["valtest_corpus"] == "ielex" or config["valtest_corpus"] == "ielex-corr":
        input_path_valtest = "data/ielex-4-26-2016.csv"
    tsv_path_valtest = config["valtest_corpus"] + "-" + config["input_type"]
    tsv_cognates_path_valtest = tsv_path_valtest + "-cognates"
    
    intersection_path = "data/ielex-northeuralex-0.9-intersection.tsv"
    
    distances_path = utility.get_distances_path(config["results_dir"], options)
    baselines_path = utility.get_baselines_path(config["results_dir"], options)
    
    if "all" in config["languages"]:
        print("Get all langs")
        config["languages"] = data.get_all_languages(input_path_train, config["train_corpus"])
    
    print("Generating all language pairs...")
    feature_matrix_phon = None
    if config["input_encoding"] == "phonetic" or config["visualize_weights"] or config["visualize_encoding"]:
        feature_matrix_phon = data.load_feature_file(config["feature_file"])
    perms = False
    if len(config["languages"]) > 2:
        perms = True
    lang_pairs = utility.generate_pairs(config["languages"], allow_permutations=perms, sample=config["sample_lang_pairs"])
    
    if config["prediction"] or config["visualize"] or config["baseline"] or config["show_n_cog"]:
        print("Training corpus:")
        print(" - Convert wordlists to tsv format, and tokenize words.")
        data.convert_wordlist_tsv(input_path_train, source=config["train_corpus"], input_type=config["input_type"], output_path=tsv_path_train + ".tsv", intersection_path=intersection_path)
        print(" - Detect cognates in entire dataset using LexStat.")
        cd.cognate_detection_lexstat(tsv_path_train, tsv_cognates_path_train, input_type=config["input_type"])
        
        print("Validation/test corpus:")
        print(" - Convert wordlists to tsv format, and tokenize words.")
        data.convert_wordlist_tsv(input_path_valtest, source=config["valtest_corpus"], input_type=config["input_type"], output_path=tsv_path_valtest + ".tsv", intersection_path=intersection_path)
        print(" - Fetch list of concepts (only for valtest corpus)")
        concepts_valtest = data.fetch_concepts(input_path_valtest, source=config["valtest_corpus"])
        print(" - Detect cognates in entire dataset using LexStat.")
        cd.cognate_detection_lexstat(tsv_path_valtest, tsv_cognates_path_valtest, input_type=config["input_type"])
        
        excluded_concepts_training = []
        if config["train_corpus"] != config["valtest_corpus"]:
            print("Loading IElex->NorthEuraLex concept mapping...")
            ielex_nelex_map = data.load_ielex_nelex_concept_mapping("data/ielex-nelex-mapping.csv")
            # All concepts in the validation/test corpus should be excluded from the training corpus
            for concept in concepts_valtest:
                if concept in ielex_nelex_map:
                    concept_nelex = ielex_nelex_map[concept]
                    excluded_concepts_training.append(concept_nelex)
        
        if config["show_n_cog"]:
            print("Show number of cognates per language")
            cog_per_lang, cliques = data.compute_n_cognates(lang_pairs, tsv_cognates_path_train, langs=config["languages"], cognates_threshold=100)
            print("Cognates per language: " + str(cog_per_lang))
            print("Number of cliques: " + str(cliques))

    # ## Language pair-specific part
    if config["prediction"] or config["visualize"] or config["visualize_weights"] or config["baseline"]:
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
        
        if config["phyl"]:
            # For phylogenetic word prediction, create one feature matrix for all languages
            print("Create feature matrix for all language pairs.")
            used_tokens = [[], []]
            tokens_set = [[], []]
            for lang_pair in lang_pairs:
                # For phylogenetic word prediction, create one feature matrix for all languages
                features_lp[lang_pair], max_len[lang_pair[0]], max_len[lang_pair[1]], _, _ = data.get_corpus_info([tsv_cognates_path_train + ".tsv", tsv_cognates_path_valtest + ".tsv"], lang_pair=lang_pair, input_encoding=config["input_encoding"], output_encoding=config["output_encoding"], feature_matrix_phon=feature_matrix_phon)
                used_tokens[0] += list(features_lp[lang_pair][0].index)
                used_tokens[1] += list(features_lp[lang_pair][1].index)
            
            tokens_set[0] = list(set(used_tokens[0]))
            tokens_set[1] = list(set(used_tokens[1]))
            if config["input_encoding"] == "character":
                features[0] = data.create_one_hot_matrix(tokens_set[0])
            elif config["input_encoding"] == "phonetic":
                features[0] = feature_matrix_phon.loc[tokens_set[0]]
            else:
                print("Embedding encoding not possible in phylogenetic tree prediction.")
                return
            # Output encoding is always character
            features[1] = data.create_one_hot_matrix(tokens_set[1])
            voc_size_general[0] = features[0].shape[1]
            voc_size_general[1] = features[1].shape[1]
            conversion_key_general = data.create_conversion_key(features)
            plot_path_phyl = utility.create_path(config["results_dir"], options, prefix="plot_")
        
        # Set batch size to 1 for weight visualization:
        # we want to feed individual words through the network
        if config["export_weights"]:
            config["batch_size"] = 1
        for lang_pair in lang_pairs:
            lang_a, lang_b = lang_pair
            context_vectors_path[lang_pair] = utility.create_path(config["results_dir"], options, prefix="context_vectors_", lang_a=lang_a, lang_b=lang_b)
            # Create export path, containing all options
            # This is used to output a prediction results file, which can then be used for visualization and cognate detection
            results_path[lang_pair] = utility.get_results_path(lang_a, lang_b, config["results_dir"], options)
            subs_st_path[lang_pair] = utility.create_path(config["results_dir"], options, prefix="subs_st_", lang_a=lang_a, lang_b=lang_b)
            subs_sp_path[lang_pair] = utility.create_path(config["results_dir"], options, prefix="subs_sp_", lang_a=lang_a, lang_b=lang_b)
            
            if config["prediction"] or config["baseline"]:
                # If data in pickle, load pickle
                data_pickle = results_path[lang_pair] + "-data.p"
                if os.path.exists(data_pickle):
                    with open(data_pickle, "rb") as f:
                        print("Loading train/val/test sets from pickle, nothing generated.")
                        train[lang_pair], val[lang_pair], test[lang_pair], conversion_key[lang_pair], max_len[lang_pair[0]], max_len[lang_pair[1]], voc_size[0], voc_size[1] = pickle.load(f)
                else:
                    # For phylogenetic word prediction, we have a language-independent feature matrix
                    if not config["phyl"]:
                        print("Create feature matrix for this specific language pair.")
                        features, max_len[lang_pair[0]], max_len[lang_pair[1]], voc_size[0], voc_size[1] = data.get_corpus_info([tsv_cognates_path_train + ".tsv", tsv_cognates_path_valtest + ".tsv"], lang_pair=lang_pair, input_encoding=config["input_encoding"], output_encoding=config["output_encoding"], feature_matrix_phon=feature_matrix_phon)
                        conversion_key[lang_pair] = data.create_conversion_key(features)
                    else:
                        # In phylogenetic mode, we created one feature matrix for all languages
                        conversion_key[lang_pair] = conversion_key_general
                        voc_size = voc_size_general
                    
                    print("Convert training corpus TSV file to data matrix")
                    dataset_train, train_mean, train_std = data.create_data_matrix(tsv_path=tsv_cognates_path_train + ".tsv", lang_pair=(lang_a, lang_b), features=features, max_len=(max_len[lang_pair[0]], max_len[lang_pair[1]]), voc_size=voc_size, batch_size=config["batch_size"], mean_subtraction=config["mean_subtraction"], feature_standardization=not config["no_standardization"], excluded_concepts=excluded_concepts_training, cognate_detection=config["cognate_detection"])
                    
                    print("Convert val/test corpus TSV file to data matrix")
                    dataset_valtest, _, _ = data.create_data_matrix(tsv_path=tsv_cognates_path_valtest + ".tsv", lang_pair=(lang_a, lang_b), features=features, max_len=(max_len[lang_pair[0]], max_len[lang_pair[1]]), voc_size=voc_size, batch_size=config["batch_size"], mean_subtraction=config["mean_subtraction"], feature_standardization=not config["no_standardization"], cognate_detection=config["cognate_detection"], valtest=True, train_mean=train_mean, train_std=train_std)
                    
                    t_set_size = dataset_train.get_size()
                    vt_set_size = dataset_valtest.get_size()
                    
                    if config["valtest_corpus"] == config["train_corpus"]:
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
                
                    if not config["cognate_detection"]:
                        print("Filter val/test sets on cognates.")
                        # Use only cognate pairs for validation and test
                        val[lang_pair] = val[lang_pair].filter_cognates()
                        test[lang_pair] = test[lang_pair].filter_cognates()
                        print("Val/test sizes after cognate filtering: " + str(val[lang_pair].get_size()) + "|" + str(test[lang_pair].get_size()))
                    
                    # Pickle train/val/test/sets
                    with open(data_pickle, "wb") as f:
                        pickle.dump((train[lang_pair], val[lang_pair], test[lang_pair], conversion_key[lang_pair], max_len[lang_pair[0]], max_len[lang_pair[1]], voc_size[0], voc_size[1]), f)
                
            if config["prediction"] and not config["seq"] and not config["phyl"]:
                print("Performing word prediction for pair (" + lang_a + ", " + lang_b + ")")
                prediction.word_prediction(lang_a, lang_b, (max_len[lang_pair[0]], max_len[lang_pair[1]]), train[lang_pair], val[lang_pair], test[lang_pair], conversion_key[lang_pair], voc_size, results_path[lang_pair], distances_path + ".txt", context_vectors_path[lang_pair] + ".p", config["output_encoding"], config)
            if config["prediction"] and config["seq"] and not config["phyl"]:
                print("Performing SeqModel word prediction for pair (" + lang_a + ", " + lang_b + ")")
                prediction.word_prediction_seq(lang_a, lang_b, train[lang_pair], val[lang_pair], test[lang_pair], conversion_key[lang_pair], results_path[lang_pair], distances_path + ".txt", config)
            if config["baseline"] and config["input_type"] == "asjp":
                print("Performing baseline results for pair(" + lang_a + ", " + lang_b + ")")
                sounds = (list(features[0].index), list(features[1].index))
                training_frame = train[lang_pair].get_dataframe(conversion_key[lang_pair], config["input_encoding"], config["output_encoding"])
                testing_frame = test[lang_pair].get_dataframe(conversion_key[lang_pair], config["input_encoding"], config["output_encoding"])
                baseline.compute_baseline(lang_a, lang_b, sounds, training_frame, testing_frame, baselines_path + ".txt")
            if config["visualize"]:
                print("Inferring sound correspondences...")
                visualize.show_output_substitutions(results_path[lang_pair], subs_st_path[lang_pair], subs_sp_path[lang_pair])
            if config["visualize_weights"]:
                visualize.visualize_weights(context_vectors_path[lang_pair], lang_pair, config["input_encoding"], config["output_encoding"], config["results_dir"], sample=None)
    for lang_pair in lang_pairs:
        if config["visualize_encoding"]:
            # Create embedding for first languages
            emb_matrix = data.create_embedding(lang_pair[0], [tsv_cognates_path_train + ".tsv", tsv_cognates_path_valtest + ".tsv"])
            visualize.visualize_encoding(emb_matrix, feature_matrix_phon, lang_pair, config["results_dir"])
    if config["cluster"]:
        # Cluster based on word prediction distances
        print("WP TREE:\n")
        cluster.cluster_languages(lang_pairs, distances_path, output_path=distances_path)
    if config["baseline_cluster"]:
        # Source prediction baseline
        print("\nSOURCE BASELINE TREE")
        cluster.cluster_languages(lang_pairs, baselines_path, output_path=baselines_path + "_source", distance_col=2)
        # PMI-based baseline
        print("\nPMI BASELINE TREE")
        cluster.cluster_languages(lang_pairs, baselines_path, output_path=baselines_path + "_pmi", distance_col=3)
    if config["cognate_detection"]:
        print("Performing WP cognate detection using clustering...")
        results_table = cd.cognate_detection_cluster(lang_pairs, config["results_dir"], options, use_distance="prediction")
        print(results_table)

    # Phylogenetic word prediction comes after datasets have been generated for
    # all language pairs. All language pairs are then taken into account at once
    # by phylogenetic word prediction
    if config["prediction"] and config["phyl"] and not config["seq"]:
        config["export_weights"] = False  # Turn off export of weights
        print("Performing phylogenetic word prediction")
        tree_string = "((nld,deu),eng)"  # unused at the moment
        if len(config["languages"]) >= 3:
            results_path_proto = utility.create_path(config["results_dir"], options, prefix="proto_")  # lang-pair independent path
            prediction.word_prediction_phyl(config["languages"], lang_pairs, tree_string, max_len, train, val, test, conversion_key_general, voc_size, results_path, results_path_proto, distances_path + ".txt", context_vectors_path, plot_path_phyl, config["output_encoding"], config)
        else:
            print("Please supply 3 languages, the first 2 being more closely related than the last.")
    
    
def print_flags(FLAGS):
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


if __name__ == "__main__":
    main()
