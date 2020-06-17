# -------------------------------------------------------------------------------
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
# -------------------------------------------------------------------------------
from dataset.Dataset import Dataset

from collections import defaultdict
from cognatedetection import cd
from util import utility


import itertools
# from lingpy import *
from lingpy.sequence.sound_classes import ipa2tokens
import numpy as np
import os
import requests
import pandas as pd
from scipy.spatial.distance import euclidean
import igraph
import pickle
import pathlib

PROBLEM_LANGUAGES = ["IE.Indic.Bengali", "IE.Iranian.Ossetic", "IE.Iranian.Pashto"]


def download_if_needed(file_path, url):
    if not os.path.exists(file_path):
        print(f"Downloading file {url}...")
        # Create parent dirs
        p = pathlib.Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'wb') as f:
            print(f"Downloading dataset from {url}")
            r = requests.get(url, allow_redirects=True)
            # Write downloaded content to file
            f.write(r.content)


def load_data(train_corpus, valtest_corpus, languages, input_type, options, cognate_detection, config):
    # Set variables for train corpus
    input_path_train = config["data_path"][train_corpus]  # TODO: does not work yet for ielex-corr
    url_train = config["data_url"][train_corpus]
    output_path_train = os.path.join(config["results_dir"], f"{train_corpus}-{input_type}.tsv")
    output_path_cognates_train = os.path.join(config["results_dir"], f"{train_corpus}-{input_type}-cognates.tsv")
    # Download train corpus, if needed
    download_if_needed(input_path_train, url_train)

    if valtest_corpus != train_corpus:
        # Set variables for val/test corpus
        input_path_valtest = config["data_path"][valtest_corpus]  # TODO: does not work yet for ielex-corr
        url_valtest = config["data_url"][valtest_corpus]
        output_path_valtest = os.path.join(config["results_dir"], f"{valtest_corpus}-{input_type}.tsv")
        output_path_cognates_valtest = os.path.join(
            config["results_dir"], f"{valtest_corpus}-{input_type}-cognates.tsv")
        # Download valtest corpus, if needed
        download_if_needed(input_path_valtest, url_valtest)
    else:
        # If valtest and train corpus are the same, let valtest corpus point to same location as train corpus
        output_path_cognates_valtest = output_path_cognates_train

    if "all" in languages:
        print("Retrieving all languages from dataset...")
        languages = get_all_languages(input_path_train, train_corpus)

    print("Loading phonetic feature matrix...")
    feature_matrix_phon = load_feature_file(config["feature_file"])

    print("Generating all language pairs...")
    lang_pairs = utility.generate_pairs(languages, allow_permutations=len(languages) > 2, sample=config["sample_lang_pairs"])

    print("Training corpus:")
    load_dataset(input_path_train, source=train_corpus, input_type=input_type, output_path=output_path_train)
    cd.cognate_detection_lexstat(output_path_train, output_path_cognates_train, input_type=input_type)

    if valtest_corpus != train_corpus:
        print("Validation/test corpus:")
        load_dataset(input_path_valtest, source=valtest_corpus, input_type=input_type, output_path=output_path_valtest)
        concepts_valtest = fetch_concepts(input_path_valtest, source=valtest_corpus)
        cd.cognate_detection_lexstat(output_path_valtest, output_path_cognates_valtest, input_type=input_type)
    else:
        print("Train corpus is valtest corpus.")

    excluded_concepts_training = []
    if train_corpus != valtest_corpus:
        print("Loading IElex->NorthEuraLex concept mapping...")
        ielex_nelex_map = load_ielex_nelex_concept_mapping("data/ielex-nelex-mapping.csv")
        # All concepts in the validation/test corpus should be excluded from the training corpus
        for concept in concepts_valtest:
            if concept in ielex_nelex_map:
                concept_nelex = ielex_nelex_map[concept]
                excluded_concepts_training.append(concept_nelex)

    # Language-pair specific variables: every dict entry is designated for a specific lang pair
    results_path = {}
    subs_sp_path = {}
    subs_st_path = {}
    context_vectors_path = {}
    train = {}
    val = {}
    test = {}
    conversion_key = {}
    features = [pd.DataFrame(), pd.DataFrame()]

    # Max_len saved per language, rather than per language pair
    max_len = {}
    voc_size = {}

    for lang_pair in lang_pairs:
        lang_a, lang_b = lang_pair
        context_vectors_path[lang_pair] = utility.create_path(
            config["results_dir"], options, prefix="context_", lang_a=lang_a, lang_b=lang_b)
        # Create export path, containing all options
        # This is used to output a prediction results file, which can then be used for visualization and cognate detection
        results_path[lang_pair] = utility.get_results_path(lang_a, lang_b, config["results_dir"], options)
        subs_st_path[lang_pair] = utility.create_path(
            config["results_dir"], options, prefix="subs_st_", lang_a=lang_a, lang_b=lang_b)
        subs_sp_path[lang_pair] = utility.create_path(
            config["results_dir"], options, prefix="subs_sp_", lang_a=lang_a, lang_b=lang_b)

        # If data in pickle, load pickle
        data_pickle = results_path[lang_pair] + "-data.p"
        if os.path.exists(data_pickle):
            with open(data_pickle, "rb") as f:
                print("Loading train/val/test sets from pickle, nothing generated...")
                (train[lang_pair], val[lang_pair], test[lang_pair], conversion_key[lang_pair], max_len[lang_a],
                 max_len[lang_b], voc_size[lang_a], voc_size[lang_b]) = pickle.load(f)
        else:
            print("Creating feature matrix for this specific language pair...")
            features, max_len[lang_a], max_len[lang_b], voc_size[lang_a], voc_size[lang_b] = get_corpus_info(
                [output_path_cognates_train, output_path_cognates_valtest], lang_pair=lang_pair, input_encoding=config["input_encoding"], output_encoding=config["output_encoding"], config=config, feature_matrix_phon=feature_matrix_phon)
            conversion_key[lang_pair] = create_conversion_key(features)

            print("Converting training corpus TSV file to data matrix...")
            dataset_train, train_mean, train_std = create_data_matrix(tsv_path=output_path_cognates_train, lang_pair=(lang_a, lang_b), features=features, max_len=(max_len[lang_a], max_len[lang_b]), batch_size=config[
                                                                      "batch_size"], mean_subtraction=config["mean_subtraction"], feature_standardization=not config["no_standardization"], excluded_concepts=excluded_concepts_training, cognate_detection=cognate_detection)

            print("Converting val/test corpus TSV file to data matrix...")
            dataset_valtest, _, _ = create_data_matrix(tsv_path=output_path_cognates_valtest, lang_pair=(lang_a, lang_b), features=features, max_len=(max_len[lang_pair[0]], max_len[lang_pair[1]]), batch_size=config[
                                                       "batch_size"], mean_subtraction=config["mean_subtraction"], feature_standardization=not config["no_standardization"], cognate_detection=cognate_detection, valtest=True, train_mean=train_mean, train_std=train_std)

            t_set_size = dataset_train.get_size()
            vt_set_size = dataset_valtest.get_size()

            if valtest_corpus == train_corpus:
                # If train and valtest corpus the same, divide one corpus into parts
                assert t_set_size == vt_set_size
                n_train, n_val, n_test = dataset_train.compute_subset_sizes(t_set_size)
            else:
                # If train and valtest corpus different, use full train corpus as train and
                # full valtest corpus for validation and testing
                # TODO: In fact this is not needed, we can directly take set size.
                n_train, _, _ = dataset_train.compute_subset_sizes(t_set_size, only_train=True)
                _, n_val, n_test = dataset_valtest.compute_subset_sizes(vt_set_size, only_valtest=True)

            print("Dividing into training, validation and test set...")
            # Even if train and valtest corpus are the same, we do this separately,
            # because valtest corpus is filtered on cognates and train corpus is not
            # Use train corpus only for train set
            train[lang_pair], _, _ = dataset_train.divide_subsets(n_train, 0, 0)
            # Use val/test corpus for validation and test set
            _, val[lang_pair], test[lang_pair] = dataset_valtest.divide_subsets(0, n_val, n_test)

            print("Filtering val/test sets on cognates...")
            # Use only cognate pairs for validation and test
            val[lang_pair] = val[lang_pair].filter_cognates()
            test[lang_pair] = test[lang_pair].filter_cognates()
            print("Val/test sizes after cognate filtering: " +
                  str(val[lang_pair].get_size()) + "|" + str(test[lang_pair].get_size()))

            # Pickle train/val/test/sets
            with open(data_pickle, "wb") as f:
                pickle.dump((train[lang_pair], val[lang_pair], test[lang_pair], conversion_key[lang_pair],
                             max_len[lang_pair[0]], max_len[lang_pair[1]], voc_size[lang_a], voc_size[lang_b]), f)
    print("Done loading data.")
    return (results_path, output_path_cognates_train, output_path_cognates_valtest, context_vectors_path, subs_sp_path, subs_st_path, lang_pairs,
            train, val, test, max_len, conversion_key, voc_size, feature_matrix_phon)


def get_all_languages(data_file, source_type):
    if source_type == "ielex" or source_type == "ielex-corr":
        return pd.read_csv(data_file)["Language"].unique()
    elif source_type == "northeuralex":
        return pd.read_csv(data_file, sep="\t")["DOCULECT"].unique()


def write_results_table(results_table, testset, results_filename):
    results_table.to_csv(results_filename, sep="\t")

    # Write tables per concept. Needed for cognate detection


def load_ielex_nelex_concept_mapping(filename):
    df = pd.read_csv(filename, header=None, index_col=0, sep="\t")
    mapping_dict = dict(zip(df.index, df[1]))
    return mapping_dict


def load_feature_file(feature_file):
    features = pd.read_csv(feature_file, index_col=0, sep="\t")
    return features


def load_dataset(input_path, source, input_type, output_path):
    print(" - Loading dataset and performing necessary conversion/tokenization.")
    if os.path.exists(output_path):
        print("Using existing wordlist file, nothing is generated.")
        return
    # No NA filter: the word form 'nan' should not be interpreted as NaN :p
    df = pd.read_csv(input_path, sep="\t", na_filter=False)

    # Depending on file format, remove and/or rename columns
    if source == "ielex" or source == "ielex-corr":
        # Rename columns
        df.rename(columns={"Language": "DOCULECT", "Meaning": "CONCEPT", "Phonological Form": "IPA",
                           "Cognate Class": "COGNATES_IELEX", "cc": "CONCEPT_COGNATES_IELEX"}, inplace=True)
        # Drop column with unused numbers
        df.drop(df.columns[[0]], axis=1, inplace=True)
    elif source == "northeuralex":
        df.rename(columns={"Language_ID": "DOCULECT", "Concept_ID": "CONCEPT"}, inplace=True)

    tokens = []
    if source == "ielex":
        # Perform IPA->ASJP conversion if source is ielex
        forms = []
        for form_ipa in df["IPA"]:
            # ipa_to_asjp method accepts both space-separated (NELex) and
            # non-separated (IELex)
            if input_type == "asjp":
                form_asjp = utility.ipa_to_asjp(form_ipa)
                forms.append(form_asjp)
                tokens_form = list(form_asjp)
            elif input_type == "ipa":
                tokens_form = ipa2tokens(form_ipa)
            tokens_string = " ".join(tokens_form)
            tokens.append(tokens_string)
        if input_type == "asjp":
            df["ASJP"] = forms
        df["TOKENS"] = tokens
    elif source == "northeuralex":
        if input_type == "asjp":
            for form_asjp in df["ASJP"]:
                tokens_form = list(form_asjp)
                tokens_string = " ".join(tokens_form)
                tokens.append(tokens_string)
            df["TOKENS"] = tokens
        elif input_type == "ipa":
            df["TOKENS"] = df[input_type]
    # Filter out rows with XXX phonology field.
    df = df[df["IPA"] != "XXX"]
    # Filter out rows with empty phonology field
    df = df[df["IPA"] != ""]

    # Apply IELex cognate judgments to NElex
    # TODO: We can only do this if there is a publicly available intersection file
    #
    # if source == "northeuralex":
    #     # Load intersection file
    #     df_intersection = pd.read_csv(intersection_path, sep="\t")
    #     # Per row, retrieve matching IELex judgment from intersection
    #     cognates_intersection = []
    #     for _, row in df.iterrows():
    #         cog = df_intersection[((df_intersection["iso_code"] == row["DOCULECT"]) & (df_intersection["gloss_northeuralex"] == row["CONCEPT"]) & (df_intersection["ortho_northeuralex"] == row["COUNTERPART"]))]["cog_class_ielex"]
    #         if cog.empty:
    #             cog = None
    #         else:
    #             cog = cog.iloc[0]
    #         cognates_intersection.append(cog)
    #     df["COGNATES_IELEX"] = cognates_intersection
    #     # Create CONCEPT_COGNATES_IELEX column with unique cognate classes across concepts
    #     df["CONCEPT_COGNATES_IELEX"] = df["CONCEPT"] + "-" + df["COGNATES_IELEX"]

    print(f" - Writing corpus (with conversions) to {output_path}")
    df.to_csv(output_path, index_label="ID", sep="\t")

    # Add . to tokens list as end of word marker
    # tokens_list = list(tokens_set)
    # tokens_list.append(".")


def fetch_concepts(input_path, source):
    print(" - Fetch list of concepts (only for valtest corpus)")
    if source == "northeuralex":
        with open(input_path, "r") as input_file:
            concepts = input_file.readline().strip().split(",")[1:]
    elif source == "ielex" or source == "ielex-corr":
        df = pd.read_csv(input_path)
        # Rename columns
        df.rename(columns={"Language": "DOCULECT", "Meaning": "CONCEPT", "Phonological Form": "IPA",
                           "Cognate Class": "COGNATES_IELEX", "cc": "CONCEPT_COGNATES_IELEX"}, inplace=True)
        concepts = df["CONCEPT"].unique()

    else:
        raise ValueError("Unknown file format.")

    return concepts


def word_surface(encoded_word, conversion_key, encoding, mask=None):
    surface_tokens = []

    if mask is not None:
        word_length = np.count_nonzero(mask)
    else:
        word_length = encoded_word.shape[0]

    if encoding == "phonetic" or encoding == "embedding":
        # In this case, conversion_key is dat dict from feature strings to tokens
        for t in np.arange(word_length):
            encoded_token = tuple(encoded_word[t])
            if encoded_token in conversion_key:
                surface_tokens.append(conversion_key[encoded_token])
            else:
                nearest_token = find_nearest_token(encoded_token, conversion_key)
                surface_tokens.append(conversion_key[nearest_token])
    elif encoding == "character":
        # In this case, conversion key is the list of tokens
        for t in np.arange(word_length):
            # Get max
            i = np.argmax(encoded_word[t])
            # Compare max to every token in conversion key
            for avail_token in conversion_key:
                key_max = np.argmax(avail_token)
                if key_max == i:
                    surface_tokens.append(conversion_key[avail_token])
                    break
    surface_word = "".join(surface_tokens)
    return surface_word, surface_tokens


def find_nearest_token(encoded_token, conversion_key):
    lowest_dist = 1000.0
    for avail_token in conversion_key:
        # Compute euclidean distance between possible tokens in conversion key
        # and current token
        encoded_token = np.round(list(encoded_token))
        dist = euclidean(list(avail_token), encoded_token)
        if dist < lowest_dist:
            lowest_dist_token = avail_token

    return lowest_dist_token


# Encode word in one-hot encoding, all words have same max length
# Returns: (max_length, voc_size)
def encode_word(word_tokens, features, max_length):
    voc_size = features.shape[1]
    encoded_word = np.zeros((max_length, voc_size))

    tokens = word_tokens.split(" ")
    word_length = min(len(tokens), max_length)  # if word longer than max_length, rest is disregarded
    for t in np.arange(max_length):
        if t < word_length:
            token = tokens[t]
            encoded_token = features.loc[token]
        else:
            # Pick '.' characters for all empty spaces
            encoded_token = features.loc["."]
        encoded_word[t] = encoded_token
    mask = np.ones(max_length)
    mask[word_length:] = 0

    return encoded_word, mask


def _fill_word(word_tokens, max_length):
    tokens = word_tokens.split(" ")
    filled_word = ["."] * max_length
    filled_word[:len(tokens)] = tokens
    return filled_word


def get_corpus_info(paths_list, lang_pair, input_encoding, output_encoding, config, feature_matrix_phon=None):
    tokens_list_corpora = []
    max_len_corpora = []
    voc_size_corpora = []

    for tsv_path in paths_list:
        # Read in TSV file
        df = pd.read_csv(tsv_path, sep="\t", engine="python", skipfooter=3, index_col=False)

        # Get tokens set and word length for both languages
        tokens_list = [[], []]
        voc_size = [0, 0]
        max_len = [0, 0]
        for ix in [0, 1]:
            lang = lang_pair[ix]
            df_lang = df[df["DOCULECT"] == lang]
            words_list = list(df_lang["TOKENS"])

            tokens_set = set()
            for word in words_list:
                split_word = word.split()
                if len(split_word) > max_len[ix]:
                    max_len[ix] = len(split_word)
                tokens_set.update(split_word)
            tokens_list[ix] = list(tokens_set)
            tokens_list[ix].append(".")
            voc_size[ix] = len(tokens_list[ix])

        # Save values for different corpora in lists
        tokens_list_corpora.append(tokens_list)
        voc_size_corpora.append(voc_size)
        max_len_corpora.append(max_len)

    # Now combine the values from the different corpora
    max_len_x_combined = max([max_len[0] for max_len in max_len_corpora])
    max_len_y_combined = max([max_len[1] for max_len in max_len_corpora])

    tokens_list_corpora_x = [f[0] for f in tokens_list_corpora]
    tokens_list_corpora_y = [f[1] for f in tokens_list_corpora]
    tokens_list_combined_x = list(set(itertools.chain.from_iterable(tokens_list_corpora_x)))
    tokens_list_combined_y = list(set(itertools.chain.from_iterable(tokens_list_corpora_y)))

    # Separate encoding for input and output (target)

    if input_encoding == "phonetic":
        # Reduce feature matrix to just tokens used in this language
        feature_matrix_x = feature_matrix_phon.loc[tokens_list_combined_x]
    elif input_encoding == "character":
        feature_matrix_x = create_one_hot_matrix(tokens_list_combined_x)
    elif input_encoding == "embedding":
        feature_matrix_x = create_embedding(lang_pair[0], paths_list, config)

    if output_encoding == "phonetic":
        # Reduce feature matrix to just tokens used in this language
        feature_matrix_y = feature_matrix_phon.loc[tokens_list_combined_y]
    elif output_encoding == "character":
        feature_matrix_y = create_one_hot_matrix(tokens_list_combined_y)
    elif output_encoding == "embedding":
        feature_matrix_y = create_embedding(lang_pair[1], paths_list, config)

    feature_matrix = (feature_matrix_x, feature_matrix_y)
    voc_size_combined = (feature_matrix_x.shape[1], feature_matrix_y.shape[1])
    return feature_matrix, max_len_x_combined, max_len_y_combined, voc_size_combined[0], voc_size_combined[1]


def create_one_hot_matrix(tokens_list):
    # Create identity matrix with length of tokens_list
    eye = np.eye(len(tokens_list))
    df = pd.DataFrame(eye, index=tokens_list)
    return df


def create_embedding(lang, tsv_paths_list, config):
    print(" - Creating embedding for " + lang)
    # Re-use existing embedding file if possible
    corpora_names = [t.split(".tsv")[0].split("/")[-1] for t in tsv_paths_list]
    emb_filename = os.path.join(config["results_dir"], f"emb_{lang}_{'_'.join(corpora_names)}.tsv")
    if os.path.exists(emb_filename):
        print(" -- Using existing embedding file for " + lang)
        df_emb_file = pd.read_csv(emb_filename, sep="\t", index_col=0)
        return df_emb_file
    embedding = defaultdict(lambda: defaultdict(float))
    for tsv_path in tsv_paths_list:
        # Read in TSV file
        df = pd.read_csv(tsv_path, sep="\t", engine="python", skipfooter=3, index_col=False)
        df_lang = df[df["DOCULECT"] == lang]
        words_list = list(df_lang["TOKENS"])
        for word in words_list:
            split_word = word.split()
            for i, token in enumerate(split_word):
                # Count preceding and following token as context
                # Left neighbour and right neighbour are saved as separate features
                if i - 1 >= 0:
                    embedding[token][split_word[i - 1] + "_LEFT"] += 1
                else:
                    # Add start of word as neighbour
                    embedding[token]["START"] += 1
                if i + 1 < len(split_word):
                    embedding[token][split_word[i + 1] + "_RIGHT"] += 1
                else:
                    # Add end of word as neighbour
                    embedding[token]["END"] += 1

                # 2 LEFT AND 2 RIGHT
                # if i-2 >= 0:
                    # embedding[token][split_word[i-2]+"LEFT2"] += 1
                # else:
                    # # Add start of word as neighbour
                    # embedding[token]["START2"] += 1
                # if i+2 < len(split_word):
                    # embedding[token][split_word[i+2]+"RIGHT2"] += 1
                # else:
                    # # Add end of word as neighbour
                    # embedding[token]["END2"] += 1

    # Create "." character, filled with all 0s
    # (By touching only START neighbour, rest is set to 0 automatically
    # by defaultdict and fillna)
    embedding["."]["START"] = 0

    # Convert dict-of-dicts to dataframe
    df = pd.DataFrame.from_dict(embedding, orient="index")
    # Normalize df per row: values for all neighbours of a token sum to 1
    df = df.divide(df.sum(axis=1), axis=0).fillna(0)
    df.to_csv(emb_filename, sep="\t")
    return df


def create_data_matrix(tsv_path, lang_pair, features, max_len, batch_size, fixed_length_x=None,
                       fixed_length_y=None, mean_subtraction=False, feature_standardization=False, excluded_concepts=[],
                       only_cognates=False, cognate_detection=False, valtest=False, train_mean=None, train_std=None):

    # Read in TSV file
    df = pd.read_csv(tsv_path, sep="\t", engine="python", skipfooter=3, index_col=False)
    df = df[df["DOCULECT"].isin(lang_pair)]
    if df.empty:
        raise ValueError("The supplied language(s) is/are not in the corpus!")
    concepts = df["CONCEPT"].unique()
    # Sort to have same list of concepts for every language for cognate detection
    concepts = sorted(concepts)

    matrix_x = []
    matrix_x_unnormalized = []
    matrix_y = []
    mask_x = []

    matrix_x_unbounded = []
    matrix_y_unbounded = []

    datafile_ids = []
    word_lengths_unbounded = []

    for concept in concepts:
        if concept in excluded_concepts:
            continue
        concept_entries = df[df["CONCEPT"] == concept]
        lang0_entries = concept_entries[concept_entries["DOCULECT"] == lang_pair[0]]
        lang1_entries = concept_entries[concept_entries["DOCULECT"] == lang_pair[1]]
        if len(lang0_entries) == 0 or len(lang1_entries) == 0:
            # Concept not available for one of the languages in langpair, skip.
            continue

        # Add word pairs for all possible combinations of words for this concept
        for _, lang0_entry in lang0_entries.iterrows():
            for _, lang1_entry in lang1_entries.iterrows():
                x = lang0_entry["TOKENS"]
                y = lang1_entry["TOKENS"]
                # Save id of line in datafile, so line can later be looked up
                x_id = lang0_entry["ID"]
                y_id = lang1_entry["ID"]
                datafile_ids.append((x_id, y_id))
                # Encode words, for use in RNN data matrix
                # (max_len, voc_size)
                word_encoded_x, word_mask_x = encode_word(x, features[0], max_len[0])
                word_encoded_y, _ = encode_word(y, features[1], max_len[1])
                # Encode unbounded words (max len of word pair is max of words in pair),
                # for use in SeqModel data matrix.
                max_len_pair = np.maximum(len(x), len(y))
                # X for SeqModel is encoded
                word_encoded_x_unbounded, _ = encode_word(x, features[0], max_len_pair)
                # Y for SeqModel is not encoded, just filled to maxlen
                word_encoded_y_unbounded = _fill_word(y, max_len_pair)
                # Keep track of word lengths, needed for SeqModel algorithm
                word_lengths_unbounded.append(max_len_pair)

                if mean_subtraction and not feature_standardization:
                    word_encoded_x_norm = perform_mean_subtraction(word_encoded_x)
                    matrix_x.append(word_encoded_x_norm)
                else:
                    matrix_x.append(word_encoded_x)
                matrix_x_unnormalized.append(word_encoded_x)
                matrix_y.append(word_encoded_y)
                mask_x.append(word_mask_x)

                # Unbounded matrix for SeqModel
                # Unbounded X matrix is always unnormalized
                matrix_x_unbounded.append(word_encoded_x_unbounded)
                matrix_y_unbounded.append(word_encoded_y_unbounded)
                # In cognate detection mode: only add one form per concept,
                # to keep languages synchronzied
                if cognate_detection:
                    break
            if cognate_detection:
                break

    word_lengths_2 = [len(x) for x in matrix_x_unbounded]
    assert np.sum(word_lengths_2) == np.sum(word_lengths_unbounded)
    # Convert list of NumPy arrays to full NumPy array
    # (n_samples, max_len, voc_size)
    matrix_x = np.array(matrix_x)
    matrix_x_unnormalized = np.array(matrix_x_unnormalized)
    matrix_y = np.array(matrix_y)
    mask_x = np.array(mask_x)
    assert matrix_x.shape[0] == matrix_y.shape[0]

    # SeqModel matrices: convert to NP array, to enable fancy indexing
    # Row lengths are uneven, because of uneven word lengths
    matrix_x_unbounded = np.array(matrix_x_unbounded)
    matrix_y_unbounded = np.array(matrix_y_unbounded)
    word_lengths_unbounded = np.array(word_lengths_unbounded)

    # Feature standardization
    train_mean_calc = None
    train_std_calc = None
    if feature_standardization:
        # During training: standardize using own mean and std
        if not valtest:
            # Save calculcated train mean and std,
            # to give to valtest set
            matrix_x, train_mean_calc, train_std_calc = standardize(matrix_x_unnormalized)
        # During valtest: standardize using mean and std from train set
        if valtest:
            print("USE TRAIN M/S")
            matrix_x, _, _ = standardize(matrix_x_unnormalized, valtest=True,
                                         train_mean=train_mean, train_std=train_std)
    return Dataset(batch_size, matrix_x, matrix_x_unnormalized, matrix_y, mask_x, max_len[0], max_len[1], matrix_x_unbounded, matrix_y_unbounded, tsv_path, datafile_ids, word_lengths_unbounded), train_mean_calc, train_std_calc


def split_predictions(predictions, word_lengths):
    start = 0
    predictions_split = []
    for length in word_lengths:
        end = start + length
        predictions_split.append(predictions[start:end])
        start = end
    return predictions_split


def standardize(matrix, valtest=False, train_mean=None, train_std=None):
    matrix_t = matrix.T
    matrix_new_t = np.zeros(matrix_t.shape)
    # After transposing matrix, every row corresponds to one feature, over all examples
    n_features = matrix_t.shape[0]

    # Create arrays for mean and std per feature
    mean = np.zeros(n_features)
    std = np.zeros(n_features)
    for feat in np.arange(n_features):
        row = matrix_t[feat]
        if not valtest:
            # Compute mean and standard deviation for this feature
            mean[feat] = np.mean(row)
            std[feat] = np.std(row)
        else:
            # Use train mean and std
            mean[feat] = train_mean[feat]
            std[feat] = train_std[feat]
        # Subtract mean from current values and divide by std deviation
        new_row = np.subtract(row, mean[feat])
        if std[feat] > 0.0:
            new_row = np.divide(new_row, std[feat])
        matrix_new_t[feat] = new_row
    matrix_new = matrix_new_t.T
    return matrix_new, mean, std


def perform_mean_subtraction(input_vector):
    mean = np.mean(input_vector)
    return np.subtract(input_vector, mean)


def _calculate_max_len(word_lengths, count_threshold):
    count = defaultdict(int)
    for word_len in word_lengths:
        count[word_len] += 1
    max_len = 0
    for word_len in count:
        if word_len > max_len and count[word_len] > count_threshold:
            max_len = word_len
    return max_len


def max_word_lengths(word_pairs):
    count_threshold = 0
    word_lengths_x = [len(p["lang0"].split(" ")) for p in word_pairs]
    word_lengths_y = [len(p["lang1"].split(" ")) for p in word_pairs]
    max_len_x = _calculate_max_len(word_lengths_x, count_threshold)
    max_len_y = _calculate_max_len(word_lengths_y, count_threshold)
    return max_len_x, max_len_y


def create_conversion_key(features):
    conversion_key = ({}, {})
    for i in [0, 1]:
        for token in features[i].index:
            # Create string of features
            key = tuple(features[i].loc[token])
            conversion_key[i][key] = token
    return conversion_key


def compute_n_cognates(lang_pairs, input_file, langs, cognates_threshold, config):
    print("Calculate number of cognates per language...")
    df = pd.read_csv(input_file, sep="\t")
    concepts = df["CONCEPT"].unique()
    cognate_count = defaultdict(int)
    for (lang_a, lang_b) in lang_pairs:
        # lang_a_entries = df[df["DOCULECT"]==lang_a].reset_index(drop=True)
        # lang_b_entries = df[df["DOCULECT"]==lang_b].reset_index(drop=True)
        # print(lang_a_entries["COGNATES_IELEX"])
        # print(lang_b_entries["COGNATES_IELEX"].count())
        # cognate_count[(lang_a,lang_b)] = lang_a_entries[lang_a_entries["COGNATES_IELEX"] == lang_b_entries["COGNATES_IELEX"]].count()

        for concept in concepts:
            concept_entries = df[df["CONCEPT"] == concept]
            lang0_entries = concept_entries[concept_entries["DOCULECT"] == lang_a]
            lang1_entries = concept_entries[concept_entries["DOCULECT"] == lang_b]
            if len(lang0_entries) == 0 or len(lang1_entries) == 0:
                # Concept not available for one of the languages in langpair, skip.
                continue

            # Add word pairs for all possible combinations of words for this concept
            for _, lang0_entry in lang0_entries.iterrows():
                for _, lang1_entry in lang1_entries.iterrows():
                    # First, try to use Ielex
                    if "COGNATES_IELEX" in df and pd.notnull(lang0_entry["COGNATES_IELEX"]) and pd.notnull(lang1_entry["COGNATES_IELEX"]):
                        if lang0_entry["COGNATES_IELEX"] == lang1_entry["COGNATES_IELEX"]:
                            cognate_count[(lang_a, lang_b)] += 1
                    # Only use LexStat judgments if IeLEX is unavailable
                    elif "COGNATES_LEXSTAT" in df and pd.notnull(lang0_entry["COGNATES_LEXSTAT"]) and pd.notnull(lang1_entry["COGNATES_LEXSTAT"]):
                        if lang0_entry["COGNATES_LEXSTAT"] == lang1_entry["COGNATES_LEXSTAT"]:
                            cognate_count[(lang_a, lang_b)] += 1

    # Create dataframe of cognate counts per language pair
    count_df = pd.DataFrame.from_dict(cognate_count, orient="index")
    count_df.columns = ["Cognates"]
    count_df = count_df.sort_values("Cognates", ascending=False)
    n_cognates_filename = os.path.join(config["results_dir"], "n_cognates.tsv")
    count_df.to_csv(n_cognates_filename, sep="\t")

    print("Calculate cliques...")
    # Create graph of language pairs with minimum number of cognates
    graph = igraph.Graph()
    graph.add_vertices(langs)
    for lang_a, lang_b in cognate_count:
        if cognate_count[lang_a, lang_b] > cognates_threshold:
            graph[lang_a, lang_b] = 1

    cliques = graph.maximal_cliques()
    cliques_labels = [[langs[item] for item in clique] for clique in cliques]
    # Sort keys
    cliques_labels.sort(key=len, reverse=True)
    cliques_filename = os.path.join(config["results_dir"], f"langs_cliques_{str(cognates_threshold)}.txt")
    with open(cliques_filename, "w") as f:
        for cl in cliques_labels:
            f.write(" ".join(cl) + "\n")
    return count_df, cliques_labels

# ~ def generate_random_Y(shape):
    # ~ # Random Y: set one random position of every one-hot-array to 1
    # ~ Y = np.zeros(shape)
    # ~ n_examples = Y.shape[0]
    # ~ word_length = Y.shape[1]
    # ~ voc_size = Y.shape[2]
    # ~ for ex in np.arange(n_examples):
    # ~ for t in np.arange(word_length):
    # ~ i = random.randint(0,voc_size-1)
    # ~ Y[ex][t][i] = 1
    # ~ return Y
