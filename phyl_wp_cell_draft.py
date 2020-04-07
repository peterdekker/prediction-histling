from util import utility
from prediction import prediction


# In phylogenetic mode, we created one feature matrix for all languages
for lang_pair in lang_pairs:
    conversion_key[lang_pair] = conversion_key_general

voc_size = voc_size_general
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

config["export_weights"] = False  # Turn off export of weights
tree_string = "((nld,deu),eng)"  # unused at the moment
if len(config["languages"]) >= 3:
    results_path_proto = utility.create_path(config["results_dir"], options, prefix="proto_")  # lang-pair independent path
    prediction.word_prediction_phyl(config["languages"], lang_pairs, tree_string, max_len, train, val, test, conversion_key_general, voc_size, results_path, results_path_proto, distances_path + ".txt", context_vectors_path, plot_path_phyl, config["output_encoding"], config)
else:
    print("Please supply 3 languages, the first 2 being more closely related than the last.")