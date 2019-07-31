
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
INITIALIZATION = "xavier_normal" # ["constant", "xavier_normal", "xavier_uniform"])
OPTIMIZER = "adagrad" #  choices=["adagrad", "adam", "sgd"]
# All gradients above this will be clipped
GRAD_CLIP = 100
# Number of epochs to train the net
N_EPOCHS = 15
GATED_LAYER_TYPE = "gru" # ["lstm", "gru"]
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
INPUT_TYPE = "asjp" # choices=["asjp", "ipa"]
INPUT_ENCODING = "character" # ["phonetic", "character", "embedding"]
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
TRAIN_CORPUS = "northeuralex" # ["northeuralex", "ielex", "ielex-corr"]
VALTEST_CORPUS = "northeuralex" # ["northeuralex", "ielex", "ielex-corr"]
TRAIN_PROTO = False

# ## Other arguments
RESULTS_DIR = "output"
FEATURE_FILE = "data/asjp_phonetic_features_new.tsv"
LANGUAGES = ["nld", "deu"]
LANG_FAMILY = "none" # choices=["none", "slav", "ger"]
SAMPLE_LANG_PAIRS = None

LANG_FAMILIES_DICT = {
"slav": ["ces", "bul", "rus", "bel", "ukr", "pol", "slk", "slv", "hrv"],
"ger": ["swe", "isl", "eng", "nld", "deu", "dan", "nor"]
}

config = {
            "n_hidden": N_HIDDEN,
            "n_units_phyl": N_UNITS_PHYL,
            "n_layers_encoder": N_LAYERS_ENCODER,
            "n_layers_decoder": N_LAYERS_DECODER,
            "dropout": DROPOUT,
            "bidirectional_encoder": BIDIRECTIONAL_ENCODER,
            "bidirectional_decoder": BIDIRECTIONAL_DECODER,
            "encoder_all_steps": ENCODER_ALL_STEPS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "reg_weight": REG_WEIGHT,
            "init": INITIALIZATION,
            "optimizer": OPTIMIZER,
            "grad_clip": GRAD_CLIP,
            "n_epochs": N_EPOCHS,
            "gated_layer_type": GATED_LAYER_TYPE,
            "n_layers_dense": N_LAYERS_DENSE,
            "prediction": PREDICTION,
            "seq": SEQ,
            "phyl": PHYL,
            "n_iter_seq": N_ITER_SEQ,
            "cluster": CLUSTER,
            "visualize": VISUALIZE,
            "visualize_weights": VISUALIZE_WEIGHTS,
            "visualize_encoding": VISUALIZE_ENCODING,
            "baseline": BASELINE,
            "baseline_cluster": BASELINE_CLUSTER,
            "cognate_detection": COGNATE_DETECTION,
            "tune_cd": TUNE_CD,
            "show_n_cog": SHOW_N_COG,
            "input_type": INPUT_TYPE,
            "input_encoding": INPUT_ENCODING,
            "output_encoding": OUTPUT_ENCODING,
            "encoder_decoder_hid_init": ENCODER_DECODER_HID_INIT,
            "validation": VALIDATION,
            "mean_subtraction": MEAN_SUBTRACTION,
            "no_standardization": NO_STANDARDIZATION,
            "learning_rate_decay": LEARNING_RATE_DECAY,
            "adaptive_lr": ADAPTIVE_LR,
            "cognacy_prior": COGNACY_PRIOR,
            "filter_train": FILTER_TRAIN,
            "export_weights": EXPORT_WEIGHTS,
            "train_corpus": TRAIN_CORPUS,
            "valtest_corpus": VALTEST_CORPUS,
            "train_proto": TRAIN_PROTO,
            "results_dir": RESULTS_DIR,
            "feature_file": FEATURE_FILE,
            "languages": LANGUAGES,
            "lang_family": LANG_FAMILY,
            "sample_lang_pairs": SAMPLE_LANG_PAIRS,
            }

if config["lang_family"] is not "none":
    config["languages"] = LANG_FAMILIES_DICT[config["lang_family"]]