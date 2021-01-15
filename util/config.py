from ete3 import TreeStyle, NodeStyle

# Number of units in the hidden (recurrent) layer
N_HIDDEN = 400
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
INITIALIZATION = "xavier_normal"  # ["constant", "xavier_normal", "xavier_uniform"])
OPTIMIZER = "adagrad"  # choices=["adagrad", "adam", "sgd"]
# All gradients above this will be clipped
GRAD_CLIP = 100
# Number of epochs to train the net
N_EPOCHS = 15
GATED_LAYER_TYPE = "gru"  # ["lstm", "gru"]
N_LAYERS_DENSE = 1
N_ITER_SEQ = 100
#TUNE_CD = False
INPUT_TYPE = "asjp"  # choices=["asjp", "ipa"]
INPUT_ENCODING = "embedding"  # ["phonetic", "character", "embedding"]
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
# TRAIN_CORPUS = "northeuralex" # ["northeuralex", "ielex", "ielex-corr"]
# VALTEST_CORPUS = "northeuralex" # ["northeuralex", "ielex", "ielex-corr"]
VIEW_EMBEDDING_IPA = True

# ## Other arguments
RESULTS_DIR = "output"
FEATURE_FILE = "data/asjp_phonetic_features_new.tsv"
SAMPLE_LANG_PAIRS = None

DATA_PATH = {"northeuralex": "data/northeuralex-cldf.tsv",
             "ielex": "data/ielex-4-26-2016.csv"}

DATA_URL = {"northeuralex": "http://www.sfs.uni-tuebingen.de/~jdellert/northeuralex/0.9/northeuralex-0.9-forms.tsv",
            "ielex": "TEST"}

CLTS_PATH = "v1.4.1.tar.gz"
CLTS_URL = "https://github.com/cldf-clts/clts/archive/v1.4.1.tar.gz"
            
# Define tree style
ETE_TREE_STYLE = TreeStyle()
ETE_TREE_STYLE.show_scale = False
ETE_TREE_STYLE.show_leaf_name = False
ETE_TREE_STYLE.force_topology = False
ETE_TREE_STYLE.show_border = False
ETE_TREE_STYLE.margin_top = ETE_TREE_STYLE.margin_bottom = ETE_TREE_STYLE.margin_right = ETE_TREE_STYLE.margin_left = 5

ETE_NODE_STYLE = NodeStyle()
ETE_NODE_STYLE["size"] = 0  # remove balls from leaves

config = {
    "n_hidden": N_HIDDEN,
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
    "n_iter_seq": N_ITER_SEQ,
    "input_type": INPUT_TYPE,
    "input_encoding": INPUT_ENCODING,
    "output_encoding": OUTPUT_ENCODING,
    "encoder_decoder_hid_init": ENCODER_DECODER_HID_INIT,
    "validation": VALIDATION,
    "mean_subtraction": MEAN_SUBTRACTION,
    "no_standardization": NO_STANDARDIZATION,
    "lr_decay": LEARNING_RATE_DECAY,
    "adaptive_lr": ADAPTIVE_LR,
    "cognacy_prior": COGNACY_PRIOR,
    "filter_train": FILTER_TRAIN,
    "export_weights": EXPORT_WEIGHTS,
    # "train_corpus": TRAIN_CORPUS,
    # "valtest_corpus": VALTEST_CORPUS,
    "results_dir": RESULTS_DIR,
    "feature_file": FEATURE_FILE,
    "sample_lang_pairs": SAMPLE_LANG_PAIRS,
    "data_url": DATA_URL,
    "data_path": DATA_PATH,
    "ete_tree_style": ETE_TREE_STYLE,
    "ete_node_style": ETE_NODE_STYLE,
    "view_embedding_ipa": VIEW_EMBEDDING_IPA,
    "clts_path": CLTS_PATH,
    "clts_url": CLTS_URL
}

# Set batch size to 1 for weight visualization:
# we want to feed individual words through the network
if config["export_weights"]:
    config["batch_size"] = 1
