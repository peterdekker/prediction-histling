
import lingpy
import os

from util.config import config
from util import utility
import random

# Set random seed to make experiments repeatable
random.seed(10)

def initialize_program():
    # Set LingPy input encoding (IPA or ASJP)
    lingpy.settings.rc(schema=config["input_type"])
    
    if not os.path.exists(config["results_dir"]):
        os.mkdir(config["results_dir"])
    options = utility.create_option_string(config)

    # Create paths
    intersection_path = "data/ielex-northeuralex-0.9-intersection.tsv"
    distances_path = utility.get_distances_path(config["results_dir"], options)
    baselines_path = utility.get_baselines_path(config["results_dir"], options)
    return options, intersection_path, distances_path, baselines_path