
import lingpy
import os

from util.config import config
from util import utility

# Set random seed to make experiments repeatable
random.seed(10)

def initialize_program():
    # Set LingPy input encoding (IPA or ASJP)
    lingpy.settings.rc(schema=config["input_type"])
    
    if not os.path.exists(config["results_dir"]):
        os.mkdir(config["results_dir"])
    options = utility.create_option_string(config)
    return options