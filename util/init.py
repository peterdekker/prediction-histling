
import lingpy
import os

from util import utility
import random
import os
import requests
import pathlib
import shutil

# Set random seed to make experiments repeatable
random.seed(10)

def download_if_needed(file_path, url, label):
    if not os.path.exists(file_path):
        # Create parent dirs
        p = pathlib.Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'wb') as f:
            print(f"Downloading {label} from {url}")
            try:
                r = requests.get(url, allow_redirects=True)
            except requests.exceptions.RequestException as e:  # This is the correct syntax
                raise SystemExit(e)
            # Write downloaded content to file
            f.write(r.content)
            if file_path.endswith(".tar.gz"):
                print("Unpacking archive.")
                shutil.unpack_archive(file_path)

def initialize_program(cognate_detection, config):
    print("Initializing program...")
    # Set LingPy input encoding (IPA or ASJP)
    lingpy.settings.rc(schema=config["input_type"])
    
    if not os.path.exists(config["results_dir"]):
        os.mkdir(config["results_dir"])
    options = utility.create_option_string(config, cognate_detection)

    # Create paths
    #intersection_path = "data/ielex-northeuralex-0.9-intersection.tsv"
    distances_path = utility.get_distances_path(config["results_dir"], options)
    baselines_path = utility.get_baselines_path(config["results_dir"], options)

    # Download CLTS
    download_if_needed(config["clts_path"], config["clts_url"], "CLTS")

    return options, distances_path, baselines_path