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
from util import init
from util.config import config
from visualize import visualize


# External libs
import pickle
import lingpy
import numpy as np
import os
import pandas as pd


def main():
    options = init.initialize_program()
    intersection_path, distances_path, baselines_path = data.load_data(config["train_corpus"])

        
    
def print_flags(FLAGS):
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


if __name__ == "__main__":
    main()
