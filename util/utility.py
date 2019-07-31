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
# ASJP conversion functions in this file are written by Gerhard Jaeger.

import distance
from lingpy.sequence.sound_classes import ipa2tokens, tokens2class
import matplotlib.pyplot as plt
import os
import re
import theano.tensor as T
import random


sounds = ['!', '3', '4', '5', '7', '8', 'C', 'E', 'G',
          'L', 'N', 'S', 'T', 'X', 'Z', 'a', 'b', 'c',
          'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
          'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
          'v', 'w', 'x', 'y', 'z']


def find(f, seq):
    """Return first item in sequence where f(item) == True."""
    for item in seq:
      if f(item): 
        return item


def clean_asjp(word):
    """
    Removes ASJP diacritics.
    """
    word = re.sub(r",", "-", word)
    word = re.sub(r"\%", "", word)
    word = re.sub(r"\*", "", word)
    word = re.sub(r"\"", "", word)
    word = re.sub(r".~", "", word)
    word = re.sub(r"(.)(.)(.)\$", r"\2", word)
    word = re.sub(r" ", "-", word)
    return word


def ipa_to_asjp(w):
    """
    Lingpy IPA-to-ASJP converter plus some cleanup.
    This function is called on IPA datasets.
    """
    w = w.replace('\"', '').replace('-', '').replace(' ', '')
    wA = ''.join(tokens2class(ipa2tokens(w, merge_vowels=False), 'asjp'))
    wAA = clean_asjp(wA.replace('0', '').replace('I', '3').replace('H', 'N'))
    asjp = ''.join([x for x in wAA if x in sounds])
    return asjp


# Return just last part of language name, without family name
def short_lang(language_name):
    return language_name.split(".")[-1]


def shorten(word, length=3):
    return "_".join([k[:length] for k in word.split("_")])


def create_path(output_dir, options, prefix, lang_a=None, lang_b=None):
    lang_string = ""
    if lang_a and lang_b:
        lang_string = short_lang(lang_a) + "-" + short_lang(lang_b) + "."
    return os.path.join(output_dir, prefix + lang_string + options)


def get_results_path(lang_a, lang_b, output_dir, options):
    return os.path.join(output_dir, short_lang(lang_a) + "-" + short_lang(lang_b) + "." + options)


def get_distances_path(output_dir, options):
    return os.path.join(output_dir, "dist" + "." + options)


def get_baselines_path(output_dir, options):
    return os.path.join(output_dir, "base" + "." + options)


def create_option_string(config):
    filename = ""
    # All modes except 'cognate_detection' are excluded from option string
    # CD result files have to be identified (they are different, also non-cognates)
    # when performing cognate detection
    omit = ["prediction", "cluster", "visualize", "visualize_weights", "visualize_encoding", "baseline", "baseline_cluster", "tune_cd", "tune_source_cd", "show_n_cog", "export_weights", "input_type", "grad_clip", "layers_encoder", "layers_decoder", "layers_dense", "adaptive_lr"]
    # Only put languages in file name during phylogenetic word prediction
    if not config["phyl"]:
        omit.append("languages")
    for key, value in sorted(config.items()):
        # Use only first letter of every word part
        key_short = shorten(key)
        if isinstance(value, bool):
            if value:  # If True
                if key not in omit:
                    filename += key_short + "."
        elif isinstance(value, str):
            filename += shorten(value, length=5) + "."
        elif isinstance(value, float) or isinstance(value, int):
            if value > 0.0:
                filename += key_short + str(value) + "."
        elif isinstance(value, list):
            filename += "_".join(value)
    return filename


def calculate_levenshtein(target_cut, predicted_cut):
    return distance.levenshtein(target_cut, predicted_cut) / float(max(len(target_cut), len(predicted_cut)))


def plot_loss(losses, distances, plot_filename):
    # Plot loss
    legend_info = []
    plt.title("Word prediction")
    loss_x = [p[0] for p in losses]
    loss_y = [p[1] for p in losses]
    loss_line, = plt.plot(loss_x, loss_y, label="Loss")
    legend_info.append(loss_line)
    
    dist_x = [p[0] for p in distances]
    dist_y = [p[1] for p in distances]
    distance_line, = plt.plot(dist_x, dist_y, label="Levenshtein distance (normalized)")
    legend_info.append(distance_line)
    
    plt.legend(handles=legend_info)
    plt.savefig(plot_filename)
    plt.close()


def plot_loss_phyl(losses_dict, distances_dict, plot_filename):
    # Plot loss
    legend_info = []
    plt.title("Loss")
    for lang_a in losses_dict:
        for lang_b in losses_dict[lang_a]:
            losses = losses_dict[lang_a][lang_b]
            loss_x = [p[0] for p in losses]
            loss_y = [p[1] for p in losses]
            loss_line, = plt.plot(loss_x, loss_y, label=lang_a + "-" + lang_b)
            legend_info.append(loss_line)
    plt.legend(handles=legend_info)
    plt.savefig(plot_filename + "_loss.png")
    plt.close()
    
    # Plot loss
    legend_info = []
    plt.title("Edit distance")
    for lang_a in distances_dict:
        for lang_b in distances_dict[lang_a]:
            distances = distances_dict[lang_a][lang_b]
            dist_x = [p[0] for p in distances]
            dist_y = [p[1] for p in distances]
            distance_line, = plt.plot(dist_x, dist_y, label=lang_a + "-" + lang_b)
            legend_info.append(distance_line)
    plt.legend(handles=legend_info)
    plt.savefig(plot_filename + "_dist.png")
    plt.close()


def print_sorted(dct, count=False, fmt_list=False):
    if count:
        count_dct = {}
        for key in dct:
            count_dct[key] = len(dct[key])
        dct = count_dct
    sorted_list = sorted(dct.items(), key=lambda x: x[1], reverse=True)
    
    if fmt_list:
        for k, v in sorted_list:
            print("&".join(k) + "&" + str(v) + "\\\\")
    else:
        print(sorted_list)


def generate_pairs(languages, languages2=None, allow_permutations=False, sample=None):
    lang_pairs = []
    if languages2 is None:
        languages2 = languages
    for a in languages:
        for b in languages2:
            if a != b and (allow_permutations or (b, a) not in lang_pairs):
                lang_pairs.append((a, b))
    
    if sample:
        lang_pairs = random.sample(lang_pairs, sample)
    return lang_pairs


# Sigmoid function which acts on tensors
def sigmoid(x):
    return (1 / (1 + T.exp(-x)))

