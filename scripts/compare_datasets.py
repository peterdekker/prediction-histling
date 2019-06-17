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
from collections import defaultdict
from lingpy.align.pairwise import sw_align, nw_align
import numpy as np
import os
import pandas as pd

from util import utility
from dataset import data


ielex_original = "data/ielex-4-26-2016.csv"
nelex_original = "data/northEuraLex.IE.asjp.csv"
ielex_file = "ielex-asjp.tsv"
nelex_file = "northeuralex-asjp.tsv"
intersection_file = "ielex-northeuralex-0.9-intersection.tsv"

K = 10
MIN_DIST = 0.0
MAX_DIST = 0.4
OUTPUT_DIR = "data_corrected"
table_filename = os.path.join(OUTPUT_DIR, "ielex_nelex_comparison.csv")
output_file = os.path.join(OUTPUT_DIR, "ielex-corr-asjp.tsv")


def compile_table(table_filename):
    # Convert original datasets to uniform table format
    data.convert_wordlist_tsv(ielex_original, source="ielex", input_type="asjp", output_path=ielex_file)
    data.convert_wordlist_tsv(nelex_original, source="northeuralex", input_type="asjp", output_path=nelex_file, intersection_path=intersection_file)
    df_ielex = pd.read_csv(ielex_file, sep="\t", index_col=False)
    df_nelex = pd.read_csv(nelex_file, sep="\t", index_col=False)

    # Load concept mapping NorthEuraLex and IELex
    ielex_nelex_map = data.load_ielex_nelex_concept_mapping("data/ielex-nelex-mapping.csv")

    # Add extra column with equivalent forms from NorthEuraLex
    nelex_entries = []
    distances = []
    for _, row in df_ielex.iterrows():
        # Default values, if no equivalent is found in NElex
        entry = ""
        min_dist = 1.0
        
        ie_concept = row["CONCEPT"]
        ie_doculect = row["DOCULECT"]
        ie_tokens = row["TOKENS"]
        if ie_concept in ielex_nelex_map:
            nelex_row = df_nelex[ie_doculect == df_nelex["DOCULECT"]][df_nelex["CONCEPT"] == ielex_nelex_map[ie_concept]]
            if not nelex_row.empty:
                # More alternatives, pick one with smallest Levenshtein distance to IELex form
                for ne_tokens in nelex_row["TOKENS"]:
                    dist = utility.calculate_levenshtein(ne_tokens.split(), ie_tokens.split())
                    if dist < min_dist:
                        min_dist = dist
                        entry = ne_tokens
        # Add one nelex entry per ielex row, so column has same length
        nelex_entries.append(entry)
        distances.append(min_dist)
    
    df_ielex["IELex"] = df_ielex["TOKENS"]
    df_ielex["NELex"] = nelex_entries
    df_ielex["Distance_IE_NE"] = distances
    df_ielex.to_csv(table_filename)
    return df_ielex


def get_top_substitutions(df, k, min_dist, max_dist, output_dir, language=None):
    df_filtered = df[(df["Distance_IE_NE"] > min_dist) & (df["Distance_IE_NE"] < max_dist)]
    # Filter on just one language, if specified
    if language is not None:
        df_filtered = df_filtered[df_filtered["DOCULECT"] == language]
    # Dictionary that stores substitution dataframes for the tree methods
    df_subs = {}
    for m in ["nw", "sw", "sw-x"]:
        subs_list = df_filtered.apply(substitutions_row, axis=1, method=m)
        subs_dict = combine_dict(subs_list)
        df_subs[m] = format_subs_dict(subs_dict, k)
        filename = os.path.join(output_dir, m)
        if language is not None:
            filename += "-" + language
        df_subs[m].to_csv(filename + ".tsv", sep="\t")
    return df_subs


def format_subs_dict(subs_dict, k):
    # subs_sorted = sorted(subs_dict.items(), key=lambda x: x[1], reverse=True)
    rows = []
    for (ie, ne) in subs_dict:
        count = subs_dict[ie, ne]
        rows.append([ie, ne, count])
    df_subs = pd.DataFrame(rows, columns=["IE", "NE", "#"])
    df_subs.sort_values("#", inplace=True, ascending=False)
    return df_subs.iloc[:k]


def substitutions_row(row, method):
    subs_list = []
    seq1 = row["IELex"].split()
    seq2 = row["NELex"].split()
    # Perform the Needleman-Wunsch algorithm for local alignment
    if method == "nw":
        seq1_align, seq2_align, _ = nw_align(seq1, seq2)
    else:
        # Else (method=sw or method=sw-x): Smith-Waterman alignment
        seq1_align, seq2_align, _ = sw_align(seq1, seq2)
    assert len(seq1_align) == len(seq2_align)
    for i in np.arange(len(seq1_align)):
        if seq1_align[i] != seq2_align[i]:
            subset_len = len(seq1_align[i])
            # Middle subset (with same length): split into tokens
            if method == "sw-x" and i == 1:
                for l in np.arange(subset_len):
                    if seq1_align[i][l] != seq2_align[i][l]:
                        token1 = seq1_align[i][l]
                        token2 = seq2_align[i][l]
                        subs_list.append((token1, token2))
            else:
                # If length is short, add whole combination as one substitution
                # This applies generally to the start and end parts
                token1 = " ".join(seq1_align[i])
                token2 = " ".join(seq2_align[i])
                subs_list.append((token1, token2))
    return subs_list


def combine_dict(subs_list):
    # Flatten list
    flattened = [val for part in subs_list for val in part]
    
    subs_dict = defaultdict(int)
    for pair in flattened:
        subs_dict[pair] += 1
    return subs_dict


def correct_ielex(df_ielex_corr, language, subs_df):
    # Make copy of dataframe, so changes are performed on old dataframe
    df_ielex_new = df_ielex_corr.copy()
    # Only use useful entries from substitution df:
    # remove
    print(language)
    subs_df = subs_df[subs_df["#"] >= 5].replace("-", "")
    subs_df = subs_df[subs_df["IE"] != ""]
    used_keys = []
    # No guarantee that longer keys (eg. 't S'), come before shorter keys (eg. 't')
    # In practice, in our data, longer keys are more frequent
    for _, row in subs_df.iterrows():
        key = row["IE"]
        if key in used_keys:
            continue
        used_keys.append(key)
        val = row["NE"]
        print("Applying " + str((key, val)))
        df_ielex_new.update(df_ielex_corr[df_ielex_corr["DOCULECT"] == language]["TOKENS"].str.replace(key, val))
    print(" ")
    return df_ielex_new


def main():
    # Create output directory, if needed
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    if os.path.exists(table_filename):
        df = pd.read_csv(table_filename)
    else:
        df = compile_table(table_filename)
    
    # Take all languages from IELex, for which there is also NELex data available
    ie_languages = df[pd.notnull(df["NELex"])]["DOCULECT"].unique()
    
    # Load IELex data, so this can be corrected
    df_ielex_corr = pd.read_csv(ielex_file, sep="\t", engine="python", index_col=False)
    
    # Correct data matrix per language
    for lang in ie_languages:
        # Get top substitutions per language, which are exported
        top_subs_lang = get_top_substitutions(df, k=K, min_dist=MIN_DIST, max_dist=MAX_DIST, output_dir=OUTPUT_DIR, language=lang)
        df_ielex_corr = correct_ielex(df_ielex_corr, language=lang, subs_df=top_subs_lang["sw-x"])
    # Replace double space (effect of some substitutions) by single space
    df_ielex_corr["TOKENS"] = df_ielex_corr["TOKENS"].str.replace("  ", " ")
    # Tokens have been corrected, now add phonological form
    df_ielex_corr["IPA"] = df_ielex_corr["TOKENS"].str.replace(" ", "")
    df_ielex_corr.to_csv(output_file, sep="\t", index=False)


if __name__ == "__main__":
    main()

