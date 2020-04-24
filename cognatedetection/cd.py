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
import bcubed
from collections import defaultdict
from itertools import permutations, chain
from lingpy import LexStat
from lingpy.algorithm.clustering import flat_upgma, fuzzy, link_clustering, mcl
from lingpy.algorithm.extra import affinity_propagation, infomap_clustering
import numpy as np
import os
import pandas as pd
from util import utility


# import igraph
# # UNFINISHED, label propagation does not work as expected
# Wrapper function, to give LP (from infograph library) same argument structure
# as LingPy clustering methods
# def community_label_propagation(threshold, dist_matrix, languages):
    # graph = igraph.Graph.Full(len(languages))
    # graph.vs["name"] = languages
    # graph.es["weights"] = 0.0
    # # Transfer information from distance matrix to graph
    # for edge in dist_matrix:
        # dist = dist_matrix[edge]
        # # Transform distances to weights
        # weight = 1 - dist
        # graph[edge] = weight
    # graph.community_label_propagation()

def invert_key_value(cluster_dict):
    new_dict = defaultdict(set)
    for key in cluster_dict:
        list_values = cluster_dict[key]
        for value in list_values:
            # Append to set, because one language may belong to multiple clusters
            new_dict[value].add(key)
    return dict(new_dict)


# Evaluate cognate detection using B-Cubed P/R/F
def evaluate_bcubed(judgments, gold):
    precision = bcubed.precision(judgments, gold)
    recall = bcubed.recall(judgments, gold)
    fscore = bcubed.fscore(precision, recall)
    return precision, recall, fscore


# Inferred cognate judgments and gold standard data have different cluster labels
# This method applies the cluster labels from the judgments to the gold standard data 
# and returns all possible label assignments
def generate_gold_relabellings(gold, judgments_clusters):
    # Create all permutations of the judgments keys, with the length of the gold
    # data
    judg_perms = permutations(judgments_clusters)
    labellings = []
    for perm in judg_perms:
        perm = list(perm)
        # Regenerate gold_items, because it is iterated and emptied every loop
        gold_items = gold.items()
        new_dict = {}
        for key in perm:
            # Check if there are gold items less, else add empty set
            if gold_items:
                _, v = gold_items.pop()
            else:
                v = set()
            new_dict[key] = v
        labellings.append(new_dict)
    
    return labellings


# Distance-based cognate detection (using threshold) on results file of word prediction
def cognate_detection_binary(results_file, distance, thresholds):
    if distance == "prediction":
        dist_label = "DISTANCE_T_P"
        cognates_label = "COGNATES_WP"
    elif distance == "source":
        dist_label = "DISTANCE_S_T"
        cognates_label = "COGNATES_SOURCE"
    df = pd.read_csv(results_file, sep="\t")
    
    for t in thresholds:
        judgments = []
        for _, row in df.iterrows():
            judgment = 1 if row[dist_label] < t else 0
            judgments.append(judgment)
        setting = cognates_label + "_" + str(t)
        df[setting] = judgments
    print(df)
    return df


# Cognate detection (using clustering) on results files of word prediction
# TODO: re-add infomap clustering?
def cognate_detection_cluster(lang_pairs, results_dir, options, use_distance="prediction", methods=[("Flat UPGMA", flat_upgma), ("Link clustering", link_clustering), ("MCL", mcl)], thresholds=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]):
    languages = list(set([e for pair in lang_pairs for e in pair]))
    gold_column = "COGNATES_LEXSTAT" # TODO: change to COGNATES_IELEX once intersection file is working

    if use_distance == "prediction":
        dist_label = "DISTANCE_T_P"
    elif use_distance == "source":
        dist_label = "DISTANCE_S_T"
    
    # Retrieve list of concepts from file from first language pair
    lang_a, lang_b = lang_pairs[0]
    try:
        concepts_path = utility.get_results_path(lang_a, lang_b, results_dir, options)
        df = pd.read_csv(concepts_path + ".tsv", sep="\t")
    except:
        # If it fails, try reverse pair
        concepts_path = utility.get_results_path(lang_b, lang_a, results_dir, options)
        df = pd.read_csv(concepts_path + ".tsv", sep="\t")
    concepts = list(df["CONCEPT"])
    max_dist = 1.0
    n_langs = len(languages)
    # Distance is temporary matrix
    distance = defaultdict(dict)
    forms = defaultdict(dict)
    cog_class_gold = defaultdict(lambda: defaultdict(set))
    # Matrix is eventual matrix, where mean of distances in two directions is taken
    matrix = defaultdict(lambda: np.zeros((n_langs, n_langs)))
    for ix_a, lang_a in enumerate(languages):
        for ix_b, lang_b in enumerate(languages):
            # Check availability of result a->b
            results_path_ab = utility.get_results_path(lang_a, lang_b, results_dir, options)
            try:
                df_ab = pd.read_csv(results_path_ab + ".tsv", sep="\t")
                ab = True
                # print(str(lang_a)+","+ str(lang_b) + " available")
                for _, row in df_ab.iterrows():
                    concept = row["CONCEPT"]
                    dist = row[dist_label]
                    distance[concept][(lang_a, lang_b)] = dist
                    forms[concept][lang_a] = row["INPUT"].replace(" ", "")
                    forms[concept][lang_b] = row["TARGET"].replace(" ", "")
                    class0 = row[f"{gold_column}0"]
                    class1 = row[f"{gold_column}1"]
                    if not pd.isnull(class0):
                        cog_class_gold[concept][class0].add(lang_a)
                    if not pd.isnull(class1):
                        cog_class_gold[concept][class1].add(lang_b)
            except:
                ab = False
                # print(str(lang_a)+","+ str(lang_b) + " not available")

            # Check availability of result b->a
            results_path_ba = utility.get_results_path(lang_b, lang_a, results_dir, options)
            try:
                df_ba = pd.read_csv(results_path_ba + ".tsv", sep="\t")
                ba = True
                # print(str(lang_b)+","+ str(lang_a) + " available")
                for _, row in df_ba.iterrows():
                    concept = row["CONCEPT"]
                    dist = row[dist_label]
                    distance[concept][(lang_b, lang_a)] = dist
                    forms[concept][lang_b] = row["INPUT"].replace(" ", "")
                    forms[concept][lang_a] = row["TARGET"].replace(" ", "")
                    class0 = row[f"{gold_column}0"]
                    class1 = row[f"{gold_column}1"]
                    if not pd.isnull(class0):
                        cog_class_gold[concept][class0].add(lang_b)
                    if not pd.isnull(class1):
                        cog_class_gold[concept][class1].add(lang_a)
            except:
                ba = False
                # print(str(lang_b)+","+ str(lang_a) + " not available")
                
            if ab and ba:
                # If a,b and b,a available: take mean and assign to both
                for concept in concepts:
                    if (lang_a, lang_b) in distance[concept] and (lang_b, lang_a) in distance[concept]:
                        m = np.mean([distance[concept][(lang_a, lang_b)], distance[concept][(lang_b, lang_a)]])
                        matrix[concept][ix_a, ix_b] = m
                        matrix[concept][ix_b, ix_a] = m
            elif ab:
                # if only a,b in distance: use it
                for concept in concepts:
                    if (lang_a, lang_b) in distance[concept]:
                        matrix[concept][ix_a, ix_b] = distance[concept][(lang_a, lang_b)]
            elif ba:
                for concept in concepts:
                    # if a,b unavailable, but b,a is available, use it
                    if (lang_b, lang_a) in distance[concept]:
                        matrix[concept][ix_a, ix_b] = distance[concept][(lang_b, lang_a)]
                # print("Pair " + str((lang_a,lang_b)) + "unavailable. Using " + str((lang_a,lang_b)))
            else:
                for concept in concepts:
                    # Unavailable language pairs receive highest distance
                    matrix[concept][ix_a, ix_b] = max_dist
                # print("Pair " + str((lang_a,lang_b)) + "unavailable. No alternative, using max dist.")
    assert len(matrix[concepts[0]]) == len(languages)
    results_df = pd.DataFrame(columns=["Method", "Precision", "Recall", "F"])
    
    for label, method in methods:
        for threshold in thresholds:
            method_threshold = label + "-" + str(threshold)
            print(f"\nMethod: {method_threshold}")
            # Arrays to hold scores for all concepts
            precision_scores = []
            recall_scores = []
            f_scores = []
            
            # Perform clustering based on distance matrix
            for concept in concepts:
                # Retrieve gold standard cognacy data
                gold = dict(cog_class_gold[concept])
                
                # print(concept)
                # Print word form in different languages, to get an impression
                # for lang in forms[concept]:
                    # print(lang + ":" + forms[concept][lang])
                
                # Infer cognate judgments using current clustering method
                judgments = method(threshold, matrix[concept], languages)
                
                # Convert values from list to set
                # print("Judgments: ")
                judgments_inv = invert_key_value(judgments)
                # print(judgments_inv)
                if len(gold) > 0:
                    # print("Gold: ")
                    gold_inv = invert_key_value(gold)
                    # print(gold_inv)
                    
                    # Remove languages which are not in both sets
                    gold_langs = set(gold_inv.keys())
                    judgments_langs = set(judgments_inv.keys())
                    redundant_j = judgments_langs - gold_langs
                    for lang in redundant_j:
                        del judgments_inv[lang]
                    redundant_g = gold_langs - judgments_langs
                    for lang in redundant_g:
                        del gold_inv[lang]
                    
                    # print(gold_inv)
                    # print(judgments_inv)
                    
                    b_prec, b_rec, b_f = evaluate_bcubed(judgments_inv, gold_inv)
                    # print(b_prec,b_rec,b_f)
                    precision_scores.append(b_prec)
                    recall_scores.append(b_rec)
                    f_scores.append(b_f)
                # print("")
            # Compute mean scores over all concepts
            mean_precision = np.mean(precision_scores)
            mean_recall = np.mean(recall_scores)
            mean_f = np.mean(f_scores)
            print("# samples: " + str(len(f_scores)))
            print("Mean B-Cubed precision: " + str(mean_precision))
            print("Mean B-Cubed recall: " + str(mean_recall))
            print("Mean B-Cubed F: " + str(mean_f))
            results_df = results_df.append({"Method":method_threshold, "Precision":mean_precision, "Recall":mean_recall, "F":mean_f}, ignore_index=True)
    results_df = results_df.set_index("Method")
    results_df = results_df.sort_values("F")
    # ?? Also return mean_precision, mean_recall, mean_f
    return results_df


# LexStat cognate detection on entire dataset
def cognate_detection_lexstat(output_path, output_cognates_path, input_type):
    print(" - Detect cognates in entire dataset using LexStat.")
    if os.path.exists(output_cognates_path):
        print(f"Using existing cognates file {output_cognates_path}, nothing is generated.")
        return
    print("Perform cognate classification, this can take a long time!")
    # TODO: Columns are NorthEuraLex-specific (at least classes=list)
    lex = LexStat(output_path,
                  model="sca",
                  segments="token",
                  transcription=input_type,
                  classes="list",
                  langid="doculect")
    
    lex.get_scorer(method="markov")
    lex.cluster(method="lexstat", threshold=0.6, ref="COGNATES_LEXSTAT")
    
    print(f"Output cognates to {output_cognates_path}.")
    output_cognates_path_no_extension = output_cognates_path.split(".")[0]
    lex.output('tsv', filename=output_cognates_path_no_extension, ignore="all", prettify=False)
    
# Average F scores over all language pairs
def best_param_languages(eval_table):
    eval_table["F_mean"] = eval_table.mean(axis=1)
    best_entry = eval_table.loc[eval_table['F_mean'].idxmax()]
    best_param = float(best_entry.name.split("_")[-1])
    return eval_table, best_param
