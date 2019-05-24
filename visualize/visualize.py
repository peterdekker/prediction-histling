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
import pickle
import cd
from collections import defaultdict
from lingpy.algorithm.clustering import flat_upgma, fuzzy, link_clustering, mcl
from lingpy.algorithm.extra import affinity_propagation, infomap_clustering
from lingpy.align.pairwise import sw_align, nw_align
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import mode
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
import utility


# from sklearn.manifold import TSNE
def show_output_substitutions(results_path, subs_st_path, subs_sp_path):
    source_target_subs = defaultdict(int)
    source_predicted_subs = defaultdict(int)
    df = pd.read_csv(results_path + ".tsv", sep="\t")
    for index, row in df.iterrows():
        source_tokens = row["INPUT"].split()
        target_tokens = row["TARGET"].split()
        predicted_tokens = row["PREDICTION"].split()
        # Defaultdicts are re-used every run of the loop
        source_target_subs = find_substitutions(source_tokens, target_tokens, source_target_subs)
        source_predicted_subs = find_substitutions(source_tokens, predicted_tokens, source_predicted_subs)
    
    st_df = pd.DataFrame.from_dict(source_target_subs, orient="index")
    st_df.columns = ["Source-target"]
    sp_df = pd.DataFrame.from_dict(source_predicted_subs, orient="index")
    sp_df.columns = ["Source-prediction"]
    joined_df = sp_df.join(st_df, how="outer")
    joined_df = joined_df.sort_values("Source-prediction", ascending=False)
    print(joined_df)
    joined_df.to_csv("output/subs.tex", sep="&", line_terminator="\\\\\n", float_format="%.0f")
    # write_subs(source_target_subs, subs_st_path+".tex")
    # write_subs(source_predicted_subs, subs_sp_path+".tex")
    
    # st_df = pd.DataFrame.from_dict(source_target_subs,orient="index")
    # sp_df = pd.DataFrame.from_dict(source_predicted_subs,orient="index")
    # st_df.to_csv("subs_st_"+results_path+".tex", sep="&", line_terminator="\\\\")
    # sp_df.to_csv("subs_sp_"+results_path+".tex", sep="&", line_terminator="\\\\")


def write_subs(subs, subs_filename):
    with open(subs_filename, "w") as subs_file:
        sorted_list = sorted(subs.items(), key=lambda x: x[1], reverse=True)
        for k, v in sorted_list:
            subs_file.write("&".join(k) + "&" + str(v) + "\\\\\n")


def find_substitutions(seq1, seq2, subs):
    # Perform the Needleman-Wunsch algorithm for local alignment
    seq1_align, seq2_align, _ = nw_align(seq1, seq2)
    assert len(seq1_align) == len(seq2_align)
    for i in np.arange(len(seq1_align)):
        if seq1_align[i] != seq2_align[i]:
            token1 = "".join(seq1_align[i])
            token2 = "".join(seq2_align[i])
            subs[token1 + "&" + token2] += 1
    return subs


def visualize_encoding(emb_matrix, phon_matrix, langs, results_dir, methods=[("Affinity propagation", affinity_propagation)], thresholds=[0.2]):
    lang_a, lang_b = langs
    emb_matrix = emb_matrix.drop(["."])
    phon_matrix = phon_matrix.drop(["."])
    
    phonemes_emb = emb_matrix.index
    phonemes_phon = phon_matrix.index
    
    #### Distance analysis
    # # Compute pairwise distances between words inside one matrix: context, input and target
    # pw_metric = "cosine"
    # emb_dist = pairwise_distances(emb_matrix, metric=pw_metric)
    # phon_dist = pairwise_distances(phon_matrix, metric=pw_metric)
    
    # # Compare distance matrices
    # dist_matrices = [("emb_dist",emb_dist),("phon_dist",phon_dist)]
    # dist_mat_pairs = utility.generate_pairs(dist_matrices, allow_permutations=False)
    
    # # Perform clusterings based on distance matrices
    # settings_df = pd.DataFrame(columns = ["Method","Threshold", "Context_min","Context_med","Context_max","Input_min","Input_med","Input_max","Target_min", "Target_med", "Target_max"])
    # for meth_label, method in methods:
        # for threshold in thresholds:
            # mcs = {}
            # clusters = {}
            # method_threshold = meth_label + "-" + str(threshold)
            # print(method_threshold)
            # for mat_label,mat in dist_matrices:
                # clusters[mat_label] = method(threshold, mat, input_words)
                # cluster_sizes = [len(x) for x in clusters[mat_label].values()]
                # # Save mean cluster size for this setting
                # mcs[mat_label] = np.min(cluster_sizes), np.median(cluster_sizes), np.max(cluster_sizes), 
                # # Nicely print clusters
                # print(mat_label)
                # for cluster in clusters[mat_label].values():
                    # print(" " + " ".join(cluster))
                # print("")
            # settings_df = settings_df.append({"Method":meth_label,"Threshold":str(threshold),"Context_min":mcs["vectors_dist"][0], "Context_med":mcs["vectors_dist"][1], "Context_max":mcs["vectors_dist"][2], "Input_min":mcs["input_dist"][0], "Input_med":mcs["input_dist"][1], "Input_max":mcs["input_dist"][2],"Target_min":mcs["target_dist"][0], "Target_med":mcs["target_dist"][1], "Target_max":mcs["target_dist"][2]}, ignore_index=True)
            
            # # For this parameter settings:
            # # compute B-Cubed scores between clusterings,
            # # for all combinations of distance matrices
            # comparison_df = pd.DataFrame(columns = ["Judgments", "Gold", "Precision","Recall","F"])
            # for (label1,_),(label2,_) in dist_mat_pairs:
                # clusters1_inv = cd.invert_key_value(clusters[label1])
                # clusters2_inv = cd.invert_key_value(clusters[label2])
                # P,R,F = cd.evaluate_bcubed(judgments=clusters1_inv,gold=clusters2_inv)
                # comparison_df = comparison_df.append({"Judgments":label1,"Gold":label2,"Precision":P,"Recall":R,"F":F}, ignore_index=True)
            # print(comparison_df)
            # # Only comparison_df for last method is outputted
            # # So this works only if you pick a specific method and threshold
            # comparison_df.to_csv(context_vectors_path + "_".join(method_threshold.split()) + ".tex", index=False, sep="&", line_terminator="\\\\\n", float_format="%.3f")
            # print("")
    
    # # Show cluster sizes per clustering method, to determine best method
    # print("")
    # print("Mean cluster size for different clustering methods:")
    # max_threshold=100
    # settings_df = settings_df.sort_values("Context_med")[(settings_df["Context_med"]>1) & (settings_df["Input_med"]>1) & (settings_df["Target_med"]>1) & (settings_df["Context_max"] <max_threshold) & (settings_df["Input_max"] <max_threshold) & (settings_df["Target_max"] <max_threshold)]
    # print(settings_df)
    # settings_df.to_csv(context_vectors_path +  "mcs.tex", index=False, sep="&", line_terminator="\\\\\n", float_format="%.1f")
    # print("")
    
    # # Compute distances between distance matrices
    # print("Direct distances between distance matrices")
    # for (label1,m1),(label2,m2) in dist_mat_pairs:
        # dist_dist = np.zeros(n_words)
        # # Compare matrices per row (word)
        # for w in np.arange(n_words):
            # # Use cosine distance: magnitude should not play a role
            # dist_dist[w] = cosine(m1[w],m2[w])
        # print(str(label1) + "," + str(label2) +": " + str(np.mean(dist_dist)))
    
    # ## Perform PCA dimensionality reduction and plot
    dim_methods = [("PCA", PCA())]  # , ("t-SNE",TSNE())]
    for meth_label, method in dim_methods:
        # Create plot of encoded input to neural network
        inp_filename = os.path.join(results_dir, "embmatrix-" + lang_a + "-" + meth_label)
        fit_plot(emb_matrix, phonemes_emb, method, "", inp_filename, encoding_plot=True)
        
        # Create plot of encoded target to neural network
        tar_filename = os.path.join(results_dir, "phonmatrix-" + meth_label)
        fit_plot(phon_matrix, phonemes_phon, method, "", tar_filename, encoding_plot=True)


def visualize_weights(context_vectors_path, langs, input_encoding, output_encoding, results_dir, sample=50, methods=[("Affinity propagation", affinity_propagation)], thresholds=[0.2]):
    lang_a, lang_b = langs
    # Load file with context vectors, created during word prediction
    with open(context_vectors_path + ".p", "rb") as f:
        context_vectors = pickle.load(f)
    vectors, input_words, target_words, input_raw, target_raw = context_vectors
    input_words = ["".join(word) for word in input_words]
    target_words = ["".join(word) for word in target_words]
    if sample:
        vectors = vectors[:sample]
        input_words = input_words[:sample]
        target_words = target_words[:sample]
        input_raw = input_raw[:sample]
        target_raw = target_raw[:sample]
    # Convert list of NP arrays to one NP array
    vectors = np.array(vectors)
    print("vectors" + str(vectors.shape))
    input_raw = np.array(input_raw)
    print("input_raw" + str(input_raw.shape))
    target_raw = np.array(target_raw)
    print("target_raw" + str(target_raw.shape))
    # Merge dimensions 1 and 2: width of feature matrix and dimensionality of network
    vectors = flatten_arr(vectors)
    input_raw = flatten_arr(input_raw)
    target_raw = flatten_arr(target_raw)
    print("After flatten")
    print("vectors" + str(vectors.shape))
    input_raw = np.array(input_raw)
    print("input_raw" + str(input_raw.shape))
    target_raw = np.array(target_raw)
    print("target_raw" + str(target_raw.shape))
    
    n_words = len(input_words)
    print("Number of words: " + str(n_words))
    
    # ## Distance analysis
    # Compute pairwise distances between words inside one matrix: context, input and target
    pw_metric = "cosine"
    vectors_dist = pairwise_distances(vectors, metric=pw_metric)
    input_dist = pairwise_distances(input_raw, metric=pw_metric)
    target_dist = pairwise_distances(target_raw, metric=pw_metric)
    
    # Compare distance matrices
    dist_matrices = [("vectors_dist", vectors_dist), ("input_dist", input_dist), ("target_dist", target_dist)]
    dist_mat_pairs = utility.generate_pairs(dist_matrices, allow_permutations=False)
    
    # Perform clusterings based on distance matrices
    settings_df = pd.DataFrame(columns=["Method", "Threshold", "Context_min", "Context_med", "Context_max", "Input_min", "Input_med", "Input_max", "Target_min", "Target_med", "Target_max"])
    for meth_label, method in methods:
        for threshold in thresholds:
            mcs = {}
            clusters = {}
            method_threshold = meth_label + "-" + str(threshold)
            print(method_threshold)
            for mat_label, mat in dist_matrices:
                clusters[mat_label] = method(threshold, mat, input_words)
                cluster_sizes = [len(x) for x in clusters[mat_label].values()]
                # Save mean cluster size for this setting
                mcs[mat_label] = np.min(cluster_sizes), np.median(cluster_sizes), np.max(cluster_sizes),
                # Nicely print clusters
                print(mat_label)
                for cluster in clusters[mat_label].values():
                    print(" " + " ".join(cluster))
                print("")
            settings_df = settings_df.append({"Method":meth_label, "Threshold":str(threshold), "Context_min":mcs["vectors_dist"][0], "Context_med":mcs["vectors_dist"][1], "Context_max":mcs["vectors_dist"][2], "Input_min":mcs["input_dist"][0], "Input_med":mcs["input_dist"][1], "Input_max":mcs["input_dist"][2], "Target_min":mcs["target_dist"][0], "Target_med":mcs["target_dist"][1], "Target_max":mcs["target_dist"][2]}, ignore_index=True)
            
            # For this parameter settings:
            # compute B-Cubed scores between clusterings,
            # for all combinations of distance matrices
            comparison_df = pd.DataFrame(columns=["Judgments", "Gold", "Precision", "Recall", "F"])
            for (label1, _), (label2, _) in dist_mat_pairs:
                clusters1_inv = cd.invert_key_value(clusters[label1])
                clusters2_inv = cd.invert_key_value(clusters[label2])
                P, R, F = cd.evaluate_bcubed(judgments=clusters1_inv, gold=clusters2_inv)
                comparison_df = comparison_df.append({"Judgments":label1, "Gold":label2, "Precision":P, "Recall":R, "F":F}, ignore_index=True)
            print(comparison_df)
            # Only comparison_df for last method is outputted
            # So this works only if you pick a specific method and threshold
            comparison_df.to_csv(context_vectors_path + "_".join(method_threshold.split()) + ".tex", index=False, sep="&", line_terminator="\\\\\n", float_format="%.3f")
            print("")
    
    # Show cluster sizes per clustering method, to determine best method
    print("")
    print("Mean cluster size for different clustering methods:")
    max_threshold = 100
    settings_df = settings_df.sort_values("Context_med")[(settings_df["Context_med"] > 1) & (settings_df["Input_med"] > 1) & (settings_df["Target_med"] > 1) & (settings_df["Context_max"] < max_threshold) & (settings_df["Input_max"] < max_threshold) & (settings_df["Target_max"] < max_threshold)]
    print(settings_df)
    settings_df.to_csv(context_vectors_path + "mcs.tex", index=False, sep="&", line_terminator="\\\\\n", float_format="%.1f")
    print("")
    
    # Compute distances between distance matrices
    print("Direct distances between distance matrices")
    for (label1, m1), (label2, m2) in dist_mat_pairs:
        dist_dist = np.zeros(n_words)
        # Compare matrices per row (word)
        for w in np.arange(n_words):
            # Use cosine distance: magnitude should not play a role
            dist_dist[w] = cosine(m1[w], m2[w])
        print(str(label1) + "," + str(label2) + ": " + str(np.mean(dist_dist)))
    
    # ## Perform PCA dimensionality reduction and plot
    dim_methods = [("PCA", PCA())]  # , ("t-SNE",TSNE())]
    for meth_label, method in dim_methods:
        # Create plot of context vectors of neural network
        cont_title = "Context vectors " + lang_a + "->" + lang_b + ", " + input_encoding + " enc, " + meth_label
        cont_filename = os.path.join(results_dir, "context_" + lang_a + "-" + lang_b + "_" + input_encoding + "_" + meth_label)
        fit_plot(vectors, input_words, method, cont_title, cont_filename, target_words_label=target_words)
    
        # Create plot of encoded input to neural network
        inp_title = "Input " + lang_a + "->" + lang_b + ", " + input_encoding + " enc, " + meth_label
        inp_filename = os.path.join(results_dir, "inp_" + lang_a + "-" + lang_b + "_" + input_encoding + "_" + meth_label)
        fit_plot(input_raw, input_words, method, inp_title, inp_filename)
        
        # Create plot of encoded target to neural network
        tar_title = "Target " + lang_a + "->" + lang_b + ", " + output_encoding + " enc, " + meth_label
        tar_filename = os.path.join(results_dir, "tar_" + lang_a + "-" + lang_b + "_" + input_encoding + "_" + meth_label)
        fit_plot(target_raw, target_words, method, tar_title, tar_filename)
        
        # ## Create baseline plot of distance matrix of input words
        # dist_inp_title = "Edit distance between input words, " + lang_a + "->" + lang_b + ", " + meth_label
        # dist_inp_filename = os.path.join(results_dir,"dist_inp_"+lang_a+"-" + lang_b + "_" + meth_label)
        # fit_plot(dist_matrix_inp, input_words, method, dist_inp_title, dist_inp_filename)
        
        # ## Create baseline plot of distance matrix of output words
        # dist_tar_title = "Edit distance between target words, " + lang_a + "->" + lang_b + ", " + meth_label
        # dist_tar_filename = os.path.join(results_dir,"dist_tar_"+lang_a+"-" + lang_b + "_" + meth_label)
        # fit_plot(dist_matrix_tar, target_words, method, dist_tar_title, dist_tar_filename)
    
    
def fit_plot(vectors, input_words, method, title, filename, target_words_label=None, encoding_plot=False):
    # Perform dimensionality reduction: from 1000+ dimensions to 2 dimensions
    vectors_2d = method.fit_transform(vectors)
    # Seperate x and y vectors
    x = vectors_2d[:, 0]
    y = vectors_2d[:, 1]
    # Create scatter plot and add input words to data points
    if encoding_plot:
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.scatter(x, y, s=0)
    else:
        fig, ax = plt.subplots()
        ax.scatter(x, y)
    for i, inp_word in enumerate(input_words):
        annotation = inp_word
        if target_words_label:
            annotation += "\n" + target_words_label[i]
        if encoding_plot:
            ax.annotate(annotation, (x[i], y[i]), xycoords="data", textcoords="data", size=15)
        else:
            ax.annotate(annotation, (x[i], y[i]), size=9)
            
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    if len(title) > 0:
        plt.title(title)
    plt.savefig(filename)
    plt.close()


# Creates an edit distance matrix between words, 
def create_dist_matrix_words(input_words):
    # Create baseline based on edit distance
    dist_df = pd.DataFrame(columns=input_words, index=input_words)
    for word1 in input_words:
        for word2 in input_words:
            dist_df.loc[word1, word2] = utility.calculate_levenshtein(word1, word2)
    return dist_df


def flatten_arr(vectors):
    if len(vectors.shape) == 3:
        multipl = vectors.shape[1] * vectors.shape[2]
    if len(vectors.shape) == 4:
        multipl = vectors.shape[1] * vectors.shape[2] * vectors.shape[3]
    return vectors.reshape(vectors.shape[0], multipl)

