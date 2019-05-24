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
# taxa = ['German','Swedish','Icelandic','English','Dutch']
# matrix = squareform([0.5,0.67,0.8,0.2,0.4,0.7,0.6,0.8,0.8,0.3])
# >>> matrix
#     DE   SE    IC    EN   NL
# DE [[0.0, 0.5, 0.67, 0.8, 0.2],
# SE [0.5, 0.0, 0.4, 0.7, 0.6],
# IC [0.67, 0.4, 0.0, 0.8, 0.8],
# EN [0.8, 0.7, 0.8, 0.0, 0.3],
# NL [0.2, 0.6, 0.8, 0.3, 0.0]]

from lingpy import *
import numpy as np


def cluster_languages(lang_pairs, distances_path, output_path, distance_col=2):
    languages = list(set([e for pair in lang_pairs for e in pair]))
    distance = {}
    with open(distances_path + ".txt", "r") as distance_file:
        for line in distance_file:
            split_line = line.rstrip("\n").split(",")
            lang_a = split_line[0]
            lang_b = split_line[1]
            # Only add languages we are taking into consideration
            # Other languages (from other runs) could also have been added to this file
            if lang_a in languages and lang_b in languages:
                # Only use designated distance column, there are multiple for baseline
                dist = split_line[distance_col]
                dist = float(dist)
                distance[(lang_a, lang_b)] = dist
    max_dist = max(distance.values())
    n_langs = len(languages)
    print(sorted(distance.items(), key=lambda x: x[0]))
    matrix = np.zeros((n_langs, n_langs))
    for ix_a, lang_a in enumerate(languages):
        row = []
        for ix_b, lang_b in enumerate(languages):
            if lang_a == lang_b:
                d = 0.0
            elif (lang_a, lang_b) in distance and (lang_b, lang_a) in distance:
                # If a,b and b,a available: take mean and assign to both
                m = np.mean([distance[(lang_a, lang_b)], distance[(lang_b, lang_a)]])
                matrix[ix_a, ix_b] = m
                matrix[ix_b, ix_a] = m
            elif (lang_a, lang_b) in distance:
                # if only a,b in distance: use it
                matrix[ix_a, ix_b] = distance[(lang_a, lang_b)]
            elif (lang_b, lang_a) in distance:
                # if a,b unavailable, but b,a is available, use it
                matrix[ix_a, ix_b] = distance[(lang_b, lang_a)]
                print("Pair " + str((lang_a, lang_b)) + " unavailable. Using " + str((lang_a, lang_b)))
            else:
                # Unavailable language pairs receive highest distance
                print("Pair " + str((lang_a, lang_b)) + " unavailable. No alternative, using max dist.")
                matrix[ix_a, ix_b] = max_dist
    assert len(matrix) == len(languages)
    
    languages_short = [l[:3] for l in languages]
    
    # Output formatted distance matrix to file
    with open(output_path + ".tex", "w") as distances_tex:
        distances_tex.write("&" + "&".join(languages_short) + "\\\\\n")
        for row in np.arange(len(matrix)):
            dists_string = ["{0:.2f}".format(v) for v in matrix[row]]
            distances_tex.write(languages_short[row] + "&" + "&".join(dists_string) + "\\\\\n")
    
    # Create phylogenetic trees and output to file
    trees = [("UPGMA", upgma(matrix, languages_short)), ("Neighbor joining", neighbor(matrix, languages_short))]
    for label, tree in trees:
        print(label)
        tree = Tree(tree)
        print(tree)
        print(tree.asciiArt())
        print("")
        with open(output_path + "_" + "_".join(label.split()) + ".tree", "w") as tree_file:
            tree_file.write(str(tree))
    
    # Calculate mean over all languages and output to file
    # Use original 'distance' dict, not 'matrix' which has been edited
    mean_distance = np.mean(distance.values())
    print("Mean distance of all lang pairs: " + "{0:.4f}".format(mean_distance))
    with open(output_path + ".mean", "w") as mdist_file:
        mdist_file.write(str(mean_distance))
    
    with open(distances_path + ".txt", "r") as f:
        print(f.read())
            
