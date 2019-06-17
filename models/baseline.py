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
#from numpy import *
import numpy as np
import pandas as pd
import sys
from util import utility


# Baseline code by Gerhard Jaeger
def compute_baseline(lang_a, lang_b, sounds, training, testing, baselines_path):
    # Merge input and output sounds sets
    sounds = list(set(sounds[0] + sounds[1]))
    levdict = pd.DataFrame(-1., index=sounds, columns=sounds)
    for s in sounds: levdict[s][s] = 0
        
    gp1 = -2.49302792222
    gp2 = -1.70573165621
    
    pmi = estimatePMI(training, levdict, -1, -1, sounds)

    for _ in range(10):
        pmi = estimatePMI(training, pmi, gp1, gp2, sounds)

    alignments = np.array([nw(w1, w2, pmi, gp1, gp2)[1]
                        for (w1, w2) in training.values])

    a = np.array([np.concatenate(list(map(list, alignments[:, 0]))),
               np.concatenate(list(map(list, alignments[:, 1])))])

    aCounts = pd.crosstab(a[0], a[1])

    soundOccs2 = pd.value_counts(a[1])
    testing['prediction'] = [prediction(w, aCounts, soundOccs2) for w in testing.values[:, 0]]
    
    baseline = np.array([ldn(x, y)
                  for (x, y) in testing.values[:, [0, 1]]])

    model = np.array([ldn(str(x), str(y))
                   for (x, y) in testing.values[:, [2, 1]]])
    baseline_str = str(np.mean(baseline))
    fo_model_str = str(np.mean(model))
    print('baseline: ' + baseline_str)
    print('first order model: ' + fo_model_str)
    # Write distances to file
    with open(baselines_path, "a") as f:
        f.write(lang_a + "," + lang_b + "," + baseline_str + "," + fo_model_str + "\n")


def nw(x, y, lodict, gp1, gp2):
    """
    Needleman-Wunsch algorithm for pairwise string alignment
    with affine gap penalties.
    'lodict' must be a pandas data frame with all symbols
    as index and as columns
    and match scores as values.
    gp1 and gp2 are gap penalties for opening/extending a gap.
    Returns the alignment score and one optimal alignment.
    """
    n, m = len(x), len(y)
    dp = np.zeros((n + 1, m + 1))
    pointers = np.zeros((n + 1, m + 1), int)
    for i in range(1, n + 1):
        dp[i, 0] = dp[i - 1, 0] + (gp2 if i > 1 else gp1)
        pointers[i, 0] = 1
    for j in range(1, m + 1):
        dp[0, j] = dp[0, j - 1] + (gp2 if j > 1 else gp1)
        pointers[0, j] = 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = dp[i - 1, j - 1] + lodict.ix[x[i - 1]][y[j - 1]]
            insert = dp[i - 1, j] + (gp2 if pointers[i - 1, j] == 1 else gp1)
            delet = dp[i, j - 1] + (gp2 if pointers[i, j - 1] == 2 else gp1)
            dp[i, j] = max([match, insert, delet])
            pointers[i, j] = np.argmax([match, insert, delet])
    alg = []
    i, j = n, m
    while(i > 0 or j > 0):
        pt = pointers[i, j]
        if pt == 0:
            i -= 1
            j -= 1
            alg = [[x[i], y[j]]] + alg
        if pt == 1:
            i -= 1
            alg = [[x[i], '-']] + alg
        if pt == 2:
            j -= 1
            alg = [['-', y[j]]] + alg
    return dp[-1, -1], np.array([''.join(x) for x in np.array(alg).T])


def estimatePMI(training, lodict, gp1, gp2, sounds):
    alignments = np.array([nw(w1, w2, lodict, gp1, gp2)[1]
                        for (w1, w2) in training.values])
    a = np.array([np.concatenate(list(map(list, alignments[:, 0]))),
               np.concatenate(list(map(list, alignments[:, 1])))])
    aCounts = pd.crosstab(a[0], a[1]).reindex(sounds, fill_value=0)
    aCounts = aCounts.T.reindex(sounds, fill_value=0).T
    aCounts[aCounts == 0] = 1e-4
    aCounts /= aCounts.sum().sum()
    soundOccs1 = pd.value_counts(a[0]).reindex(sounds, fill_value=1e-2)
    soundOccs2 = pd.value_counts(a[1]).reindex(sounds, fill_value=1e-2)
    soundOccs1 /= soundOccs1.sum()
    soundOccs2 /= soundOccs2.sum()
    pmi = ((np.log(aCounts) - np.log(soundOccs2)).T - np.log(soundOccs1)).T
    return pmi


def prediction(w, aCounts, soundOccs2):
    return ''.join([aCounts.ix[s].argmax()
                    if s in aCounts.index
                    else soundOccs2.argmax()
                    for s in w])


def ldn(x, y):
    return utility.calculate_levenshtein(x, y)
