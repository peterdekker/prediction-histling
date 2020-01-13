# -------------------------------------------------------------------------------
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
# -------------------------------------------------------------------------------
import pandas as pd
from util import utility


distances = pd.read_csv("distances.txt", header=None)
baselines = pd.read_csv("baselines.txt", header=None)
new = pd.DataFrame()
new["lang1"] = distances[0].apply(utility.short_lang)
new["lang2"] = distances[1].apply(utility.short_lang)
new["distance"] = distances[2]
new["baseline_source"] = baselines[2]
new["baseline_firstorder"] = baselines[3]

i = 0
j = 0
for row in new.iterrows():
    j += 1
    if(row[1]["distance"] < row[1]["baseline_source"] and row[1]["distance"] < row[1]["baseline_firstorder"]):
        i += 1
print(i)
print(j)
new.sort_values(by="distance", inplace=True)
print(new)
new.to_csv("distances_baselines.txt", index=False, sep="&", line_terminator="\\\\\n", float_format="%.2f")
