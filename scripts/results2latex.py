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
from collections import OrderedDict
import pandas as pd
import sys


if len(sys.argv) > 1:
    filename = sys.argv[1]
    df = pd.read_csv(sys.argv[1], sep="\t")
    rel_cols = ["INPUT", "TARGET", "PREDICTION"]
    new_dict = OrderedDict()
    for col in rel_cols:
        new_col = []
        for item in df[col]:
            new_col.append(item.replace(" ", ""))
        new_dict[col] = new_col
    new_df = pd.DataFrame(new_dict)
    new_df.to_csv("latex_output.txt", sep="&", index=False, line_terminator="\\\\\n")
