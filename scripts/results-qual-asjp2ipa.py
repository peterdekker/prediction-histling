#-------------------------------------------------------------------------------
# Copyright (C) 2020 Peter Dekker
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

# Script to convert results tables for the paper, in ASJP, to IPA
# Input: LaTeX file with four columns: Input, Target, Prediction, Distance

import pandas as pd
from astropy.table import Table
from asjp2ipa import asjp_to_ipa

filename = "20200617-qualitative"

df = Table.read(f"{filename}.tex").to_pandas()
#df = pd.read_csv(f"{filename}.tex",sep="&", skipfooter=1, engine="python")
df["Input"] = df["Input"].apply(asjp_to_ipa)
df["Target"] = df["Target"].apply(asjp_to_ipa)
df["Prediction"] = df["Prediction"].apply(asjp_to_ipa)
df.to_latex(f"{filename}-converted.tex", index=False)