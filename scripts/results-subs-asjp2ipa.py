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

# Script to convert rsubstituion table for the paper, in ASJP, to IPA
# Input: LaTeX file with four columns: Substitution 1,Substitution 2,Source-prediction frequency,Source-target frequency

import sys, os
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from util.asjp2ipa import asjp_to_ipa

import pandas as pd
from astropy.table import Table

filename = "20200617-subs"


df = Table.read(f"{filename}.tex").to_pandas()
#df = pd.read_csv(f"{filename}.tex",sep="&", skipfooter=1, engine="python")
df["Substitution 1"] = df["Substitution 1"].apply(lambda x: x if x=="-" else asjp_to_ipa(x))
df["Substitution 2"] = df["Substitution 2"].apply(lambda x: x if x=="-" else asjp_to_ipa(x))
df.to_latex(f"{filename}-converted.tex", index=False)