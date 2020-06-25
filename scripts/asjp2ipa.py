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

# Helper function for all results-*-asjp2ipa.py scripts
# For this script to work, CLTS data directory must be set: https://github.com/cldf-clts/pyclts#install

from pyclts import CLTS

clts = CLTS("asdf")
asjp = clts.transcriptionsystem('asjpcode')

def asjp_to_ipa(word):
    word_spaced = " ".join(word)
    translated = asjp.translate(word_spaced, clts.bipa)
    translated_nospace = "".join(translated.split(" "))
    return translated_nospace
