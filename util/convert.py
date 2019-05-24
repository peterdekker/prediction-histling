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
from lingpy.sequence.sound_classes import ipa2tokens, tokens2class
import re


sounds = ['!', '3', '4', '5', '7', '8', 'C', 'E', 'G',
          'L', 'N', 'S', 'T', 'X', 'Z', 'a', 'b', 'c',
          'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
          'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
          'v', 'w', 'x', 'y', 'z']


def clean_asjp(word):
    """
    Removes ASJP diacritics.
    """
    word = re.sub(r",", "-", word)
    word = re.sub(r"\%", "", word)
    word = re.sub(r"\*", "", word)
    word = re.sub(r"\"", "", word)
    word = re.sub(r".~", "", word)
    word = re.sub(r"(.)(.)(.)\$", r"\2", word)
    word = re.sub(r" ", "-", word)
    return word


def ipa_to_asjp(w):
    """
    Lingpy IPA-to-ASJP converter plus some cleanup.
    This function is called on IPA datasets.
    """
    w = w.replace('\"', '').replace('-', '').replace(' ', '')
    wA = ''.join(tokens2class(ipa2tokens(w, merge_vowels=False), 'asjp'))
    wAA = clean_asjp(wA.replace('0', '').replace('I', '3').replace('H', 'N'))
    asjp = ''.join([x for x in wAA if x in sounds])
    assert len(asjp) > 0
    return asjp
