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

# Script to convert ASJP to IPA using input prompt
from asjp2ipa import asjp_to_ipa

while(True):
    input_word = input("String to convert from ASJP to IPA: ")
    print(asjp_to_ipa(input_word))
