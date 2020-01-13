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
import lasagne
from lasagne.layers import LSTMLayer, GRULayer


def gated_layer(incoming, num_units, grad_clipping, only_return_final, backwards, gated_layer_type, mask_input=None, cell_init=lasagne.init.Constant(0.), hid_init=lasagne.init.Constant(0.), resetgate=lasagne.layers.Gate(W_cell=None), updategate=lasagne.layers.Gate(W_cell=None), hidden_update=lasagne.layers.Gate(W_cell=None, nonlinearity=lasagne.nonlinearities.tanh), name=None):
    if gated_layer_type == "gru":
        return GRULayer(incoming, num_units, mask_input=mask_input,
                        grad_clipping=grad_clipping,
                        only_return_final=only_return_final, backwards=backwards,
                        hid_init=hid_init,
                        resetgate=resetgate,
                        updategate=updategate,
                        hidden_update=hidden_update,
                        name=name)
    else:
        return LSTMLayer(incoming, num_units, mask_input=mask_input,
                         grad_clipping=grad_clipping,
                         nonlinearity=lasagne.nonlinearities.tanh,
                         only_return_final=only_return_final, backwards=backwards,
                         cell_init=cell_init,
                         hid_init=hid_init,
                         resetgate=resetgate,
                         updategate=updategate,
                         hidden_update=hidden_update,
                         name=name)
