import numpy as np
import math

import xpsi

from xpsi import Parameter
from xpsi.utils import make_verbose

from astropy.io import fits
from astropy.table import Table
from astropy.table import QTable

nCH_NICER = 1501
nIN_NICER = 3451

# Instrument Class

class CustomInstrument(xpsi.Instrument):

    """ Telescope response, from fits files. """

    def __call__(self, signal, *args):
        """ Overwrite base just to show it is possible.

        We loaded only a submatrix of the total instrument response
        matrix into memory, so here we can simplify the method in the
        base class.

        """
        matrix = self.construct_matrix()

        self._folded_signal = np.dot(matrix, signal)

        return self._folded_signal

    @classmethod
    def from_response_files(cls, ARF, RMF, channel_edges,max_input,
                            min_input=0,channel=[1,1500],
                            ):
        """ Constructor which converts response files into :class:`numpy.ndarray`s.
        :param str ARF: Path to ARF which is compatible with
                                :func:`numpy.loadtxt`.
        :param str RMF: Path to RMF which is compatible with
                                :func:`numpy.loadtxt`.
        :param str channel_edges: Path to edges which is compatible with
                                  :func:`numpy.loadtxt`.
        """

        if min_input != 0:
            min_input = int(min_input)

        max_input = int(max_input)

        matrix = np.ascontiguousarray(RMF[min_input:max_input,channel[0]:channel[1]].T, dtype=np.double)

        edges = np.zeros(ARF[min_input:max_input,2].shape[0]+1, dtype=np.double)

        edges[0] = ARF[min_input,0]; edges[1:] = ARF[min_input:max_input,1]

        for i in range(matrix.shape[0]):
            matrix[i,:] *= ARF[min_input:max_input,2]


        channels = np.arange(channel[0],channel[1])

        return cls(matrix, edges, channels, channel_edges[channel[0]:channel[1]+1,1])

    
    @classmethod
    @make_verbose('Loading response matrix',
                  'Response matrix loaded')
    def from_ogip_fits(cls, bounds, values, ARF, RMF, **kwargs):
        
        # Extract values
        min_input = input[0]
        max_input = input[1]
        min_channel = channel[0]
        max_channel = channel[1]

        if min_input != 0:
            min_input = int(min_input)
        max_input = int(max_input)

        # READ ARF
        ARFdata = Table.read(ARF, 'SPECRESP')

        # READ RMF
        RMFdata = Table.read(RMF, 'MATRIX')
        f_channels = RMFdata['F_CHAN']
        n_channels = RMFdata['N_CHAN']

        if min_channel == 0:
            min_channel = 1 #the first channel is channel 1
        
        # Make the edges and initiate matrix
        try:
            channels = np.arange( min_channel, max_channel )
            channels_edges = np.arange( min_channel, max_channel+1 ) * 1e-2
            energy_edges = RMFdata['ENERG_LO'][min_input:max_input+1]
            matrix = np.zeros(shape = [max_channel-min_channel,max_input-min_input])
        except:
            raise ValueError

        # Loop over RMF inputs
        for i in range(min_input,max_input):

            # Extract relevant values
            f_channel = f_channels[i]
            n_channel = n_channels[i]
            RMF_line = RMFdata['MATRIX'][i]

            # Get indexes
            if (f_channel+n_channel>=min_channel) and (f_channel<=max_channel):

                # Channel indexes
                ch_i = 0 if f_channel>=min_channel else min_channel - f_channel
                ch_f = n_channel if f_channel+n_channel<= max_channel else max_channel-f_channel#-1
                
                ch_i_m = 0 if f_channel<= min_channel else f_channel- min_channel
                ch_f_m = max_channel-min_channel if max_channel<f_channel+n_channel else f_channel+n_channel- min_channel#-1
                
                # print( ch_i, ch_f, ch_i_m, ch_f_m )
                matrix[ch_i_m:ch_f_m,i-min_input] += RMF_line[ch_i:ch_f] * ARFdata['SPECRESP'][i]
        
        alpha = Parameter('alpha',
                          strict_bounds = (0.1,1.9),
                          bounds = bounds.get('alpha', None),
                          doc='NICER energy-independent scaling factor',
                          symbol = r'$\alpha_{\rm NICER}$',
                          value = values.get('alpha', None))

        return cls(matrix, energy_edges, channels, channels_edges, alpha, **kwargs)