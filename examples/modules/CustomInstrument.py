""" Instrument module for X-PSI modelling. Includes loading of any instrument's response."""

import os
import numpy as np
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt

import xpsi

from xpsi import Parameter
from xpsi.utils import make_verbose
from xpsi.Instrument import ResponseError

class CustomInstrument(xpsi.Instrument):

    """ XTI, PN, MOS1, and MOS2. """
    def construct_matrix(self):
        """ Implement response matrix parameterisation. """
        matrix = self['energy_independent_effective_area_scaling_factor'] * self.matrix
        matrix[matrix < 0.0] = 0.0

        return matrix

    def __call__(self, signal, *args):
        """ Overwrite. """

        matrix = self.construct_matrix()
        self._cached_signal = np.dot(matrix, signal)

        return self._cached_signal

    @classmethod
    @make_verbose('Loading instrument response matrix',
                  'Response matrix loaded')
    def from_ogip_fits(cls,
              ARF_path,
              RMF_path,
              min_channel=0,
              max_channel=-1,
              min_input=1,
              max_input=-1,
              bounds=dict(),
              values=dict(),
              datafolder=None,
              **kwargs):
        
        """ Load any instrument response matrix. """

        if datafolder:
            ARF_path = os.path.join( datafolder, ARF_path )
            RMF_path = os.path.join( datafolder, RMF_path )

        # Open useful values in ARF/RMF    
        with fits.open( ARF_path ) as ARF_hdul:
            ARF_instr = ARF_hdul['SPECRESP'].header['INSTRUME']
            
        with fits.open( RMF_path ) as RMF_hdul:
            RMF_header = RMF_hdul['MATRIX'].header
        RMF_instr = RMF_header['INSTRUME'] 
        DETCHANS = RMF_header['DETCHANS']
        NUMGRP = RMF_header['NAXIS2']
        TLMIN = RMF_header['TLMIN4']
        TLMAX = RMF_header['TLMAX4']

        # Get the values and change the -1 values if requried
        if max_channel == -1:
            max_channel = DETCHANS -1
        if max_input == -1:
            max_input = NUMGRP
        channels = np.arange( min_channel, max_channel+1 )
        inputs = np.arange( min_input, max_input+1  )

        # Perform routine checks
        assert ARF_instr == RMF_instr
        assert min_channel >= TLMIN and max_channel <= TLMAX
        assert min_input >= 0 and max_input <= NUMGRP

        # If everything in order, get the data
        with fits.open( RMF_path ) as RMF_hdul:
            RMF_MATRIX = RMF_hdul['MATRIX'].data
            RMF_EBOUNDS = RMF_hdul['EBOUNDS'].data

        # Get the channels from the data
        RMF = np.zeros((DETCHANS, NUMGRP))
        for i, (N_GRP, F_CHAN, N_CHAN, RMF_line) in enumerate( zip(RMF_MATRIX['N_GRP'], RMF_MATRIX['F_CHAN'], RMF_MATRIX['N_CHAN'], RMF_MATRIX['MATRIX']) ):

            # Skip if needed
            if N_GRP == 0:
                continue

            # Check the values
            if not isinstance(F_CHAN, np.ndarray ):
                F_CHAN = [F_CHAN]
                N_CHAN = [N_CHAN]

            # Add the values to the RMF
            n_skip = 0 
            for f_chan, n_chan in zip(F_CHAN,N_CHAN):

                if n_chan == 0:
                    continue

                RMF[f_chan:f_chan+n_chan,i] += RMF_line[n_skip:n_skip+n_chan]
                n_skip += n_chan

        # Make the RSP
        ARF = Table.read(ARF_path, 'SPECRESP')
        ARF_area = ARF['SPECRESP']

        # Extract the required matrix
        RSP = RMF * ARF_area
        RSP = RSP[min_channel:max_channel+1,min_input-1:max_input]

        # Find empty columns and lines
        empty_channels = np.all(RSP == 0, axis=1)
        empty_inputs = np.all(RSP == 0, axis=0)
        RSP = RSP[~empty_channels][:,~empty_inputs]
        channels = channels[ ~empty_channels ]
        inputs = inputs[ ~empty_inputs ]
        if empty_inputs.sum() > 0:
            print(f'Triming the response matrix because it contains lines with only 0 values.\n Now min_input={inputs[0]} and max_input={inputs[-1]}')
        if empty_channels.sum() > 0:
            print(f'Triming the response matrix because it contains columns with only 0 values.\n Now min_channel={channels[0]} and max_channel={channels[-1]}')

        # Get the edges of energies for both inputand channel
        energy_edges = np.append( ARF['ENERG_LO'][inputs-1], ARF['ENERG_HI'][inputs[-1]-1])
        channel_energy_edges = np.append(RMF_EBOUNDS['E_MIN'][channels],RMF_EBOUNDS['E_MAX'][channels[-1]])

        # Make the scaling
        alpha_name = 'energy_independent_effective_area_scaling_factor'
        alpha = Parameter(alpha_name,
                          strict_bounds = (0.1,1.9),
                          bounds = bounds.get(alpha_name, None),
                          doc=f'{RMF_instr} {alpha_name}',
                          symbol = r'$\alpha_{\rm INSTRUMENT}$'.replace('INSTRUMENT', RMF_instr),
                          value = values.get(alpha_name, 1.0 if bounds.get(alpha_name, None) is None else None))

        return cls(RSP,
                   energy_edges,
                   channels,
                   channel_energy_edges,
                   alpha, **kwargs)
    
