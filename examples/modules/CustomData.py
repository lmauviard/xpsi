
import numpy as np
import matplotlib.pyplot as plt

import xpsi

from astropy.io import fits

class CustomData(xpsi.Data):

    @classmethod
    def XTI_from_fits(cls, XTI_path, n_phases=32, min_channel=20, max_channel=300):

        # Read the fits file
        with fits.open( XTI_path ) as hdul:
            Header = hdul[1].header
            XTI_data = hdul[1].data['EVENTS']

        # Check data
        try:
            assert (Header['TELESCOP'] == 'NICER') and (Header['INSTRUME'] == 'XTI')
        except AssertionError:
            raise IOError(f'Data from the wrong telescope or instrument. Check your {XTI_path} data')
        
        # Extract useful data
        exposure = Header['EXPOSURE']
        channel = XTI_data['PI']
        phase = XTI_data['PULSE_PHASE']

        # Make the 2D histogram
        mask = [ ch>=min_channel and ch<=max_channel for ch in channel ]
        loaded_data,_,phases = np.histogram2d( channel[mask] , phase[mask], 
                                            bins=[max_channel-min_channel+1, n_phases])
        
        # Get intrinsinc values
        channels = np.arange(min_channel,max_channel+1)
        first = 0
        last = max_channel - min_channel

        # Instatiate the class
        return cls( loaded_data,
                    channels=channels,
                    phases=phases,
                    first=first,
                    last=last,
                    exposure_time=exposure )
    
    @classmethod
    def EPIC_from_fits( cls, EPIC_path ):

        # Read the fits files
        with fits.open( EPIC_path ) as hdul:
            Header = hdul[1].header 
            EPIC_data = hdul[1].data['SPECTRUM']

        # Check the data
        assert Header['TELESCOP'] == 'XMM'
        
    def plot(self):

        # Get phase values (not only borders)
        phases = (self.phases[1:] + self.phases[:-1])/2

        # Do the plot
        plt.figure()
        plt.pcolormesh( phases, self.channels , self.counts)
        plt.xlabel('Phase bins')
        plt.ylabel('PI channel')
        plt.yscale('log')
        plt.ylim()