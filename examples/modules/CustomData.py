import os
import numpy as np
import matplotlib.pyplot as plt

import xpsi

from astropy.io import fits

class CustomData(xpsi.Data):

    @classmethod
    def load(cls, path,
             n_phases=32, 
             channels=None, 
             phase_column='PULSE_PHASE',
             channel_column='PI',
             datafolder = None):
        
        # Add the path if required
        if datafolder:
            path = os.path.join( datafolder, path )
        
        # Check whether event file or PHA
        with fits.open( path ) as hdul:
            HDUCLAS1 = hdul[1].header['HDUCLAS1']

        # Select the case based on HDUCLAS1
        if HDUCLAS1 == 'SPECTRUM':
            return cls.from_pha(path, channels=channels )
        
        elif HDUCLAS1 == 'EVENTS':
            return cls.from_evt(path, 
                 n_phases=n_phases, 
                 channels=channels, 
                 phase_column=phase_column,
                 channel_column=channel_column)
        
        else:
            raise IOError('HDUCLAS1 of Header does not match PHA or EVT files values. Could not load.')
        
    @classmethod
    def from_evt(cls, path, 
                 n_phases=32, 
                 channels=None, 
                 phase_column='PULSE_PHASE',
                 channel_column='PI'):

        # Read the fits file
        with fits.open( path ) as hdul:
            Header = hdul['EVENTS'].header
            EvtList = hdul['EVENTS'].data

        # Extract useful data
        exposure = Header['EXPOSURE']
        channel_data = EvtList[channel_column]
        phases_data = EvtList[ phase_column ]

        # No channels specified, use everything
        if channels is None:
            min_channel = np.min( channel_data )
            max_channel = np.max( channel_data )
            channels = np.arange(min_channel,max_channel+1)
        else:
            min_channel = channels[0]
            max_channel = channels[-1]
            
         # Get intrinsinc values
        first = 0
        last = max_channel - min_channel
        phases_borders = np.linspace( 0.0 , 1.0 , n_phases+1 )

        # Make the 2D histogram
        mask = [ ch>=min_channel and ch<=max_channel for ch in channel_data ]
        counts_histogram, _, _ = np.histogram2d( channel_data[mask] , phases_data[mask], 
                                                 bins=[last+1, phases_borders])

        # Instatiate the class
        return cls( counts_histogram.astype( dtype=np.double ),
                    channels=channels,
                    phases=phases_borders,
                    first=first,
                    last=last,
                    exposure_time=exposure )
    

    @classmethod
    def from_pha( cls, path, 
                  channels=None ):

        # Read the fits files
        with fits.open( path ) as hdul:
            Header = hdul['SPECTRUM'].header 
            spectrum = hdul['SPECTRUM'].data

        # Extract useful data
        exposure = np.double( Header['EXPOSURE'] )
        channel_data = spectrum['CHANNEL']
        counts_data = spectrum['COUNTS']

        # No channels specified, use everything
        if channels is None:
            min_channel = np.min( channel_data )
            max_channel = np.max( channel_data )
            channels = np.arange(min_channel,max_channel+1)
        else:
            min_channel = channels[0]
            max_channel = channels[-1]
            
         # Get intrinsinc values
        first = 0
        last = max_channel - min_channel
        phases = np.array([0.0, 1.0])

        # Match channels
        channel_counts_map = dict(zip(channel_data, counts_data))
        if not all(ch in channel_counts_map for ch in channels):
            raise ValueError("Not all channels exist in channel_data.")
        counts = np.array( [[float(channel_counts_map[ch]) for ch in channels]] , dtype=np.double).T
        
        Data = cls( counts,
                    channels=channels,
                    phases=phases,
                    first=first,
                    last=last,
                    exposure_time=exposure)
        
        # Add useful paths
        Data.ancrfile = Header['ANCRFILE']
        Data.respfile = Header['RESPFILE']
        if Header['HDUCLAS2'] == 'TOTAL':
            Data.backfile = Header['BACKFILE']

        return Data
         

    def spectra_support(self, n, smoothing=True):

        # Get the background spectrum
        spectrum = self.counts.sum(axis=1)
        support = np.array([spectrum-n*np.sqrt(spectrum),spectrum+n*np.sqrt(spectrum)]).T
        support[support[:,0] < 0.0, 0] = 0.0

        # Apply support smoothing if one upper value is not defined
        for i in range(support.shape[0]):
            if support[i,1] == 0.0 and smoothing:
                for j in range(1, support.shape[0]):
                    if i+j < support.shape[0] and support[i+j,1] > 0.0:
                        support[i,1] = support[i+j,1]
                        break
                    elif i-j >= 0 and support[i-j,1] > 0.0:
                        support[i,1] = support[i-j,1]
                        break
        
        # Clean
        support = np.ascontiguousarray( support, dtype=np.double )
        return support

    def plot(self, num_rot = 2):

        # Get the counts
        counts_list = [ self.counts for i in range(num_rot) ]
        phase_list = [self.phases[:-1] + i for i in range(num_rot)] 
        counts = np.concatenate( (counts_list), axis=1 )
        phases = np.concatenate( (phase_list), axis=0 )

        # Do the plot
        mosaic = [['A','.'],['B','C']]
        fig,axs = plt.subplot_mosaic( mosaic , height_ratios=[1.,1.], width_ratios=[3,1])

        ax1 = axs['A']
        ax1.step( phases, counts.sum(axis=0) , color='black')
        ax1.set_ylabel('Counts')
        ax1.vlines( 1. , ymin=np.min(counts.sum(axis=0)), ymax =np.max(counts.sum(axis=0)), ls='--', color='blue' )

        ax2 = axs['B']
        ax2.pcolormesh( phases, self.channels , counts)
        ax2.sharex( ax1 )
        ax2.set_xlabel('Phase bins')
        ax2.set_ylabel('PI channel')
        ax2.set_yscale('log')

        ax3 = axs['C']
        ax3.sharey( ax2 )
        ax3.get_yaxis().set_visible(False)
        ax3.step( counts.sum(axis=1), self.channels , color='black')
        ax3.set_yscale('log')
        ax3.set_xlabel('Counts per channel')

    def plot_spectra(self, num_rot=2):

         # Get phase values (not only borders)
        phases = (self.phases[1:] + self.phases[:-1])/2

        # Get the counts
        counts_list = [ self.counts for i in range(num_rot) ]
        phase_list = [phases + i for i in range(num_rot)] 
        counts = np.concatenate( (counts_list), axis=1 )
        phases = np.concatenate( (phase_list), axis=0 )

        # 
        plt.figure()
        plt.step( phases, counts.sum(axis=0) )
        plt.xlabel('Phase bins')
        plt.ylabel('Counts')
        plt.ylim()
