""" General Photosphere module for reading NSX.

From README :  """

import numpy as np

import xpsi

def load_NSX_table( path ):
    
    # Load tables and get sizes
    table = np.loadtxt(path, dtype=np.double)
    lenlogT = len( np.unique(table[:,3]) )
    lenlogg = len( np.unique(table[:,4]) )
    lenmu = len( np.unique(table[:,1]) )
    lenlogE = len( np.unique(table[:,0]) )

    # Make respective tables
    logT = np.zeros( lenlogT )
    logg = np.zeros( lenlogg )
    mu = np.zeros( lenmu )
    logE = np.zeros( lenlogE )

    reorder_buf = np.zeros((lenlogT,
                            lenlogg,
                            lenmu,
                            lenlogE,))

    index = 0
    for i in range(lenlogT):
        for j in range(lenlogg):
            for k in range(lenlogE):
                for l in range(lenmu):
                    logT[i] = table[index,3]
                    logg[j] = table[index,4]
                    logE[k] = table[index,0]
                    mu[reorder_buf.shape[2] - l - 1] = table[index,1]
                    reorder_buf[i,j,reorder_buf.shape[2] - l - 1,k] = 10.0**(table[index,2])
                    index += 1

    buf = np.zeros(np.prod(reorder_buf.shape))

    bufdex = 0
    for i in range(lenlogT):
            for j in range(lenlogg):
                for k in range(lenmu):
                    for l in range(lenlogE):
                        buf[bufdex] = reorder_buf[i,j,k,l]; bufdex += 1

    return logT, logg, mu, logE, buf


class CustomPhotosphere(xpsi.Photosphere):

    @xpsi.Photosphere.hot_atmosphere.setter
    def hot_atmosphere(self, path):
        
        # Read and set attributes
        logT, logg, mu, logE, buf = load_NSX_table( path )
        self._hot_atmosphere = (logT, logg, mu, logE, buf)

    @xpsi.Photosphere.elsewhere_atmosphere.setter
    def elsewhere_atmosphere(self, path):

        # Read and set attributes
        logT, logg, mu, logE, buf = load_NSX_table( path )
        self._elsewhere_atmosphere = (logT, logg, mu, logE, buf)

