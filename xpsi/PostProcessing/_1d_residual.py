from ._global_imports import *
from ._signalplot import SignalPlot

class Residual1DPlot(SignalPlot):
    """ Plot the count data, the posterior-expected count signal, and residuals.

    The figure contain one panel :

        * histogram of the standardised residuals between the
          data and posterior-expected count signal over joint channel-phase
          intervals.

    The following example is (improved) from `Riley et al. 2019 <https://ui.adsabs.harvard.edu/abs/2019ApJ...887L..21R/abstract>`_:

    .. image:: _static/_1d_residualplot.png

    :param ... Todo

    """

    __figtype__ = 'signalplot_1dresiduals'

    # do not change at runtime (see base class comments):
    __caching_targets__ = ['expected_counts']

    __rows__          = 1
    __columns__       = 1
    __ax_rows__       = 1
    __ax_columns__    = 1
    __height_ratios__ = [1]
    __width_ratios__  = [1] # second column for colorbars
    __wspace__        = 0.025
    __hspace__        = 0.125

    @make_verbose('Instantiating a 1D residual plotter for residual distribution checking',
                  'Residual plotter instantiated')
    def __init__(self,
                 nbins=20,
                 plot_estimated_distribution=True,
                 theta_to_plot=None,
                 **kwargs):
        super(Residual1DPlot, self).__init__(**kwargs)

        # Get the input parameters
        self._nbins = int( nbins )
        self._plot_estimated_distribution = plot_estimated_distribution
        if not theta_to_plot is None:
            self.theta_to_plot = theta_to_plot

         # Get on plotting
        self._get_figure()
        self._ax_1d_residuals = self._add_subplot(0,0)
        self._ax_1d_residuals.set_xlabel(r'$(c_{ik}-d_{ik})/\sqrt{c_{ik}}$')
        self._ax_1d_residuals.set_ylabel('Occurence')

        plt.close()

    @make_verbose('Residual1DPlot object iterating over samples',
                  'Residual1DPlot object finished iterating')
    def execute(self, thetas, wrapper):
        """ Loop over posterior samples. """
        self._num_samples = thetas.shape[0]

        wrapped = wrapper(self, 'model_sum')
        for i in range(self._num_samples):
            wrapped(None, thetas[i,:])

    def __next__(self):
        """ Update posterior expected model given the updated signal.

        .. note::

            You cannot make an iterator from an instance of this class.

        """
        try:
            self._model_sum
        except AttributeError:
            self._model_sum = self._signal.expected_counts.copy()
        else:
            self._model_sum += self._signal.expected_counts

    @property
    def model_sum(self):
        """ Get the current posterior sum of the count numbers. """
        return self._model_sum

    @model_sum.deleter
    def model_sum(self):
        del self._model_sum

    @property
    def expected_counts(self):
        """ Get the estimated posterior expectation of the count numbers. """
        return self._model_sum / self._num_samples

    @make_verbose('ResidualPlot object finalizing',
                  'ResidualPlot object finalized')
    def finalize(self):
        self._add_1d_residuals()

    def _add_1d_residuals(self):
        """ Display the residuals in the third panel. """

        # Get the residuals 
        self._residuals = self.expected_counts - self._signal.data.counts
        self._residuals /= _np.sqrt(self.expected_counts)
        self._residuals = self._residuals.flatten()

        # Plot them
        self._ax_1d_residuals.hist(self._residuals,bins=self._nbins)