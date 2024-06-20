from ._global_imports import *

from ._signalplot import SignalPlot

class ResidualPlot(SignalPlot):
    """ Plot the count data, the posterior-expected count signal, and residuals.

    The figure contains three panels which share phase as an x-axis:

        * the top panel displays the data count numbers in joint channel-phase
          intervals, identically split over two rotational phase cycles;
        * the center panel displays the posterior-expected count signal
          over joint channel-phase intervals;
        * the bottom panel displays the standardised residuals between the
          data and posterior-expected count signal over joint channel-phase
          intervals.

    The following example is (improved) from `Riley et al. 2019 <https://ui.adsabs.harvard.edu/abs/2019ApJ...887L..21R/abstract>`_:

    .. image:: _static/_residualplot.png

    :param str data_cmap:
         Colormap name from :mod:`matplotlib` to use for the data count numbers
         over joint channel-phase intervals.

    :param str model_cmap:
         Colormap name from :mod:`matplotlib` to use for the posterior-expected
         count numbers over joint channel-phase intervals.

    :param str residual_cmap:
         Colormap name from :mod:`matplotlib` to use for the residuals between
         the data and posterior-expected count numbers over joint
         channel-phase intervals. A diverging colormap is recommended.

    """

    __figtype__ = 'signalplot_residuals'

    # do not change at runtime (see base class comments):
    __caching_targets__ = ['expected_counts']

    __rows__          = 3
    __columns__       = 1
    __ax_rows__       = 3
    __ax_columns__    = 2
    __height_ratios__ = [1]*3
    __width_ratios__  = [50, 1] # second column for colorbars
    __wspace__        = 0.025
    __hspace__        = 0.125

    @make_verbose('Instantiating a residual plotter for posterior checking',
                  'Residual plotter instantiated')
    def __init__(self,
                 data_cmap='inferno',
                 model_cmap='inferno',
                 residual_cmap='PuOr',
                 theta_to_plot=None,
                 plot_pulse=False,
                 **kwargs):
        super(ResidualPlot, self).__init__(**kwargs)

        self._data_cmap = data_cmap
        self._model_cmap = model_cmap
        self._residual_cmap = residual_cmap
        self._plot_pulse = plot_pulse
        if not theta_to_plot is None:
            self.theta_to_plot = theta_to_plot

        self._data_row = 0
        self._model_row = 1 
        self._resid_row = 2

        # Changes some constants if required
        if self._plot_pulse:

            # Move all rows by one
            cls = type(self)
            cls.__rows__ = 4
            cls.__ax_rows__ = 4
            cls.__height_ratios__ = [1]*4
            self._pulse_row = 0
            self._data_row = 1
            self._model_row = 2 
            self._resid_row = 3

        # Get on plotting
        self._get_figure()

        self._ax_data = self._add_subplot(self._data_row,0)
        self._ax_data_cb = self._add_subplot(self._data_row,1)

        self._ax_model = self._add_subplot(self._model_row,0)
        self._ax_model_cb = self._add_subplot(self._model_row,1)

        self._ax_resid = self._add_subplot(self._resid_row,0)
        self._ax_resid_cb = self._add_subplot(self._resid_row,1)

        # Plot the pulse if required
        if self._plot_pulse:
            self._ax_pulse = self._add_subplot(self._pulse_row,0)
            self._ax_pulse.set_ylabel(r'Counts')

        self._ax_resid.set_xlabel('$\phi$ [cycles]')
        for ax in (self._ax_data, self._ax_model):
            ax.tick_params(axis='x', labelbottom=False)

        for ax in (self._ax_data, self._ax_model, self._ax_resid):
            ax.set_ylabel('channel')
            ax.xaxis.set_major_locator(MultipleLocator(0.2))
            ax.xaxis.set_minor_locator(MultipleLocator(0.05))
            ax.set_xlim([0.0,2.0])

        if "yscale" in kwargs:
            self.yscale = kwargs.get("yscale")
        else:
            self.yscale = "log"

        plt.close()

    @make_verbose('ResidualPlot object iterating over samples',
                  'ResidualPlot object finished iterating')
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
        self._add_data()
        self._add_expected_counts()
        self._add_residuals()
        if self._plot_pulse:
            self.add_pulses()

    def _set_vminmax(self):
        """ Compute minimum and maximum for data and model colorbars. """
        self._vmin = min(_np.min(self.expected_counts),
                         _np.min(self._signal.data.counts))
        self._vmax = max(_np.max(self.expected_counts),
                         _np.max(self._signal.data.counts))

    def add_pulses( self ):
        """ Display the pulse over the data if requested """

        # Get the pulse of data and model
        doubles_phases = _np.concatenate( (self._signal.data.phases[:-1] , (self._signal.data.phases[:-1]+1.0)), axis=0 ) 
        pulse_data = self._signal.data.counts.sum( axis = 0 )
        double_pulse_data = _np.concatenate( (pulse_data , pulse_data), axis=0 )
        pulse_model = self.expected_counts.sum( axis = 0 )
        double_pulse_model = _np.concatenate( (pulse_model , pulse_model) )

        # Plot pulse
        self._ax_pulse.errorbar(x=doubles_phases,
                                y=double_pulse_data,
                                yerr=_np.sqrt( double_pulse_data ),
                                ds='steps-mid',
                                label='Data', color='black')
        
        self._ax_pulse.errorbar(x=doubles_phases,
                                y=double_pulse_model, 
                                ds='steps-mid',
                                label='Model', color='steelblue')

    def _add_data(self):
        """ Display data in topmost panel. """

        try:
            self._vmin
        except AttributeError:
            self._set_vminmax()

        #Calculate channel edges by averaging:
        channels = self._signal.data.channels
        channel_edges = _np.zeros((len(self._signal.data.channels)+1))
        channel_edges[1:len(channels)] = (channels[:len(channels)-1]+channels[1:])/2.0
        chandiff1 = (channels[1]-channels[0])/2.0
        chandiff2 = (channels[len(channels)-1]-channels[len(channels)-2])/2.0
        channel_edges[0] = channels[0]-chandiff1
        channel_edges[len(channels)] = channels[len(channels)-1]+chandiff2

        data = self._ax_data.pcolormesh(self._signal.data.phases,
                                        channel_edges,
                                        self._signal.data.counts,
                                        cmap = cm.get_cmap(self._data_cmap),
                                        vmin = self._vmin,
                                        vmax = self._vmax,
                                        linewidth = 0,
                                        rasterized = self._rasterized)
        data.set_edgecolor('face')

        data = self._ax_data.pcolormesh(self._signal.data.phases + 1.0,
                                        channel_edges,
                                        self._signal.data.counts,
                                        cmap = cm.get_cmap(self._data_cmap),
                                        vmin = self._vmin,
                                        vmax = self._vmax,
                                        linewidth = 0,
                                        rasterized = self._rasterized)
        data.set_edgecolor('face')

        self._ax_data.set_ylim([channel_edges[0],
                                channel_edges[-1]])
        self._ax_data.set_yscale(self.yscale)

        self._data_cb = plt.colorbar(data, cax=self._ax_data_cb,
                                     ticks=_get_default_locator(None),
                                     format=_get_default_formatter())
        self._data_cb.ax.set_frame_on(True)
        self._data_cb.ax.yaxis.set_minor_locator(AutoMinorLocator())
        self._data_cb.set_label(label=r'counts', labelpad=15)

    def _add_expected_counts(self):
        """ Display posterior expectation of model in second panel. """

        try:
            self._vmin
        except AttributeError:
            self._set_vminmax()

        #Calculate channel edges by averaging:
        channels = self._signal.data.channels
        channel_edges = _np.zeros((len(self._signal.data.channels)+1))
        channel_edges[1:len(channels)] = (channels[:len(channels)-1]+channels[1:])/2.0
        chandiff1 = (channels[1]-channels[0])/2.0
        chandiff2 = (channels[len(channels)-1]-channels[len(channels)-2])/2.0
        channel_edges[0] = channels[0]-chandiff1
        channel_edges[len(channels)] = channels[len(channels)-1]+chandiff2

        model = self._ax_model.pcolormesh(self._signal.data.phases,
                                          channel_edges,
                                          self.expected_counts,
                                          cmap = cm.get_cmap(self._model_cmap),
                                          vmin = self._vmin,
                                          vmax = self._vmax,
                                          linewidth = 0,
                                          rasterized = self._rasterized)
        model.set_edgecolor('face')

        model = self._ax_model.pcolormesh(self._signal.data.phases + 1.0,
                                          channel_edges,
                                          self.expected_counts,
                                          cmap = cm.get_cmap(self._model_cmap),
                                          vmin = self._vmin,
                                          vmax = self._vmax,
                                          linewidth = 0,
                                          rasterized = self._rasterized)
        model.set_edgecolor('face')

        self._ax_model.set_ylim([channel_edges[0],
                                 channel_edges[-1]])
        self._ax_model.set_yscale(self.yscale)

        self._model_cb = plt.colorbar(model, cax=self._ax_model_cb,
                                      ticks=_get_default_locator(None),
                                      format=_get_default_formatter())
        self._model_cb.ax.set_frame_on(True)
        self._model_cb.ax.yaxis.set_minor_locator(AutoMinorLocator())

        self._model_cb.set_label(label=r'counts', labelpad=15)

    def _add_residuals(self):
        """ Display the residuals in the third panel. """

        self._residuals = self.expected_counts - self._signal.data.counts
        self._residuals /= _np.sqrt(self.expected_counts)

        vmax =  _np.max( _np.abs( self._residuals ) )

        #Calculate channel edges by averaging:
        channels = self._signal.data.channels
        channel_edges = _np.zeros((len(self._signal.data.channels)+1))
        channel_edges[1:len(channels)] = (channels[:len(channels)-1]+channels[1:])/2.0
        chandiff1 = (channels[1]-channels[0])/2.0
        chandiff2 = (channels[len(channels)-1]-channels[len(channels)-2])/2.0
        channel_edges[0] = channels[0]-chandiff1
        channel_edges[len(channels)] = channels[len(channels)-1]+chandiff2

        resid = self._ax_resid.pcolormesh(self._signal.data.phases,
                                      channel_edges,
                                      self._residuals,
                                      cmap = cm.get_cmap(self._residual_cmap),
                                      vmin = -vmax,
                                      vmax = vmax,
                                      linewidth = 0,
                                      rasterized = self._rasterized)
        resid.set_edgecolor('face')

        resid = self._ax_resid.pcolormesh(self._signal.data.phases + 1.0,
                                      channel_edges,
                                      _np.abs(self._residuals),
                                      cmap = cm.get_cmap(self._residual_cmap),
                                      vmin = -vmax,
                                      vmax = vmax,
                                      linewidth = 0,
                                      rasterized = self._rasterized)
        resid.set_edgecolor('face')

        self._ax_resid.axvline(1.0, lw=self._tick_width, color='k')

        self._ax_resid.set_ylim([channel_edges[0],
                                 channel_edges[-1]])
        self._ax_resid.set_yscale(self.yscale)

        self._resid_cb = plt.colorbar(resid, cax = self._ax_resid_cb,
                                      ticks=AutoLocator())
        self._resid_cb.ax.set_frame_on(True)
        self._resid_cb.ax.yaxis.set_minor_locator(AutoMinorLocator())

        self._resid_cb.set_label(label=r'$(c_{ik}-d_{ik})/\sqrt{c_{ik}}$',
                                 labelpad=15)

