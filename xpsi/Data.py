
__all__ = ["Data"]

from xpsi.global_imports import *
from xpsi.utils import make_verbose

from xpsi.Instrument import ChannelError

class Data(object):
    """ A container for event data.

    The working assumption is that the sampling distribution of this event data
    can be written in terms of a set of channel-by-channel *count*\ -rate
    signals. The instrument associated with this data in an instance of
    :class:`~.Signal.Signal` must transform incident signals into a structure
    congruent to that of the event data. The attributes and methods of this
    class and any derived classes must therefore store information required for
    this operation.

    The initialiser assigns observation settings to attributes which are
    required for the treating the incident specific flux signals using the
    model instrument. The body of the initialiser may be changed, but to ensure
    inter-module compatibility, only the phase handling should really be
    modified, for instance if you want to implement an unbinned likelihood
    function with respect to phase; the phase bins defined in this concrete
    implementation are only used by a custom implementation of the
    likelihood function (i.e., by a subclass of :class:`xpsi.Signal`), and
    not by the other concrete classes used to construct the likelihood
    callable. The initialiser can also be extended if appropriate using a call
    to ``super().__init__``. Specialist constructors can be defined in a
    subclass using the ``@classmethod`` decorator, for instance to load event
    data from disk into a compatible data structure in memory; an example of
    this may be found below.

    .. note::

        You can subclass in order to tailor the handling of the event data, for
        instance to implement a likelihood functions for unbinned event data.

    :param ndarray[n,m] counts:
        A :class:`~numpy.ndarray` of count numbers. The columns must map to
        the phase intervals given by :obj:`phases`. The rows of the array map
        to some subset of instrument channels.

    :param ndarray[n] channels:
        Instrument channel numbers which must be equal in number to the first
        dimension of the :attr:`matrix`: the number of channels must be
        :math:`p`. These channels will correspond to the nominal response
        matrix and any deviation from this matrix (see above). In common usage
        patterns, the channel numbers will increase monotonically with row
        number, and usually increment by one (but this is not necessary). It is
        advisable that these numbers are the actual instrument channel numbers
        so that plots generated by the post-processing module using these
        labels are clear.

    :param ndarray[m+1] phases:
        A :class:`~numpy.ndarray` of phase interval edges, where events are
        binned into these same intervals in each instrument channel.

    :param int first:
        The index of the first row of the loaded response matrix containing
        events (see note below).

    :param int last:
        The index of the last row of the loaded response matrix containing
        events (see note below).

    .. note::

        The :obj:`counts` matrix rows  *must* span a contiguous subset of the
        rows of the loaded response matrix, but in general can span an
        arbitrary subset and order of instrument channels. Note that the
        :obj:`first` and :obj:`last+1` numbers are used to index the loaded
        instrument response matrix.  Therefore, if you load only a submatrix of
        the full instrument response matrix, these indices must be appropriate
        for the loaded submatrix, and must not be the true channel numbers
        (this information is instead loaded in the :class:`xpsi.Instrument`).
        Of course, in all sensible usage patterns the order of the instrument
        channels, when mapped to matrix rows, will be such that channel number
        increases with matrix row number monotonically because, then the
        nominal event energy increases monotonically with row number, which is
        important for visualisation of data and model (because spatial order
        matches energy order and correlations in space can be discerned easily).
        However, there could in principle be channel cuts that mean an increment
        of more than one channel between matrix adjacent rows, and the
        response matrix needs to be manipulated before or during a custom
        loading phase such that its rows match the channel numbers assigned to
        the :obj:`counts` matrix rows.

    :param float exposure_time:
        The exposure time, in seconds, to acquire this set of event data.

    """
    def __init__(self, counts, channels, phases, first, last, exposure_time):

        if not isinstance(counts, _np.ndarray):
            try:
                counts = _np.array(counts)
            except TypeError:
                raise TypeError('Counts object is not a ``numpy.ndarray``.')

        if counts.ndim not in [1,2]:
            raise TypeError('Counts must be in a one- or two-dimensional '
                            'array.')

        #if (counts < 0.0).any():
        #    raise ValueError('Negative count numbers are invalid.')

        self.channels = channels

        if not isinstance(phases, _np.ndarray):
            try:
                phases = _np.array(phases)
            except TypeError:
                raise TypeError('Phases object is not a ``numpy.ndarray``.')

        if phases.ndim != 1:
            raise ValueError('Phases must form a one-dimensional sequence.')

        self._phases = phases

        try:
            self._first = int(first)
            self._last = int(last)
        except TypeError:
            raise TypeError('The first and last channels must be integers.')

        #if self._first >= self._last:
        if self._first > self._last:        
            raise ValueError('The first channel number must be equal or lower '
                             'than the last channel number.')

        if counts.shape[0] != self._last - self._first + 1:
            raise ValueError('The number of rows must be compatible '
                             'with the first and last channel numbers.')

        self._counts = counts

        try:
            self._exposure_time = float(exposure_time)
        except TypeError:
            raise TypeError('Exposure time must be a float.')

    @property
    def exposure_time(self):
        """ Get the total exposure time in seconds. """
        return self._exposure_time

    @property
    def counts(self):
        """ Get the photon count data. """
        return self._counts

    @property
    def phases(self):
        """ Get the phases. """
        return self._phases

    @property
    def index_range(self):
        """ Get a 2-tuple of the bounding response-matrix row indices. """
        return (self._first, self._last + 1) # plus one for array indexing

    @property
    def channel_range(self):
        """ Deprecated property name. To be removed for v1.0. """
        return self.index_range

    @property
    def channels(self):
        """ Get the array of channels that the event data spans. """
        return self._channels

    @channels.setter
    @make_verbose('Setting channels for event data',
                  'Channels set')
    def channels(self, array):
        if not isinstance(array, _np.ndarray):
            try:
                array = _np.array(array)
            except TypeError:
                raise ChannelError('Channel numbers must be in a '
                                   'one-dimensional array, and must all be '
                                   'positive integers including zero.')

        if array.ndim != 1 or (array < 0).any():
            raise ChannelError('Channel numbers must be in a '
                               'one-dimensional array, and must all be '
                               'positive integers including zero.')

        try:
            if array.shape[0] != array.shape[0]:
                raise ChannelError('Number of channels does')
        except AttributeError:
            pass # if synthesising there will not be any counts loaded here

        if (array[1:] - array[:-1] != 1).any():
            yield ('Warning: Channel numbers do not uniformly increment by one.'
                   '\n         Please check for correctness.')

        self._channels = array

        yield

    @classmethod
    @make_verbose('Loading event list and phase binning',
                  'Events loaded and binned')
    def phase_bin__event_list(cls, path, channels, phases,
                              channel_column,
                              phase_column=None,
                              phase_averaged=False,
                              phase_shift=0.0,
                              channel_edges=None,
                              skiprows=1,
                              eV=False,
                              dtype=_np.int32,
                              *args, **kwargs):
        """ Load a phase-folded event list and bin the events in phase.

        :param str path:
            Path to event list file containing two columns, where the first
            column contains phases on the unit interval, and the second
            column contains the channel number.

        :param ndarray[n] channels:
            An (ordered) subset of instrument channels. It is advisable that
            these channels are a contiguous subset of instrument channels, but
            this not a strict requirement if you are comfortable with the
            handling the instrument response matrix and count number matrix to
            match in row-to-channel definitions.

        :param list phases:
            An ordered sequence of phase-interval edges on the unit interval.
            The first and last elements will almost always be zero and unity
            respectively.

        :param float phase_shift:
            A phase-shift in cycles to be applied when binning the events in
            phase.

        :param int phase_column:
            The column in the loaded file containing event phases.

        :param bool phase_averaged:
            Is the event data phase averaged?

        :param int channel_column:
            The column in the loaded file containing event channels.

        :param ndarray[n+1] channel_edges:
            The nominal energy edges of the instrument channels, assumed to
            be contiguous if binning event energies in channel number.

        :param int skiprows:
            The number of top rows to skip when loading the events from file.
            The top row of couple of rows will typically be reserved for
            column headers.

        :param bool eV:
            Are event energies in eV, instead of channel number?

        :param type dtype:
            The type of the count data. Sensible options are ``numpy.int`` (the
            default) or a :mod:`numpy` floating point type. The choice here
            only matters when executing custom likelihood evaluation code, which
            might expect a type without checking or casting.

        """
        events = _np.loadtxt(path, skiprows=skiprows)

        channels = list(channels)

        yield 'Total number of events: %i.' % events.shape[0]

        data = _np.zeros((len(channels), len(phases)-1), dtype=dtype)

        for i in range(events.shape[0]):
            _channel = None
            if eV:
                for j in range(len(channel_edges) - 1):
                    if channel_edges[j] <= events[i, channel_column]/1.0e3 < channel_edges[j+1]:
                        _channel = channels[j]
                        break
            else:
                _channel = events[i, channel_column]

            if _channel is not None and _channel in channels:
                if not phase_averaged:
                    _temp = events[i,phase_column] + phase_shift
                    _temp -= _np.floor(_temp)
                    for j in range(phases.shape[0] - 1):
                        if phases[j] <= _temp <= phases[j+1]:
                            data[channels.index(int(events[i,channel_column])),j] += 1
                            break
                else:
                    data[channels.index(int(_channel)), 0] += 1

        yield 'Number of events constituting data set: %i.' % _np.sum(data)

        yield cls(data, channels, _np.array(phases), *args, **kwargs)

    @classmethod
    def bin__event_list(cls, *args, **kwargs):
        return cls.phase_bin__event_list(*args, **kwargs)
