from xpsi import _verbose
from xpsi.Likelihood import Likelihood
from xpsi.Prior import Prior

import numpy as np

try:
    from nessai.flowsampler import FlowSampler
    from nessai.livepoint import live_points_to_array
    from nessai.model import Model as NessaiModel

except ImportError:
    print('Check your nessai installation. (pip install nessai)')
    raise
else:
    if _verbose:
        print('Imported Nessai.')


class _NessaiModel(NessaiModel):
    """ Sample the unit hypercube using the X-PSI inverse prior transform. """

    allow_vectorised = False
    # The wrapper disables this when a constant rejection floor is configured.
    allow_multi_valued_likelihood = True

    def __init__(self, param_names, prior, likelihood):
        self.names = ['u_' + name for name in param_names]
        self.bounds = {name: (0.0, 1.0) for name in self.names}
        self._prior = prior
        self._likelihood = likelihood

    def log_prior(self, x):
        """ Uniform density on the unit hypercube. """
        return np.where(self.in_bounds(x), 0.0, -np.inf)

    def to_unit_hypercube(self, x):
        """ i-nessai's model coordinates are already in the unit hypercube. """
        return x.copy()

    def from_unit_hypercube(self, x):
        """ Keep cube coordinates; the physical transform is applied in log_likelihood. """
        return x.copy()

    def log_likelihood(self, x):
        """ Transform one live point and evaluate the X-PSI likelihood. """
        hypercube = [x[name].item() for name in self.names]
        params = self._prior.inverse_sample(hypercube)
        # X-PSI also evaluates any prior support/density defined in Prior.__call__.
        return self._likelihood(params)


class NessaiSampler:
    """ Wrapper for the Nessai (https://nessai.readthedocs.io/en/latest/) package.

    The importance nested sampling variant (i-nessai) is used by default.

    :param likelihood: An instance of :class:`~.Likelihood.Likelihood`.

    :param prior: An instance of :class:`~.Prior.Prior`.

    :param sampler_build_kwargs: A dictionary of keyword arguments passed to
        :class:`nessai.flowsampler.FlowSampler`. Set
        ``importance_nested_sampler=False`` to use standard Nessai.

    :param log_likelihood_floor: Optional finite log-likelihood floor. Values
        below it are clamped to this constant, including X-PSI's random rejection
        values. Choose a value at least ``0.1 * likelihood.llzero`` and well
        below the likelihoods that contribute to the posterior. This changes
        the likelihood below the floor; it is not a universal default.

    Sampling is serial and uses :meth:`~.Prior.Prior.inverse_sample`, as in
    the Nautilus wrapper. Native Nessai results contain unit-hypercube
    coordinates named ``u_<parameter name>``. Use :attr:`posterior_samples`
    to obtain physical parameter values. No X-PSI run files are exported.
    Without a floor, X-PSI's finite, random rejection values are preserved.
    With a floor, i-nessai defaults to strict thresholds and quantile selection.
    Startup checks prevent a threshold on the constant plateau from reporting
    premature convergence. Start a new output directory when changing the floor
    or sampler settings; Nessai checkpoints retain their previous settings.

    """

    def __init__(
        self,
        likelihood : Likelihood,
        prior : Prior,
        sampler_build_kwargs=None,
        log_likelihood_floor=None,
    ):

        if not isinstance(likelihood, Likelihood):
            raise TypeError('Invalid type for likelihood object.')
        self._likelihood = likelihood

        if not isinstance(prior, Prior):
            raise TypeError('Invalid type for prior object.')
        self._prior = prior

        if log_likelihood_floor is not None:
            log_likelihood_floor = float(log_likelihood_floor)
            if not np.isfinite(log_likelihood_floor):
                raise ValueError('log_likelihood_floor must be finite.')
            if log_likelihood_floor < 0.1 * likelihood.llzero:
                raise ValueError('log_likelihood_floor must be at least '
                                 '0.1 * likelihood.llzero to remove rejection noise.')
        self.log_likelihood_floor = log_likelihood_floor

        self._param_names = self._likelihood.names

        self.model = _NessaiModel(
            self._param_names, self._prior, self.likelihood_wrapper
        )
        sampler_build_kwargs = dict(sampler_build_kwargs or {})
        sampler_build_kwargs.setdefault('importance_nested_sampler', True)
        if log_likelihood_floor is not None:
            self.model.allow_multi_valued_likelihood = False
            if sampler_build_kwargs['importance_nested_sampler']:
                sampler_build_kwargs.setdefault('strict_threshold', True)
                sampler_build_kwargs.setdefault('threshold_method', 'quantile')
        self._sampler = FlowSampler(self.model, **sampler_build_kwargs)

    def __call__(self, runtime_params):
        """ Run the sampler until target convergence criteria are fulfilled with the
        given runtime parameters.

        :param runtime_params: A dictionary of the keyword arguments passed to :func:`run`.

        """

        if (self.log_likelihood_floor is not None
                and self._sampler.importance_nested_sampler):
            self._check_floor_startup()
        self._sampler.run(**runtime_params)

    def _check_floor_startup(self):
        """ Refuse a floor plateau before it can look like convergence. """
        ns = self._sampler.ns
        if not ns.strict_threshold:
            raise ValueError('A clamped i-nessai run requires strict_threshold=True. '
                             'Use a new output directory if resuming old settings.')
        if ns.finalised:
            return
        if ns.live_points_unit is None:
            # Retain every initial draw for evidence accounting. The normal run
            # will initialise the proposal and reuse these evaluated points.
            ns.populate_live_points()
        n_above = np.count_nonzero(
            ns.training_samples.samples['logL'] > self.log_likelihood_floor
        )
        if n_above < ns.min_samples:
            raise RuntimeError(
                f'Only {n_above} initial training points lie above the likelihood '
                f'floor, but min_samples={ns.min_samples}. Increase n_initial '
                'or lower min_samples before starting a fresh run.'
            )
        if ns.n_update is None:
            threshold = ns.determine_log_likelihood_threshold(
                ns.live_points_unit, method=ns.threshold_method, **ns.threshold_kwargs
            )
        else:
            threshold = ns.live_points_unit[ns.n_update]['logL']
        if threshold <= self.log_likelihood_floor:
            raise RuntimeError(
                'The initial i-nessai threshold lies on the constant likelihood '
                'floor. Increase threshold_kwargs["q"] with '
                'threshold_method="quantile", and ensure enough initial points '
                'lie above the floor. Sampling stopped before false convergence.'
            )

    def likelihood_wrapper(self, params):
        """Calculate the loglikelihood value for a given set of parameter values.

        :param params: List of parameter values.

        :returns: Float with loglikelihood value for given set of parameter values.

        """

        logl = self._likelihood(p=list(params), reinitialise=True)
        logl = float(np.asarray(logl).item())
        if self.log_likelihood_floor is not None:
            logl = max(logl, self.log_likelihood_floor)
        return logl

    @property
    def posterior_samples(self):
        """ Equal-weight physical samples, with columns in ``likelihood.names`` order. """
        hypercube = live_points_to_array(
            self._sampler.posterior_samples, names=self.model.names
        )
        return np.asarray([
            self._prior.inverse_sample(point) for point in hypercube
        ]).reshape(-1, len(self._param_names))
