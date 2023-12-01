"""
This module brings together the main components of the TurbSim program
and defines the primary high-level objects that users of the Python
interface will utilize.

This module, however, contains more functions and objects than the
typical user will want access to. For a more compact version of the
PyTurbSim interface import the ./api.py package.

"""
from .base import ts_complex, gridProps, dbg, np, statObj
from .profModels.base import profModelBase, profObj
from .specModels.base import specModelBase, specObj
from .cohereModels.base import cohereModelBase, cohereObj, cohereUser
from .stressModels.base import stressModelBase, stressObj
from .phaseModels.api import randPhase
import _version as ver
from .io import write
from numpy import random
from numpy import ulonglong
from numpy.fft import irfft
import time

# !!!VERSION_INCONSISTENCY
# inconsistency between this and older versions of TurbSim
# !!!CHECKTHIS
# means I need to ensure that something is right.
# !!!FIXTHIS
# means I know I am doing something wrong.
# !!!ADDDOC
# Means add documentation here

# TODO:
#  - Testing:
#     . Test 'user-defined' models
#  - Documentation
#     . Document plotting tools.
################
## These require branches
#  - Fix Reynold's stress!
#  - Break Cholesky from coherence models/objects and into 'main', or tslib.
#     . How do callbacks work fortran->python?
#     . Can we implement this so that 'models/objects' are separated from tslib?
#  - Add 'mods':
#      . Add ability to rotate mean velocity field (for a prof instance and a profModel).
#      . Add ability to add veer to mean velocity field (prof instance and profModel).
#  - Write .sum summary files (io package), (so they are fully self-contained).
#     . Add parameter logging, so that we can write summary files that
#       track all parameters that were input.
################
## Low priority
#  - Write FF files (tsio.py).
#  - Write HubHeight files (tsio.py).
#  - Add KHtest functionality? (rgrep for '#KHTEST')
#  - Write 'events' (includes adding 'coherent events' to TS)


class tsrun(object):
    """
    This is the PyTurbSim 'run' class. This class provides the
    interface for controlling PyTurbSim simulations and output.

    Examples of how to use this class, and the PyTurbSim interface in
    general can be found in the PyTurbSim /examples directory.

    Parameters
    ----------

    RandSeed : int,optional ('random value')
               Initialize the run-object with a RandSeed.
    ncore : int,optional (1)
            Number of cores (processors) to use for the pyTurbSim run

    """
    def __init__(self, RandSeed=None, ncore=1):
        """
        PyTurbSim 'run' objects can be initialized with a specific
        random seed, `RandSeed`, and number of cores, `ncore`.
        """
        # Initialize the random number generator before doing anything else.
        if RandSeed is None:
            self.RandSeed = random.randint(-2147483647, 2147483647)
        else:
            self.RandSeed = RandSeed
        self.ews_update=False
        # Seeds for numpy must be positive, but original-TurbSim had
        # negative seeds.  In order to attempt to be consistent, we
        # use the values in the files but make them positive for the
        # numpy random generator.
        self.randgen = random.RandomState(
            ulonglong(self.RandSeed + 2147483648))
        self.ncore = ncore
        if dbg:
            self.timer = dbg.timer('Veers84')
    # For now this is a place-holder, I may want to make this an
    # 'input property' eventually.
    phase = randPhase()

    @property
    def prof(self):
        """
        This is the 'mean velocity profile' input property.

        This property returns a 'profObj'.

        This property can be defined with three types of objects:

        1) define it with a 'profile model' (recommended)::

                  ts_run.prof=a_prof_model

           In this case the model is set to my_ts_run.profModel, and
           this model is called to produce a profObj AS NEEDED.  At
           the end of the ts_run call that profObj is cleared so
           that subsequent runs do not use a fixed profObj (i.e. in
           the case that the model is modified or another
           model/object that the profile model depends on is
           changed between runs).

        2) define it with a profObj directly (profile
           statistic-object)::

              ts_run.prof=a_prof_model(ts_run)

           In this case the profObj is FIXED. That is, all
           subsequent PyTurbSim runs will utilize this profile,
           which is based on the state of the a_prof_model and
           ts_run at the time of the profObj creation.

        3) define it with an array directly::

               ts_run.prof=a_numpy_array   [units: m/s]

           In this case the profObj is again fixed and defined by
           the input array.  The numpy array dimensions must match
           those of the gridObj.  That is, the dimensions of the
           array should be (3 x grid.n_z x grid.n_y).  The first
           dimension is for each component of the profile (u,v,w),
           the next two are for each point (z,y) in the grid.

        See Also
        --------
        pyts.profModels.api : to see available profile models.
        tsrun.spec
        tsrun.cohere
        tsrun.stress

        """
        if hasattr(self, 'profModel') and not hasattr(self, '_prof'):
            self._prof = self.profModel(self)
        return self._prof

    @prof.setter
    def prof(self, val):
        if profModelBase in val.__class__.__mro__:
            self.profModel = val
        elif np.ndarray in val.__class__.__mro__:
            self._prof = profObj(self)
            self._prof.array[:] = val
        elif profObj in val.__class__.__mro__:
            self._prof = val
        else:
            raise Exception('The input must be a profile model, '
                            'profile object or numpy array; it is none of these.')

    @prof.deleter
    def prof(self,):
        if hasattr(self, 'profModel'):
            del self._prof

    @property
    def spec(self):
        """
        This is the 'tke spectrum' input property.

        This property always returns a `.specObj`.

        This property can be defined with three types of objects:

        1) define it with a 'spectral model' (recommended)::

                  ts_run.spec=a_spec_model

           In this case the model is set to my_ts_run.specModel, and
           this model is called to produce a specObj AS NEEDED.  At
           the end of the ts_run call that specObj is cleared so
           that subsequent runs do not use a fixed specObj (i.e. in
           the case that another model/object that the spectral model
           depends on is changed between runs).

        2) define it with a specObj directly::

              ts_run.spec=a_spec_model(ts_run)

           In this case the specObj is FIXED. That is, all
           subsequent PyTurbSim runs will utilize this spectral
           model, which is based on the state of ts_run at the time
           of the specObj creation.

        3) define it with an array directly::

               ts_run.spec=a_numpy_array  - [units: m^2/(s^2.Hz)]

           In this case the specObj is again fixed and defined by
           the input array.  The numpy array dimensions must match
           those of the gridObj.  That is, the dimensions of the
           array should be (3 x grid.n_z x grid.n_y x grid.n_f).
           The first dimension is for each component of the spectrum
           (u,v,w), the next two are for each point (z,y) in the
           grid, and the last dimension is the frequency dependence
           of the spectrum.

        See Also
        --------
        pyts.specModels.api : to see available spectral models.
        tsrun.prof
        tsrun.cohere
        tsrun.stress

        """
        if hasattr(self, 'specModel') and not hasattr(self, '_spec'):
            self._spec = self.specModel(self)
        return self._spec

    @spec.setter
    def spec(self, val):
        if specModelBase in val.__class__.__mro__:
            self.specModel = val
        elif np.ndarray in val.__class__.__mro__:
            self._spec = specObj(self)
            self._spec.array[:] = val
        elif specObj in val.__class__.__mro__:
            self._spec = val
        else:
            raise Exception('The input must be a spectral model, '
                            'spectra object or numpy array; it is none of these.')

    @spec.deleter
    def spec(self,):
        if hasattr(self, 'specModel'):
            del self._spec

    @property
    def cohere(self):
        """
        This is the 'coherence' input property.

        This property always returns a :class:`~.cohereModels.base.cohereObj`.

        Because the bulk of PyTurbSim's computational requirements
        (memory and processor time) are consumed by dealing with this
        statistic, it behaves somewhat differently from the others. In
        particular, rather than relying on arrays for holding data
        'coherence objects' define functions that are called as
        needed.  This dramatically reduces the memory requirements of
        PyTurbSim without increasing.  See the cohereModels package
        documentation for further details.  Fortunately, at this
        level, coherence is specified identically to other
        statistics...

        This property can be defined with three types of objects:

        1) define it with a 'coherence model' (recommended)::

                  ts_run.cohere=a_coherence_model

           In this case the model is set to my_ts_run.cohereModel,
           and this model sets the is called at runtime to produce the phase
           array. At the end of the ts_run call that phase array is
           cleared so that subsequent runs do not use a fixed
           phase information (i.e. in the case that the coherence
           model is modified or another model/object that the
           coherence model depends on is changed between runs).

        2) define it with a cohereObj directly ::

              ts_run.spec=a_coherence_model(ts_run)

           In this case the cohereObj is FIXED. That is, all
           subsequent PyTurbSim runs will utilize this coherence
           model, which is based on the state of ts_run at the time
           of execution of this command.

        3) define it with an array directly::

               ts_run.cohere=a_numpy_array  - [units: non-dimensional]

           In this case the coherence will be fixed and defined by
           this input array.  Th e numpy array dimensions must match
           those of the gridObj.  That is, the dimensions of the
           array should be (3 x grid.n_p x grid.n_p xgrid.n_f).
           The first dimension is for each component of the spectrum
           (u,v,w), the next two are for each point-pair (z,y) in the
           grid, and the last dimension is the frequency dependence
           of the spectrum.

           This approach for specifying the coherence - while
           explicit and flexible - requires considerably more memory
           than the 'coherence model' approach.  Furthermore using
           this approach one must be careful to make sure that the
           ordering of the array agrees with that of the 'flattened
           grid' (see the gridObj.flatten method, and/or the
           cohereUser coherence model for more information).

        See Also
        --------
        pyts.cohereModels.api : to see a list of available coherence models.
        pyts.cohereModels.base.cohereUser : the 'user-defined' or 'array-input' coherence model.
        tsrun.prof
        tsrun.spec
        tsrun.stress

        """
        if hasattr(self, 'cohereModel') and not hasattr(self, '_cohere'):
            self._cohere = self.cohereModel(self)
        return self._cohere

    @cohere.setter
    def cohere(self, val):
        if cohereModelBase in val.__class__.__mro__:
            self.cohereModel = val
        elif np.ndarray in val.__class__.__mro__:
            self.cohereModel = cohereUser(val)
        elif cohereObj in val.__class__.__mro__:
            self.cohere = val
        else:
            raise Exception('The input must be a coherence model, '
                            'coherence object or numpy array; it is none of these.')

    @cohere.deleter
    def cohere(self,):
        if hasattr(self, 'cohereModel'):
            del self._cohere

    @property
    def stress(self):
        """
        This is the Reynold's stress input property.

        This property always returns a :class:`.stressObj`.

        This property can be defined with three types of objects:

        1) define it with a `specModel` (recommended)::

              ts_run.stress=a_stress_model

           In this case the model is set to my_ts_run.stressModel, and
           this model is called to produce a stressObj AS NEEDED.  At
           the end of the ts_run call that stressObj is cleared so
           that subsequent runs do not use a fixed stressObj (i.e. in
           the case that another model/object that the stress model
           depends on is changed between runs).

        2) define it with a `stressObj` directly::

              ts_run.stress=a_stress_model(ts_run)

           In this case the stressObj is FIXED. That is, all
           subsequent PyTurbSim runs will utilize this stress
           model, which is based on the state of ts_run at the time
           of the stressObj creation.

        3) define it with an array directly::

              ts_run.stress=a_numpy_array  - [units: m^2/s^2]

           In this case the stressObj is again fixed and defined by
           the input array.  The numpy array dimensions must match
           those of the gridObj.  That is, the dimensions of the
           array should be (3 x grid.n_z x grid.n_y).
           The first dimension is for each component of the stress
           (u,v,w), the next two are for each point (z,y) in the
           grid.

        See Also
        --------
        pyts.stressModels.api : To see available stress models.

        tsrun.prof
        tsrun.spec
        tsrun.cohere

        """
        if hasattr(self, 'stressModel') and not hasattr(self, '_stress'):
            self._stress = self.stressModel(self)
        return self._stress

    @stress.setter
    def stress(self, val):
        if stressModelBase in val.__class__.__mro__:
            self.stressModel = val
        elif np.ndarray in val.__class__.__mro__:
            self._stress = stressObj(self)
            self._stress.array[:] = val
        elif stressObj in val.__class__.__mro__:
            self._stress = val
        else:
            raise Exception('The input must be a stress model, '
                            'stress object or numpy array; it is none of these.')

    @stress.deleter
    def stress(self,):
        if hasattr(self, 'stressModel'):
            del self._stress

    def reset(self, seed=None):
        """
        Clear the input statistics and reset the Random Number
        generator to its initial state.
        """
        del self.prof
        del self.spec
        del self.cohere
        del self.stress
        if seed is None:
            self.randgen.seed(self.RandSeed)
        else:
            self.randgen.seed(seed)

    @property
    def info(self,):
        """
        Model names and initialization parameters.
        """
        out = dict()
        out['version'] = (ver.__prog_name__, ver.__version__, ver.__version_date__)
        out['RandSeed'] = self.RandSeed
        out['StartTime'] = self._starttime
        if hasattr(self, '_config'):
            out['config'] = self._config
        for nm in ['profModel', 'specModel', 'cohereModel', 'stressModel']:
            if hasattr(self, nm):
                mdl = getattr(self, nm)
                out[nm] = dict(name=mdl.model_name,
                               description=mdl.model_desc,
                               params=mdl.parameters,
                               sumstring=mdl._sumfile_string(self),
                               )
            else:
                out[nm] = None
        out['RandSeed'] = self.RandSeed
        out['RunTime'] = time.time() - time.mktime(self._starttime)
        return out

    def run(self,):
        """
        Run PyTurbSim.

        Before calling this method be sure to set the following
        attributes to their desired values:

        - :attr:`tsrun.prof`: The mean profile model, object or array.
        - :attr:`tsrun.spec`: The tke spectrum model, object or array.
        - :attr:`tsrun.cohere`: The coherence model, object or array.
        - :attr:`tsrun.stress`: The Reynold's stress model, object or array.

        Returns
        -------
        tsdata : :class:`tsdata`

        """
        self._starttime = time.localtime()
        self.timeseries = self._calcTimeSeries()
        out = self._build_outdata()
        return out

    __call__ = run

    def _build_outdata(self,):
        """
        Construct the output data object and return it.
        """
        out = tsdata(self.grid)
        out.uturb = self.timeseries
        out.uprof = self.prof.array
        out.info = self.info
        return out

    def _calcTimeSeries(self,):
        """
        Compute the u,v,w, timeseries based on the spectral, coherence
        and Reynold's stress models.

        This method performs the work of taking a specified spectrum
        and coherence function and transforming it into a spatial
        timeseries.  It performs the steps outlined in Veers84's [1]_
        equations 7 and 8.

        Returns
        -------
        turb : the turbulent velocity timeseries array (3 x nz x ny x
               nt) for this PyTurbSim run.

        Notes
        -----

        1) Veers84's equation 7 [1]_ is actually a 'Cholesky
        Factorization'.  Therefore, rather than writing this
        functionality explicitly we call 'cholesky' routines to do
        this work.

        2) This function uses one of two methods for computing the
        Cholesky factorization.  If the Fortran library tslib is
        available it is used (it is much more efficient), otherwise
        the numpy implementation of Cholesky is used.

        .. [1] Veers, Paul (1984) 'Modeling Stochastic Wind Loads on
               Vertical Axis Wind Turbines', Sandia Report 1909, 17
               pages.

        """
        grid = self.grid
        tmp = np.zeros((grid.n_comp, grid.n_z, grid.n_y, grid.n_f + 1),
                       dtype=ts_complex)
        if dbg:
            self.timer.start()
        # First calculate the 'base' set of random phases:
        phases = self.phase(self)
        # Now correlate the phases at each point to set the Reynold's stress:
        phases = self.stress.calc_phases(phases)
        # Now correlate the phases between points to set the spatial coherence:
        phases = self.cohere.calc_phases(phases)
        # Now multiply the phases by the spectrum...
        tmp[..., 1:] = np.sqrt(self.spec.array) * grid.reshape(phases)
        # and compute the inverse fft to produce the timeseries:
        ts = irfft(tmp)
        if dbg:
            self.timer.stop()
        # Select only the time period requested:
        # Grab a random number of where to cut the timeseries.
        i0_out = self.randgen.randint(grid.n_t - grid.n_t_out + 1)
        ts = ts[..., i0_out:i0_out + grid.n_t_out] / (grid.dt / grid.n_f) ** 0.5
        ts -= ts.mean(-1)[..., None]  # Make sure the turbulence has zero mean.
        return ts


class tsdata(gridProps):
    """
    TurbSim output data object.  In addition to the output of a
    simulation (velocity timeseries array) it also includes all
    information for reproducing the simulation.

    Parameters
    ----------
    grid : :class:`gridObj`
           TurbSim data objects are initialized with a TurbSim grid.
    """

    @property
    def _sumdict(self):
        out = dict()

        # Start by pulling values from the config file
        # if there was one.
        if 'config' in self.info:
            out.update(self.info['config'])

        uhub = out['uhub'] = statObj(self.uhub)
        out['vhub'] = statObj(self.vhub, uhub.mean)
        out['whub'] = statObj(self.whub, uhub.mean)
        out['hhub'] = statObj(np.sqrt(self.uhub ** 2 + self.vhub ** 2))
        out['grid'] = self.grid
        out['upvp'] = statObj(self.uhub * self.vhub)
        out['upwp'] = statObj(self.vhub * self.whub)
        out['vpwp'] = statObj(self.vhub * self.whub)
        out['upvp'].scale = 1
        out['upwp'].scale = 1
        out['vpwp'].scale = 1
        out['tke'] = statObj((self.uturb ** 2).sum(0))
        out['tke'] = statObj((self.uturb ** 2).sum(0))
        out['ctke'] = statObj(0.5 * np.sqrt(
            (self.uturb[0] * self.uturb[1]) ** 2 +
            (self.uturb[0] * self.uturb[2]) ** 2 +
            (self.uturb[1] * self.uturb[2]) ** 2))
        out['u_sigma'] = self.uturb[0].flatten().std()
        out['v_sigma'] = self.uturb[1].flatten().std()
        out['w_sigma'] = self.uturb[2].flatten().std()
        out['TurbModel_desc'] = self.info['specModel']['description']
        out['RandSeed1'] = self.info['RandSeed']

        out['profModel_sumstring'] = self.info['profModel']['sumstring']
        out['specModel_sumstring'] = self.info['specModel']['sumstring']
        out['stressModel_sumstring'] = self.info['stressModel']['sumstring']
        out['cohereModel_sumstring'] = self.info['cohereModel']['sumstring']
        out['ver'] = ver
        out['NowDate'] = time.strftime('%a %b %d, %Y', self.info['StartTime'])
        out['NowTime'] = time.strftime('%H:%M:%S', self.info['StartTime'])
        out['RunTime'] = self.info['RunTime']
        out['FreqNyquist'] = self.f[-1]
        out['GridBase'] = self.grid.z[0]
        out['HeightOffset'] = 0.0  # Is this correct?
        out['ydata'] = self.grid.y
        out['z_ustd'] = np.concatenate((self.grid.z[:, None], self.uturb[0].std(-1)), axis=1)
        out['z_vstd'] = np.concatenate((self.grid.z[:, None], self.uturb[1].std(-1)), axis=1)
        out['z_wstd'] = np.concatenate((self.grid.z[:, None], self.uturb[2].std(-1)), axis=1)
        u, v, w = self.uprof.mean(-1)[:, :, None]
        out['WINDSPEEDPROFILE'] = np.concatenate((
            self.grid.z[:],
            np.sqrt(u[0,:][0] ** 2 + v[0,:][0] ** 2),
            np.angle(u[0,:][0] + 1j * v[0,:][0]) * 180 / np.pi,
            u[0,:][0], v[0,:][0], w[0,:][0], ))
        out['HFlowAng'] = np.angle(self.uprof[0][self.ihub] + 1j * self.uprof[1][self.ihub])
        out['VFlowAng'] = np.angle(self.uprof[0][self.ihub] + 1j * self.uprof[2][self.ihub])
        out['TurbModel'] = self.info['specModel']['name']
        out['gridheader'] = '---------   ' * self.grid.n_y
        for nm in ['Zref', 'RefHt', 'ZRef', ]:
            if nm in self.info['profModel']['params']:
                out['RefHt'] = self.info['profModel']['params'][nm]
        for nm in ['URef', 'Uref', ]:
            if nm in self.info['profModel']['params']:
                out['URef'] = self.info['profModel']['params'][nm]
        out['PLExp'] = self.info['profModel']['params'].get('PLexp', None)
        return out

    def __getitem__(self, ind):
        if not hasattr(ind, '__len__'):
            ind = [ind]
        else:
            list(ind)
        for idx, val in enumerate(ind):
            if val.__class__ is not slice:
                ind[idx] = slice(val, val + 1)
        out = type(self)(self.grid[ind])
        ind = [slice(None)] + list(ind)
        out.uturb = self.uturb[ind]
        out.uprof = self.uprof[ind]
        return out

    @property
    def parameters(self,):
        out = {}
        if hasattr(self, 'info'):
            for nm in ['profModel_params',
                       'specModel_params',
                       'cohereModel_params',
                       'stressModel_params']:
                if nm in self.info:
                    out.update(self.info[nm])
        return out

    def __init__(self, grid):
        """
        Initialize a tsdata object with a grid object.
        """
        self.grid = grid
        self.ews_update=False
    @property
    def shape(self,):
        """
        The shape of the turbulence time-series (output) array.
        """
        return self.uturb.shape

    @property
    def ihub(self,):
        """
        The index of the hub.
        """
        return self.grid.ihub

    @property
    def time(self,):
        """
        The time vector, in seconds, starting at zero.
        """
        if not hasattr(self, '_time'):
            self._time = np.arange(0, self.uturb.shape[-1] * self.dt, self.dt)
        return self._time

    def __repr__(self,):
        return ('<TurbSim data object:\n'
                '%d %4.2fs-timesteps, %0.2fx%0.2fm (%dx%d) z-y grid (hubheight=%0.2fm).>' %
                (self.uturb.shape[-1],
                 self.dt,
                 self.grid.height,
                 self.grid.width,
                 self.grid.n_z,
                 self.grid.n_y,
                 self.grid.zhub))
    def ews(self):
        self.ews_update=True
        self.uprof = np.zeros([3] + list(self.grid.shape) + [self.uturb.shape[-1]])
        import numpy as npp
        positive_shear = True
        negative_shear = False
        for i in range(self.uturb.shape[-1]):
            T = 12
            D = 236
            t = i * self.dt
            if 0 < t and t < T:
                sig1 = self.info["config"]["IECturbc"]/100 * (0.75 * self.info["profModel"]["params"]["Uref"] + 5.6)
                if positive_shear:
                    liste = self.info["profModel"]["params"]["Uref"] * (
                                self.grid.z / self.info["profModel"]["params"]["Zref"]) ** \
                           self.info["profModel"]["params"]["PLexp"] + (
                                       (self.grid.z - self.info["profModel"]["params"]["Zref"]) / D) * (
                                       2.5 + 0.2 * 6.4 * sig1 * (D / 42) ** 0.25) * (1 - npp.cos(2 * npp.pi * t / T))
                if negative_shear:
                    liste = self.info["profModel"]["params"]["Uref"] * (
                                self.grid.z / self.info["profModel"]["params"]["Zref"]) ** \
                           self.info["profModel"]["params"]["PLexp"] - (
                                       (self.grid.z - self.info["profModel"]["params"]["Zref"]) / D) * (
                                       2.5 + 0.2 * 6.4 * sig1 * (D / 42) ** 0.25) * (1 - npp.cos(2 * npp.pi * t / T))
            else:
                liste = self.info["profModel"]["params"]["Uref"] * (
                        self.grid.z / self.info["profModel"]["params"]["Zref"]) ** self.info["profModel"]["params"][
                           "PLexp"]
            for k in range(self.grid.n_y):
                self.uprof[0, :, k, i] = liste



    @property
    def utotal(self,):
        """
        The total (mean + turbulent), 3-d velocity array
        """
        if self.ews_update==False:
            self.ews()

        return self.uturb + self.uprof#[:, :, :, None]

    @property
    def u(self,):
        """
        The total (mean + turbulent), u-component of velocity.
        """
        if self.ews_update==False:
            self.ews()
        return self.uturb[0] + self.uprof[0, :, :]

    @property
    def v(self,):
        """
        The total (mean + turbulent), v-component of velocity.
        """
        return self.uturb[1] + self.uprof[1, :, :]

    @property
    def w(self,):
        """
        The total (mean + turbulent), w-component of velocity.
        """
        return self.uturb[2] + self.uprof[2, :, :]

    @property
    def UHUB(self,):
        """
        The hub-height mean velocity.
        """
        if self.ews_update==False:
            self.ews()
        return self.uprof[0][self.ihub][0]

    @property
    def uhub(self,):
        """
        The hub-height u-component time-series.
        """
        return self.u[self.ihub]

    @property
    def vhub(self,):
        """
        The hub-height v-component time-series.
        """
        return self.v[self.ihub]

    @property
    def whub(self,):
        """
        The hub-height w-component time-series.
        """
        return self.w[self.ihub]

    @property
    def tke(self,):
        """
        The turbulence kinetic energy.
        """
        return (self.uturb ** 2).mean(-1)

    @property
    def ctke(self,):
        return 0.5 * np.sqrt((self.stress ** 2).mean(-1).sum(0))

    @property
    def Ti(self,):
        """
        The turbulence intensity, std(u')/U, at each point in the grid.
        """
        return np.std(self.uturb[0], axis=-1) / self.uprof[0]

    @property
    def stress(self,):
        """
        The Reynold's stress tensor.
        """
        if not hasattr(self, '_dat_stress'):
            self._stress_dat = np.concatenate(
                (np.mean(self.uturb[0] * self.uturb[1], axis=-1)[None],
                 np.mean(self.uturb[0] * self.uturb[2], axis=-1)[None],
                 np.mean(self.uturb[1] * self.uturb[2], axis=-1)[None]),
                0)
        return self._stress_dat

    @property
    def upvp_(self,):
        """
        The u'v' component of the Reynold's stress.
        """
        return self.stress[0]

    @property
    def upwp_(self,):
        """
        The u'w' component of the Reynold's stress.
        """
        return self.stress[1]

    @property
    def vpwp_(self,):
        """
        The v'w' component of the Reynold's stress.
        """
        return self.stress[2]

    @property
    def stats(self,):
        """
        Compute and return relevant statistics for this turbsim time-series.

        Returns
        -------
        stats : dict
                A dictionary containing various statistics of interest.
        """
        slc = [slice(None)] + list(self.ihub)
        stats = {}
        stats['Ti'] = self.tke[slc] / self.UHUB
        return stats

    def write_formatted(self, filename):
        """
        Save the data in this tsdata object in 'formatted' format (.u, .v, .w files).

        Parameters
        ----------

        filename : string
                '.u', '.v', and '.w' will be appended to the end of the filename.
        """
        write.formatted(filename, self)

    def write_bladed(self, filename):
        """
        Save the data in this tsdata object in 'bladed' format (.wnd).

        Parameters
        ----------
        filename : str
                   The filename to which the data should be written.

        """
        write.bladed(filename, self)

    def write_turbsim(self, filename):
        """Save the data in this tsdata object in 'TurbSim' format.

        Parameters
        ----------
        filename : str
                   The filename to which the data should be written.
        """
        write.turbsim(filename, self)

    def write_sum(self, filename):
        """
        Currently PyTurbSim does not support writing summary (.sum) files.
        """
        write.sum(filename, self._sumdict)

    if write.h5py is not None:
        def write_hdf5(self, filename):
            """Save the data in this tsdata object as an hdf5 file.

            Parameters
            ----------
            filename : str
                       The filename to which the data should be written.
            """
            write.hdf5(filename, self)
