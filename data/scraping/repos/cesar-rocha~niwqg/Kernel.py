import numpy as np
from numpy import pi
import logging, os
import h5py

from .Diagnostics import *
from .Saving import *

class Kernel(object):

    """ Python class that the kernel of single-vertical wavenumber near-inertial
        wave models–––different wave models are defined as subclasses that define
        the barotropic quasigeostrophic potential vorticity inversion. The model is
        pseudospectral model in a doubly periodic domain. Physical parameters observe
        SI units.

    Parameters
    -----------
        nx: integer (optional)
                Number of grid points in the x-direction.
                The number of modes is nx/2+1.
        ny: integer (optional)
                Number of grid points in the y-direction.
                If None, then ny=nx.
        L:  float (optional)
                Domain size.
        dt: float (optional)
                Time step for time integration.
        twrite: integer (optional)
                Print model status to screen every twrite time steps.
        tmax: float (optional)
                Total time of simulation.
        U: float (optional)
                Uniform zonal flow
        f:  float (optional)
                Coriolis frequency
        N:  float (optional)
                Buoyancy frequency
        m:  float (optional)
                Vertical wavenumber of near-inertial waves
        use_filter: bool (optional)
                If True, then uses exponential spectral filter.
        nu4: float (optional)
                Fouth-order hyperdiffusivity of potential vorticity.
        nu: float (optional)
                Diffusivity of potential vorticity.
        mu: float (optional)
                Linear drag of potential vorticity.
        nu4w: float (optional)
                Fouth-order hyperviscosity for near-inertial waves.
        nuw: float (optional)
                Viscosity for for near-inertial waves.
        muw: float (optional)
                Linear drag for for near-inertial waves.
        dealias: bool (optional)
                If True, then dealias solution using 2/3 rule.
        save_to_disk: bool (optional)
                If True, then save parameters and snapshots to disk.
        overwrite: bool (optional)
                If True, then overwrite extant files.
        tsave_snapshots: integer (optional)
                Save snapshots every tsave_snapshots time steps.
        tdiags: integer (optional)
                Calculate diagnostics every tdiags time steps.
        path: string (optional)
                Location for saving output files.
    """


    def __init__(
        self,
        nx=128,
        ny=None,
        L=5e5,
        dt=10000.,
        twrite=1000.,
        tmax=250000.,
        use_filter = True,
        cflmax = 0.8,
        U = .0,
        f = 1.e-4,
        N = 0.01,
        m = 0.025,
        g= 9.81,
        nu4=0,
        nu4w=0,
        nu=20,
        nuw=50.,
        mu=0,
        muw=0,
        dealias = False,
        save_to_disk=False,
        overwrite=True,
        tsave_snapshots=10,
        tdiags=10,
        path = 'output/',
        use_mkl=False,
        nthreads=1):

        self.nx = nx
        self.ny = nx
        self.L = L
        self.W = L

        self.dt = dt
        self.twrite = twrite
        self.tmax = tmax
        self.dealias = dealias

        self.U = U
        self.g = g
        self.nu4 = nu4
        self.nu4w = nu4w
        self.nu = nu
        self.nuw = nuw
        self.mu = mu
        self.muw = muw
        self.f = f
        self.N = N
        self.m = m
        self.kappa = self.m*self.f/self.N
        self.kappa2 = self.kappa**2
        self.cflmax = cflmax

        self.hslash = self.f/self.kappa2

        self.save_to_disk = save_to_disk
        self.overwrite = overwrite
        self.tsnaps = tsave_snapshots

        self.tdiags = tdiags
        self.path = path

        self.use_filter = use_filter

        self.use_mkl = use_mkl
        self.nthreads = nthreads

        self._initialize_logger()
        self.logger.info(self.model)
        self._initialize_grid()
        self._allocate_variables()
        self._initialize_filter()
        self._initialize_etdrk4()
        self._initialize_time()

        initialize_save_snapshots(self,self.path)
        save_setup(self,)

        self._initialize_fft()

        self._initialize_diagnostics()

    def _allocate_variables(self):
        """ Allocate variables in memory """

        raise NotImplementedError(
            'needs to be implemented by Model subclass')


    def run_with_snapshots(self, tsnapstart=0., tsnapint=432000.):

        """ Run the model for prescribed time and yields to user code.

            Parameters
            ----------

            tsnapstart : float
                            The timestep at which to begin yielding.
            tstapint : int (number of time steps)
                            The interval at which to yield.

        """

        tsnapints = np.ceil(tsnapint/self.dt)

        while(self.t < self.tmax):
            self._step_forward()
            if self.t>=tsnapstart and (self.tc%tsnapints)==0:
                yield self.t
        return

    def run(self):

        """ Run the model until the end (`tmax`).

            The algorithm is:
                1) Save snapshots (i.e., save the initial condition).
                2) Take a tmax/dt steps forward.
                3) Save diagnostics.
        """

        # save initial conditions
        if self.save_to_disk:
            save_snapshots(self,fields=['t','q','phi'])

        # run the model
        while(self.t < self.tmax):
            self._step_forward()

        # save diagnostics
        if self.save_to_disk:
            save_diagnostics(self)

    def _step_forward(self):

        """ Step solutions forwards. The algorithm is:
                1) Take one time step with ETDRK4 scheme.
                2) Incremente diagnostics.
                3) Print status.
                4) Save snapshots.
        """

        self._step_etdrk4()
        increment_diagnostics(self,)
        self._print_status()
        save_snapshots(self,fields=['t','q','phi'])

    def _initialize_time(self):

        """ Initialize model clock and other time variables.
        """

        self.t=0        # time
        self.tc=0       # time-step number

    def _initialize_grid(self):

        """ Create spatial and spectral grids and normalization constants.
        """

        self.x,self.y = np.meshgrid(
            np.arange(0.5,self.nx,1.)/self.nx*self.L,
            np.arange(0.5,self.ny,1.)/self.ny*self.W )

        self.dk = 2.*pi/self.L
        self.dl = 2.*pi/self.L

        # wavenumber grids
        self.nl = self.ny
        self.nk = self.nl
        self.ll = self.dl*np.append( np.arange(0.,self.nx/2),
            np.arange(-self.nx/2,0.) )
        self.kk = self.ll.copy()

        self.k, self.l = np.meshgrid(self.kk, self.ll)
        self.ik = 1j*self.k
        self.il = 1j*self.l

        # physical grid spacing
        self.dx = self.L / self.nx
        self.dy = self.W / self.ny

        # constant for spectral normalizations
        self.M = self.nx*self.ny

        # isotropic wavenumber^2 grid
        # the inversion is not defined at kappa = 0
        self.wv2 = self.k**2 + self.l**2
        self.wv = np.sqrt( self.wv2 )
        self.wv4 = self.wv2**2

        iwv2 = self.wv2 != 0.
        self.wv2i = np.zeros_like(self.wv2)
        self.wv2i[iwv2] = self.wv2[iwv2]**-1

    def _initialize_filter(self):

        """ Set up spectral filter or dealiasing."""

        if self.use_filter:
            cphi=0.65*pi
            wvx=np.sqrt((self.k*self.dx)**2.+(self.l*self.dy)**2.)
            self.filtr = np.exp(-23.6*(wvx-cphi)**4.)
            self.filtr[wvx<=cphi] = 1.
            self.logger.info(' Using filter')
        elif self.dealias:
            self.filtr = np.ones_like(self.wv2)
            self.filtr[self.nx//3:2*self.nx//3,:] = 0.
            self.filtr[:,self.ny//3:2*self.ny//3] = 0.
            self.logger.info(' Dealiasing with 2/3 rule')
        else:
            self.filtr = np.ones_like(self.wv2)
            self.logger.info(' No dealiasing; no filter')

    def _initialize_logger(self):

        """ Initialize logger.
        """

        self.logger = logging.getLogger(__name__)

        fhandler = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s: %(message)s')

        fhandler.setFormatter(formatter)

        if not self.logger.handlers:
            self.logger.addHandler(fhandler)

        self.logger.setLevel(10)

        self.logger.propagate = False
        self.logger.info(' Logger initialized')


    def _step_etdrk4(self):

        """ Take one step forward using an exponential time-dfferencing method
            with a Runge-Kutta 4 scheme.

            Rereferences
            ------------
            See Cox and Matthews, J. Comp. Physics., 176(2):430-455, 2002.
            Kassam and Trefethen, IAM J. Sci. Comput., 26(4):1214-233, 2005.

        """

        self._calc_energy_conversion()
        k1 = -(self.gamma1+self.gamma2) + (self.xi1+self.xi2) + self._calc_ep_psi()
        p1 = self.gamma1+self.gamma2 + self._calc_chi_phi()
        a1 = self._calc_ep_phi()

        # q-equation
        self.qh0 = self.qh.copy()
        Fn0 = -self.jacobian_psi_q()
        self.qh = (self.expch_h*self.qh0 + Fn0*self.Qh)*self.filtr
        self.qh1 = self.qh.copy()

        # phi-equation
        self.phih0 = self.phih.copy()
        Fn0w = -self.jacobian_psi_phi() - 0.5j*self.fft(self.phi*self.q_psi)
        self.phih = (self.expch_hw*self.phih0 + Fn0w*self.Qhw)*self.filtr
        self.phih1 = self.phih.copy()

        # q-equation
        self.phi = self.ifft(self.phih)
        self._invert()
        self._calc_rel_vorticity()

        self._calc_energy_conversion()
        k2 = -(self.gamma1+self.gamma2) + (self.xi1+self.xi2) + self._calc_ep_psi()
        p2 = self.gamma1+self.gamma2 + self._calc_chi_phi()
        a2 = self._calc_ep_phi()

        Fna = -self.jacobian_psi_q()
        self.qh = (self.expch_h*self.qh0 + Fna*self.Qh)*self.filtr

        # phi-equation
        Fnaw = -self.jacobian_psi_phi() - 0.5j*self.fft(self.phi*self.q_psi)
        self.phih = (self.expch_hw*self.phih0 + Fnaw*self.Qhw)*self.filtr

        # q-equation
        self.phi = self.ifft(self.phih)
        self._invert()
        self._calc_rel_vorticity()

        self._calc_energy_conversion()
        k3 = -(self.gamma1+self.gamma2) + (self.xi1+self.xi2) + self._calc_ep_psi()
        p3 = self.gamma1+self.gamma2 + self._calc_chi_phi()
        a3 = self._calc_ep_phi()

        Fnb = -self.jacobian_psi_q()
        self.qh = (self.expch_h*self.qh1 + ( 2.*Fnb - Fn0 )*self.Qh)*self.filtr

        # phi-equation
        Fnbw = -self.jacobian_psi_phi() - 0.5j*self.fft(self.phi*self.q_psi)
        self.phih = (self.expch_hw*self.phih1 + ( 2.*Fnbw - Fn0w )*self.Qhw)*self.filtr

        # q-equation
        self.phi = self.ifft(self.phih)
        self._invert()
        self._calc_rel_vorticity()

        self._calc_energy_conversion()
        k4 = -(self.gamma1+self.gamma2) + (self.xi1+self.xi2) + self._calc_ep_psi()
        p4 = self.gamma1+self.gamma2 + self._calc_chi_phi()
        a4 = self._calc_ep_phi()

        Fnc = -self.jacobian_psi_q()
        self.qh = (self.expch*self.qh0 + Fn0*self.f0 +  2.*(Fna+Fnb)*self.fab\
                  + Fnc*self.fc)*self.filtr

        # phi-equation
        Fncw = -self.jacobian_psi_phi() - 0.5j*self.fft(self.phi*self.q_psi)
        self.phih = (self.expchw*self.phih0 + Fn0w*self.f0w +  2.*(Fnaw+Fnbw)*self.fabw\
                  + Fncw*self.fcw)*self.filtr


        self.Ke += self.dt*(k1 + 2*(k2+k3) + k4)/6.
        self.Pw += self.dt*(p1 + 2*(p2+p3) + p4)/6.
        self.Kw += self.dt*(a1 + 2*(a2+a3) + a4)/6.

        # invert
        self.phi = self.ifft(self.phih)
        self._invert()
        self._calc_rel_vorticity()


    def _initialize_etdrk4(self):

        """ Compute coefficients of the exponential time-dfferencing method
            with a Runge-Kutta 4 scheme.

            Rereferences
            ------------
            See Cox and Matthews, J. Comp. Physics., 176(2):430-455, 2002.
            Kassam and Trefethen, IAM J. Sci. Comput., 26(4):1214-233, 2005.

        """

        #
        # coefficients for q-equation
        #

        # the exponent for the linear part
        self.c = np.zeros((self.nl,self.nk),self.dtype_cplx) - 1j*self.k*self.U
        self.c += -self.nu4*self.wv4 - self.nu*self.wv2 - self.mu
        ch = self.c*self.dt
        self.expch = np.exp(ch)
        self.expch_h = np.exp(ch/2.)
        self.expch2 = np.exp(2.*ch)

        M = 32    # number of points for line integral in the complex plane
        rho = 1.  # radius for complex integration
        r = rho*np.exp(2j*np.pi*((np.arange(1.,M+1))/M)) # roots for integral
        LR = ch[...,np.newaxis] + r[np.newaxis,np.newaxis,...]
        LR2 = LR*LR
        LR3 = LR2*LR
        self.Qh   =  self.dt*(((np.exp(LR/2.)-1.)/LR).mean(axis=-1))
        self.f0  =  self.dt*( ( ( -4. - LR + ( np.exp(LR)*( 4. - 3.*LR + LR2 ) ) )/ LR3 ).mean(axis=-1) )
        self.fab =  self.dt*( ( ( 2. + LR + np.exp(LR)*( -2. + LR ) )/ LR3 ).mean(axis=-1) )
        self.fc  =  self.dt*( ( ( -4. -3.*LR - LR2 + np.exp(LR)*(4.-LR) )/ LR3 ).mean(axis=-1) )

        #
        # coefficients for phi-equation
        #

        # the exponent for the linear part
        self.c = np.zeros((self.nl,self.nk),self.dtype_cplx)  -1j*self.k*self.U
        self.c += -self.nu4w*self.wv4 - 0.5j*self.f*(self.wv2/self.kappa2)\
                        - self.nuw*self.wv2 - self.muw
        ch = self.c*self.dt
        self.expchw = np.exp(ch)
        self.expch_hw = np.exp(ch/2.)
        self.expch2w = np.exp(2.*ch)

        LR = ch[...,np.newaxis] + r[np.newaxis,np.newaxis,...]
        LR2 = LR*LR
        LR3 = LR2*LR
        self.Qhw   =  self.dt*(((np.exp(LR/2.)-1.)/LR).mean(axis=-1))
        self.f0w  =  self.dt*( ( ( -4. - LR + ( np.exp(LR)*( 4. - 3.*LR + LR2 ) ) )/ LR3 ).mean(axis=-1) )
        self.fabw =  self.dt*( ( ( 2. + LR + np.exp(LR)*( -2. + LR ) )/ LR3 ).mean(axis=-1) )
        self.fcw  =  self.dt*( ( ( -4. -3.*LR - LR2 + np.exp(LR)*(4.-LR) )/ LR3 ).mean(axis=-1) )


    def jacobian_psi_phi(self):

        """ Compute the advective term–––the Jacobian between psi and phi.

        Returns
        -------
        complex array of floats
            The Fourier transform of Jacobian(psi,phi)
        """

        jach = self.fft( (self.u*self.phix + self.v*self.phiy) )
        jach[0,0] = 0
        return jach

    def jacobian_psi_q(self):

        """ Compute the advective term–––the Jacobian between psi and q.

        Returns
        -------
        complex array of floats
            The Fourier transform of Jacobian(psi,q)
        """

        self.u, self.v = self.ifft(-self.il*self.ph).real, self.ifft(self.ik*self.ph).real
        q = self.ifft(self.qh).real
        jach = self.ik*self.fft(self.u*q) + self.il*self.fft(self.v*q)
        jach[0,0] = 0
        #jach[0],jach[:,0] = 0, 0
        return jach

    def _invert(self):
        raise NotImplementedError(
            'needs to be implemented by Model subclass')

    def _calc_rel_vorticity(self):

        """ Compute the geostrophic relative vorticity–––the Laplacian of the
            streamfuctions.

            This methods is surpassed by subclass-specific methods.

        """

        self.q_psi = (self.q)

    def _calc_strain(self):

        """ Compute the geostrophic rate of strain.
        """
        pxx,pyy = self.ifft(-self.k*self.k*self.ph).real, self.ifft(-self.l*self.l*self.ph).real
        pxy = self.ifft(-self.k*self.l*self.ph).real
        self.qg_strain =  4*(pxy**2)+(pxx-pyy)**2

    def _calc_OW(self):

        """ Compute the Okubo-Weiss parameter.
        """

        self._calc_rel_vorticity()
        self._calc_strain()
        return self.qg_strain**2 - self.q_psi**2

    def set_q(self,q):

        """ Initialize the potential vorticity.

        Parameters
        ----------
        q: an array of floats of dimension (nx,ny):
                The potential vorticity in physical space.
        """

        self.q = q
        self.qh = self.fft(self.q)
        self._invert()
        self._calc_rel_vorticity()
        self.u, self.v = self.ifft(-self.il*self.ph).real, self.ifft(self.ik*self.ph).real
        self.Ke = self.ke = self._calc_ke_qg()


    def set_phi(self,phi):

        """ Initialize the near-inertial velocity =.

        Parameters
        ----------
        phi: an array of complex floats of dimension (nx,ny):
                The near-inertial velocity, phi = uw + i vw, in physical space.
        """

        self.phi = phi
        self.phih = self.fft(self.phi)
        self.Pw = self._calc_pe_niw()
        self.Kw = self._calc_ke_niw()

    def _initialize_fft(self):

        """ Define the two-dimensional FFT methods.
        """

        if self.use_mkl:
            import mkl
            mkl.set_num_threads(self.nthreads)
            import mkl_fft
            self.fft =  (lambda x : mkl_fft.fft2(x))
            self.ifft = (lambda x : mkl_fft.ifft2(x))
        else:
            self.fft =  (lambda x : np.fft.fft2(x))
            self.ifft = (lambda x : np.fft.ifft2(x))

    def _print_status(self):

        """ Print out the the model status.
                Step: integer
                        Number of time steps completed
                Time: float
                        The elapsed time.
                P: float
                        The percentage of simulation completed.
                Ke: float
                        The geostrophic kinetic energy.
                Kw: float
                        The near-inertial kinetic energy.
                Pw: float
                        The near-inertial potential energy.
                CFL: float
                        The CFL number.
        """

        self.tc += 1
        self.t += self.dt

        if  (self.tc % self.twrite)==0:
            self.ke = self._calc_ke_qg()
            self.kew = self._calc_ke_niw()
            self.pew = self._calc_pe_niw()
            self.cfl = self._calc_cfl()
            self.logger.info('Step: %4i, Time: %2.1e, P: %2.1e, Ke: %4.3e, Kw: %4.3e, Pw: %4.3e, CFL: %3.2f'
                    , self.tc,self.t, self.t/self.tmax,self.ke,self.kew,self.pew,self.cfl )

            assert self.cfl<self.cflmax, self.logger.error('CFL condition violated')

    def _calc_ke_qg(self):
        """ Compute geostrophic kinetic energy, Ke. """
        return 0.5*self.spec_var(self.wv*self.ph)

    def _calc_ke_niw(self):
        """ Compute near-inertial kinetic energy, Kw. """
        return 0.5*(np.abs(self.phi)**2).mean()

    def _calc_pe_niw(self):
        """ Compute near-inertial potential energy, Pw. """
        self.phix, self.phiy = self.ifft(self.ik*self.phih),self.ifft(self.il*self.phih)
        return 0.25*( np.abs(self.phix)**2 +  np.abs(self.phiy)**2 ).mean()/self.kappa2

    def _calc_conc(self):
        """ Compute the correlation, C, between near-inertial velocity variance and
            relative vorticity.
            A measure of wave concentration in cyclones of anticyclones.
        """
        self.upsilon = np.abs(self.phi)**2 -  (np.abs(self.phi)**2).mean()
        return (self.upsilon*self.q_psi).mean()/self.upsilon.std()/self.q_psi.std()

    def _calc_skewness(self):
        """ Compute skewness of relative vorticity. """
        return ( (self.q_psi**3).mean() / (((self.q_psi**2).mean())**1.5) )

    def _calc_ens(self):
        """ Compute geostrophic potential enstrophy, S. """
        return 0.5*(self.q**2).mean()

    def _calc_ep_phi(self):
        """ Compute dissipation of Kw.  """
        return -self.nu4w*(np.abs(self.lapphi)**2).mean()\
                - self.nuw*(np.abs(self.phix)**2+np.abs(self.phiy)**2).mean()\
                -self.muw*(np.abs(self.phi)**2).mean()

    def _calc_ep_psi(self):
        """ Compute dissipation of QG KE. """
        lap2psi = self.ifft(self.wv4*self.ph).real
        lapq = self.ifft(-self.wv2*self.qh).real
        return self.nu4*(self.q*lap2psi).mean() - self.nu*(self.p*lapq).mean()\
                + self.mu*(self.p*self.q).mean()

    def _calc_chi_q(self):
        """"  Compute dissipation of S. """
        return -self.nu4*self.spec_var(self.wv2*self.qh)

    def _calc_chi_phi(self):
        """"  Compute dissipation of Pw. """
        lphix, lphiy = self.ifft(-self.ik*self.wv2*self.phih),\
                            self.ifft(-self.il*self.wv2*self.phih)
        return -0.5*self.nu4w*(np.abs(lphix)**2 + np.abs(lphiy)**2).mean()/self.kappa2\
                -0.5*self.nuw*(np.abs(self.lapphi)**2).mean()/self.kappa2\
                -0.5*self.muw*(np.abs(self.phix)**2 + np.abs(self.phiy)**2).mean()/self.kappa2

    def spec_var(self, ph):
        """ Compute variance of a variable `p` from its Fourier transform `ph` """
        var_dens = np.abs(ph)**2 / self.M**2
        var_dens[0,0] = 0.
        return var_dens.sum()

    def _calc_cfl(self):
        """ Compute the CFL number. """
        return np.abs(np.hstack([self.u, self.v,np.abs(self.phi)])).max()*self.dt/self.dx

    def _calc_energy_conversion(self):

        """ Compute energy conversion terms.

                gamma1: the refractive conversion between geostrophic kinetic
                            energy and near-inertial potential energy.
                gamma2: the advective conversion between geostrophic kinetic
                            energy and near-inertial potential energy.
                xi1: the refractive generation of geostrophic kinetic energy due
                            wave dissipation.
                xi2: the advective generation of geostrophic kinetic energy due
                            wave dissipation.
                pi: the conversion of kinetic energy from laterally coherent
                    to incoherent near-inertial waves (a measure of loss of
                    lateral coherence).
        """

        self.u, self.v = self.ifft(-self.il*self.ph).real, self.ifft(self.ik*self.ph).real
        self._calc_rel_vorticity()

        J_psi_phi = self.u*self.phix+self.v*self.phiy
        self.lapphi = np.fft.ifft2(-self.wv2*self.phih)

        # dissipative source of QG KE
        lap2phi = self.ifft(self.wv4*self.phih)
        diss_phi= -self.nu4w*lap2phi + self.nuw*self.lapphi - self.muw*self.phi
        J_diss_phi = -(diss_phi*np.conj(J_psi_phi)).imag
        L_diss_phi = 0.5*(diss_phi*np.conj(self.phi)).real*self.q_psi

        # div fluxes
        divFw = 0.5*self.hslash*(np.conj(self.phi)*self.lapphi).imag

        # correlations
        self.gamma1 = (0.5*self.q_psi*divFw).mean()/self.f
        self.gamma2 = 0.5*self.hslash*((np.conj(self.lapphi)*J_psi_phi).real).mean()/self.f
        self.xi1 = J_diss_phi.mean()/self.f
        self.xi2 = L_diss_phi.mean()/self.f
        self.pi = (0.5*self.phi.mean()*(self.q_psi*np.conj(self.phi)).mean()).imag

    def _calc_icke_niw(self):
        self.ke_niw = self._calc_ke_niw()
        self.cke_niw = 0.5*(np.abs(self.phi.mean())**2)
        self.ike_niw = self.ke_niw-self.cke_niw

    def _initialize_diagnostics(self):

        """ Initialize kernel and subclass-specific the diagnostics dictionary
            with each diganostic and an entry.
        """

        self.diagnostics = dict()
        self._initialize_kernel_diagnostics()
        self._initialize_class_diagnostics()

    def _initialize_kernel_diagnostics(self):


        add_diagnostic(self,'time',
                description='Time',
                units='seconds',
                types = 'scalar',
                function = (lambda self: self.t)
        )

        add_diagnostic(self, 'Ke',
                description='Quasigeostrophic Kinetic Energy, from energy equation',
                units=r'm^2 s^{-2}',
                types = 'scalar',
                function = (lambda self: self.Ke)
        )

        add_diagnostic(self, 'Pw',
                description='NIW Potential Energy, from energy equation',
                units=r'm^2 s^{-2}',
                types = 'scalar',
                function = (lambda self: self.Pw)
        )

        add_diagnostic(self, 'Kw',
                description='NIW Kinetic Energy, from energy equation',
                units=r'm^2 s^{-2}',
                types = 'scalar',
                function = (lambda self: self.Kw)
        )

        add_diagnostic(self, 'ke_qg',
                description='Quasigeostrophic Kinetic Energy',
                units=r'm^2 s^{-2}',
                types = 'scalar',
                function = (lambda self: self._calc_ke_qg())
        )


        add_diagnostic(self, 'ens',
                description='Quasigeostrophic Potential Enstrophy',
                units=r's^{-2}',
                types = 'scalar',
                function = (lambda self: 0.5*(self.q**2).mean())
        )


        add_diagnostic(self, 'ke_niw',
                description='Near-inertial Kinetic Energy',
                units=r'm^2 s^{-2}',
                types = 'scalar',
                function = (lambda self: self.ke_niw)
        )

        add_diagnostic(self, 'cke_niw',
                description='Kinetic Energy of Laterally Coherent Near-Inertial Waves',
                units=r'm^2 s^{-2}',
                types = 'scalar',
                function = (lambda self: self.cke_niw)
        )

        add_diagnostic(self, 'ike_niw',
                description='Kinetic Energy of Laterally Incoherent Near-Inertial Waves',
                units=r'm^2 s^{-2}',
                types = 'scalar',
                function = (lambda self: self.ike_niw)
        )

        add_diagnostic(self, 'pe_niw',
                description='Near-inertial Potential Energy',
                units=r'm^2 s^{-2}',
                types = 'scalar',
                function = (lambda self: self._calc_pe_niw())
        )

        add_diagnostic(self, 'conc_niw',
                description='Correlation between relative vorticity and near-inertial KE',
                units=r'unitless',
                types = 'scalar',
                function = (lambda self: self._calc_conc())
        )

        add_diagnostic(self, 'skew',
                description='Skewness',
                units=r'unitless',
                types = 'scalar',
                function = (lambda self: self._calc_skewness())
        )

        add_diagnostic(self, 'gamma_r',
                description='The energy conversion due to refraction',
                units=r'$m^2 s^{-3}$',
                types = 'scalar',
                function = (lambda self: self.gamma1)
        )

        add_diagnostic(self, 'gamma_a',
                description='The energy conversion due to advection',
                units=r'$m^2 s^{-3}$',
                types = 'scalar',
                function = (lambda self: self.gamma2)
        )

        add_diagnostic(self, 'xi_r',
                description='The QG energy generation due to wave dissipation, vorticity',
                units=r'$m^2 s^{-3}$',
                types = 'scalar',
                function = (lambda self: self.xi1)
        )

        add_diagnostic(self, 'xi_a',
                description='The QG energy generation due to wave dissipation, advection',
                units=r'$m^2 s^{-3}$',
                types = 'scalar',
                function = (lambda self: self.xi2)
        )

        add_diagnostic(self, 'pi',
                description='The NIW kinetic energy conversion from coherent to incoherent',
                units=r'$m^2 s^{-3}$',
                types = 'scalar',
                function = (lambda self: self.pi)
        )

        add_diagnostic(self, 'ep_phi',
                description='The hyperviscous dissipation of NIW kinetic energy',
                units=r'$m^2 s^{-3}$',
                types = 'scalar',
                function = (lambda self: self._calc_ep_phi())
        )

        add_diagnostic(self, 'ep_psi',
                description='The hyperviscous dissipation of QG kinetic energy',
                units=r'$m^2 s^{-3}$',
                types = 'scalar',
                function = (lambda self: self._calc_ep_psi())
        )

        add_diagnostic(self, 'chi_q',
                description='The hyperviscous dissipation of QG kinetic energy',
                units=r'$s^{-3}$',
                types = 'scalar',
                function = (lambda self: self._calc_chi_q())
        )

        add_diagnostic(self, 'chi_phi',
                description='The hyperviscous dissipation of NIW potential energy',
                units=r'$s^{-3}$',
                types = 'scalar',
                function = (lambda self: self._calc_chi_phi())
        )

    def _calc_derived_fields(self):
        """ Compute derived fields necessary for model diagnostics. """
        self._calc_kernel_derived_fields()
        self._calc_class_derived_fields()

    def _calc_kernel_derived_fields(self):
        """ Compute kernel-specific derived fields necessary for model diagnostics. """
        self._calc_energy_conversion()
        self._calc_icke_niw()
