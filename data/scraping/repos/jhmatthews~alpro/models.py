import numpy as np
from scipy.interpolate import interp1d
import types
import os
from scipy.integrate import simps


class units:
    '''
    class containing some units. Should probably use astropy units
    but I find them a bit annoying.
    '''

    def __init__(self):
        self.kpc = 3.0857e21
        self.pc = 3.0857e18
        self.c = 2.997925e10
        self.yr = 3.1556925e7
        self.myr = 3.1556925e13
        self.kyr = 3.1556925e10
        self.radian = 57.29577951308232
        self.msol = 1.989e33
        self.mprot = 1.672661e-24
        self.melec = 9.10938356e-28
        self.melec_csq = self.melec * self.c * self.c
        self.mprot_csq = self.mprot * self.c * self.c
        self.e = 4.8032045057134676e-10     # fundamental charge
        self.kev = 1.602192e-9  # kilo electron volts in CGS
        self.ev = 1.602192e-12  # electron volts in CGS
        self.kb = 1.38062e-16   # boltzmann
        self.h = 6.6262e-27     # plank
        self.hbar = self.h / np.pi / np.pi
        self.hbar_ev = 6.582119569e-16
        self.g = 6.670e-8       # gravitational
        self.hbar_c = self.hbar * self.c
        self.alpha = self.e * self.e / self.hbar_c
        self.thomson = 0.66524e-24
        self.unit_gauss_natural = 0.01953548032
        self.unit_length_natural = 50676.79373667135


# class to use for units
unit = units()


def random_angle(size=None):
    '''
    compute theta and phi coordinates for
    a random isotropic angle
    '''
    costheta = (2.0 * np.random.random(size=size)) - 1.0
    phi = 2.0 * np.pi * np.random.random(size=size)
    theta = np.arccos(costheta)
    return (theta, phi)


def fterm(r, C, alpha):
    term1 = -alpha * np.cos(alpha * r)
    term2 = np.sin(alpha * r) / r
    F0 = C * alpha * alpha * (alpha * np.cos(alpha) - np.sin(alpha))
    term3 = F0 * r * r / alpha / alpha
    f = C * (term1 + term2) + term3
    return f


def fprime(r, C, alpha):
    term1 = alpha * alpha * np.sin(alpha * r)
    term2 = (alpha * np.cos(alpha * r) / r) - (np.sin(alpha * r) / r / r)
    F0 = C * alpha * alpha * (alpha * np.cos(alpha) - np.sin(alpha))
    term3 = 2.0 * F0 * r / alpha / alpha
    f = C * (term1 + term2) + term3
    return f


def libanov_Br(r, alpha=5.76, theta=np.pi / 4.0, C=6e-8):
    f = fterm(r, C, alpha)
    Br = 2.0 * np.cos(theta) * f / r / r
    return (-Br)


def get_libanov_B(r, theta=np.pi / 4, Rcavity=93.0, alpha=5.76, C=6e-8):
    rnorm = r / Rcavity
    fr = fterm(rnorm, C, alpha)
    Br = 2.0 * np.cos(theta) * fr / rnorm / rnorm
    Btheta = -np.sin(theta) * fprime(rnorm, C, alpha) / rnorm
    Bphi = alpha * np.sin(theta) * fr / rnorm

    # truncate the field beyond Rcavity
    Btheta[rnorm > 1] = 0.0
    Bphi[rnorm > 1] = 0.0
    Br[rnorm > 1] = 0.0
    return (Btheta, Bphi, Br)


def get_libanov_B_old(r, include_radial=True):
    x = r / (93.0)
    Bx = (0.00312443 * (x**18)) - (0.0319991 * (x**16)) + (0.260311 * (x**14)) - (1.63197 * (x**12)) + \
        (7.58002 * (x**10)) - (24.721 * (x**8)) + (52.3929 * (x**6)) - \
        (63.8794 * (x**4)) + (35.8973 * (x**2)) - 5.86899
    By = (0.0102459 * (x**17)) - (0.0937683 * (x**15)) + (0.671841 * (x**13)) - (3.6406 * (x**11)) + \
        (14.2479 * (x**9)) - (37.7455 * (x**7)) + \
        (61.3611 * (x**5)) - (51.7231 * (x**3)) + (16.9128 * x)

    if include_radial:
        Bz = libanov_Br(x)
        return 1e-6 * Bx, 1e-6 * By, Bz
    else:
        return 1e-6 * Bx, 1e-6 * By


def churazov_density(r):
    '''
    Density function from churazov et al 2003
    Given as equation (2) of Reynolds et al. 2020

    Parameters:
            r 	float
                    distance from cluster centre in kpc
    '''
    term1 = 3.9e-2 / ((1.0 + (r / 80.0)**2)**1.8)
    term2 = 4.05e-3 / ((1.0 + (r / 280.0)**2)**0.87)
    return (term1 + term2)


class ClusterProfile:
    '''
    container for a magnetic field and density profile for a cluster
    '''

    def __init__(self, model="a", plasma_beta=100, B_rms=None, n=None):

        self.plasma_beta = plasma_beta
        self.model = model

        if model == "a":
            # Model A from Reynolds+ 2020
            self.get_B = self.B_modA
            self.density = churazov_density
            self.n0 = self.density(0.0)
            self.B0 = 2.5e-5
            self.B_exponent = 0.7

        elif model == "b":
            # Model B from Reynolds+ 2020
            self.get_B = self.B_modB
            self.density = churazov_density
            self.n25 = self.density(25.0)

        elif model == "flat":
            '''
            allow to just have a uniform field
            '''
            self.B_rms = B_rms
            self.n = n
            self.get_B = self.Bflat
            self.density = churazov_density

        elif model == "murgia":
            self.n0 = 1e-3
            self.r0 = 400.0
            self.B_exponent = 0.5
            self.beta = 0.6
            self.B0 = 1e-6
            self.density = self.beta_density
            self.get_B = self.B_modA

        elif model == "russell":
            self.n0 = 2.63
            self.pl_alpha = 1.16
            self.P0 = 1.85e-9
            self.r_bend = 511.0
            self.a_low = 0.47
            self.a_high = 2.54
            self.density = self.pl_density
            self.get_B = self.BendingPL_B

        elif model == "custom":
            print("Warning: Custom model specified - make sure get_B & density methods are populated or domain set manually!")

        else:
            raise ValueError(
                "ClusterProfile did not understand model type {}".format(model))

    def beta_r(self, r):
        if callable(self.plasma_beta):
            return (self.plasma_beta(r))
        else:
            return (self.plasma_beta)

    def churazov_density(self, r):
        return (churazov_density(r))

    def beta_density(self, r):
        '''
        beta law density.
        '''
        exponent = -3.0 * self.beta / 2.0
        n = self.n0 * (1 + (r / self.r0)**2) ** exponent
        return (n)

    def pl_density(self, r):
        '''
        power law density.
        '''
        return (self.n0 * (r ** -self.pl_alpha))

    def BendingPL_B(self, r):
        '''
        bending power law density.
        '''
        numer = r**-self.a_low
        denom = 1 + (r / self.r_bend)**(self.a_high - self.a_low)
        P = self.P0 * (numer / denom)
        B = np.sqrt(P * 4.0 * np.pi / self.beta_r(r))
        return B

    def Bflat(self):
        '''
        uniform magentic field
        '''
        return (self.B_rms)

    def nflat(self):
        '''
        uniform density
        '''
        return (self.n)

    def B_modA(self, r):
        '''
        Model A from Reynolds et al. 2020

        Parameters:
                r 	float
                        distance from cluster centre in kpc
        '''
        beta = self.beta_r(r)
        return (self.B0 * (self.density(r) / self.n0)
                ** self.B_exponent * 100.0 / beta)

    def B_modB(self, r, B25=7.5e-6):
        '''
        Model B from Reynolds et al. 2020

        Parameters:
                r 	float
                        distance from cluster centre in kpc
        '''
        beta = self.beta_r(r)
        B = B25 * np.sqrt(self.density(r) / self.n25 * 100.0 / beta)
        return (B)

    def profile(self, r):
        '''
        wrapper to the density and magentic field functions
        '''
        return (self.density(r), self.get_B(r))


class ClusterFromFile:
    def __init__(self, fname="Bfield.npy", model_type="cube"):
        # load the array
        self.Bfull = np.load(fname)
        self.N = self.Bfull.shape[0]
        self.mid = self.N // 2
        self.density = churazov_density

        if model_type == "cube":
            if any(i != self.N for i in self.Bfull.shape[:-1]):
                raise ValueError(
                    "File supplied must be cube shaped but has shape {}".format(
                        self.Bfull.shape))
        elif model_type == "1d":
            self.z = self.Bfull[0, :]
            self.B = np.transpose(self.Bfull[1:, :])
            interp_x_temp = interp1d(self.z, self.B[:, 0], kind='slinear')
            interp_y_temp = interp1d(self.z, self.B[:, 1], kind='slinear')
            interp_z_temp = interp1d(self.z, self.B[:, 2], kind='slinear')
            # actual interpolation always done using 2nd order interp
            kind = "quadratic"
            # kind='quadratic'
            self.interp_x = interp1d(self.z, self.B[:, 0], kind=kind)
            self.interp_y = interp1d(self.z, self.B[:, 1], kind=kind)
            self.interp_z = interp1d(self.z, self.B[:, 2], kind=kind)

    def slice(self, z, L=100.0, axis=0, sign=1, degrade=1, normalise=1.0):

        if axis == 0:
            self.B = self.Bfull[:, self.mid, self.mid, :]
        elif axis == 1:
            self.B = self.Bfull[self.mid, :, self.mid, :]
        elif axis == 2:
            self.B = self.Bfull[self.mid, self.mid, :, :]

        if sign > 0:
            self.B = self.B[self.mid:, :]
        else:
            self.B = self.B[:self.mid, :]

        self.B *= normalise
        # take a slice along the B field

        from scipy.interpolate import interp1d
        ztrue = z
        self.z = np.linspace(0, L, len(self.B[:, 0]) / degrade)

        if degrade > 1:
            # these functions will allow us to degrade the resolution using
            # linear spline interp
            interp_x_temp = interp1d(ztrue, self.B[:, 0], kind='slinear')
            interp_y_temp = interp1d(ztrue, self.B[:, 1], kind='slinear')
            interp_z_temp = interp1d(ztrue, self.B[:, 2], kind='slinear')

            self.B = np.zeros((len(self.z), 3))

            self.B[:, 0] = interp_x_temp(self.z)
            self.B[:, 1] = interp_y_temp(self.z)
            self.B[:, 2] = interp_z_temp(self.z)

        elif degrade < 1:
            raise ValueError("degrade needs to be >= 1!")

        # actual interpolation always done using 2nd order interp
        kind = "quadratic"
        # kind='quadratic'
        self.interp_x = interp1d(self.z, self.B[:, 0], kind=kind)
        self.interp_y = interp1d(self.z, self.B[:, 1], kind=kind)
        self.interp_z = interp1d(self.z, self.B[:, 2], kind=kind)

    def get_Bz(self, z):
        Bz = self.interp_z(z)
        return (Bz)

    def get_B(self, z):
        '''
        get the two perpendicular components of the magnetic field at
        distance z
        '''
        Bx = self.interp_x(z)
        By = self.interp_y(z)
        return (Bx, By)

    def profile(self, r):
        return (self.density(r), self.get_B(r))


def omega_p(ne):
    '''
    calculate the plasma frequency in natural (eV) units from an electron density

    Parameters:
            ne 		float/array-like
                            electron density in cm^-3
    '''
    omega_p = np.sqrt(4.0 * np.pi * unit.e * unit.e *
                      ne / unit.melec) * unit.hbar_ev
    return (omega_p)

# Possibly this should be renamed to Domain, or similar


class FieldModel:
    def __init__(self, profile, plasma_beta=100, coherence_r0=None):
        self.profile = profile
        self.beta = plasma_beta

        # coherence_r0 scales the coherence lengths with radius
        # by a factor of (1 + r/coherence_r0), in kpc
        self.coherence_r0 = coherence_r0
        self.Bz = 1.0

    def create_libanov_field(self, deltaL=1.0, Lmax=93.0,
                             density=None, theta=np.pi / 4.0):
        '''
        Initialise  uniform field model of Libanov & Troitsky.

        Parameters:
                deltaL 		float
                                        resolution of domain in kpc

                Lmax 		float
                                        maximum radius in kpc

                density 	str / Nonetype
                                        if None, use vanishing density. if set

        '''
        self.r = np.arange(0, Lmax, deltaL)
        self.deltaL = np.ones_like(self.r) * deltaL
        self.rcen = self.r + (0.5 * self.deltaL)
        self.Bx, self.By, self.Bz = get_libanov_B(self.rcen, theta=theta)
        self.B = np.sqrt(self.Bx**2 + self.By**2)
        self.phi = np.arctan2(self.Bx, self.By)

        if density is None:
            self.ne = 1e-20 * np.ones_like(self.rcen)  #  vanishing density
        elif density == "churazov":
            self.ne = churazov_density(self.rcen)
        else:
            raise ValueError("density keyword must be Nonetype or churazov")
        #self.rm = self.get_rm()
        self.omega_p = omega_p(self.ne)

    def uniform_field_z(self, deltaL=1.0, Lmax=1800.0):
        '''
        Set up a uniform radial field model with a uniform field sampled at N points.

        Parameters:
                deltaL 		float
                                        size of boxes in kpc

                Lmax 		float
                                        size of domain in kpc
        '''
        self.r = np.arange(0, Lmax - deltaL, deltaL)
        self.deltaL = np.ones_like(self.r) * deltaL
        self.rcen = self.r + (0.5 * self.deltaL)
        self.Bx, self.By = 0.0, 0.0
        self.ne, self.Bz = self.profile(self.rcen)
        self.B = np.sqrt(self.Bx**2 + self.By**2)
        self.phi = np.zeros_like(self.Bx)
        self.omega_p = omega_p(self.ne)

    def single_box(self, phi, B, L, ne, N=1):
        '''
        Set up a Field model with a uniform field sampled at N points.

        Parameters:
                phi 	float
                                angle between perpendicular magnetic field and y axis

                B 		float
                                magnetic field strength in Gauss

                L 		float
                                size of domain in kpc

                ne 		float
                                electron density in cm^-3
        '''
        self.r = np.linspace(0, L, N)
        self.deltaL = np.ones_like(self.r) * (L - self.r[-1])
        self.rcen = self.r + (0.5 * self.deltaL)
        self.ne = np.ones_like(self.r) * ne
        self.phi = np.ones_like(self.r) * phi
        self.B = np.ones_like(self.r) * B
        self.Bx = self.B * np.sin(self.phi)
        self.By = self.B * np.cos(self.phi)
        self.Bz = np.zeros_like(self.phi)

    def get_rm(self, cell_centered=True):
        r'''
        Calculate the rotation measure of the field model using Simpson integration.
        Equation is :math:`RM= 812 \int n_e B_z dz` with the field in microGauss

        Returns:
                the rotation measure of the field model in rad m^-2
        '''
        #prefactor = (unit.e ** 3) / 2.0 / np.pi / unit.melec_csq / unit.melec_csq
        prefactor = 812.0
        if cell_centered:
            r = self.rcen
        else:
            r = self.r
        integral = simps(self.ne * self.Bz * 1e6, r)

        return (prefactor * integral)

    def domain_from_slice(self, Cluster, deltaL=1.0, Lmax=500.0, r0=0.0):
        npoints = int((Lmax - r0) // deltaL)
        self.r = np.linspace(r0, Lmax - deltaL, npoints)
        self.deltaL = np.ones_like(self.r) * deltaL
        self.rcen = self.r + (0.5 * self.deltaL)
        self.Bx, self.By = Cluster.get_B(self.rcen)
        self.Bz = Cluster.get_Bz(self.rcen)
        self.B = np.sqrt(self.Bx**2 + self.By**2)
        self.phi = np.arctan2(self.Bx, self.By)
        self.ne = Cluster.density(self.r)
        self.omega_p = omega_p(self.ne)

    def resample_box(self, new_redge, interp1d_kwargs={
                     "kind": "quadratic", "fill_value": "extrapolate"}, profile=True):
        '''
        Resample a box array on to a new 1D grid using 1d interpolation.
        Must be called after the Bx, By, r arrays are already populated.
        '''

        interp_array_r = np.concatenate(
            (self.r[0:1], self.rcen, self.r[-1:] + self.deltaL[-1:]))
        interp_Bx = np.concatenate((self.Bx[0:1], self.Bx, self.Bx[-1:]))
        interp_By = np.concatenate((self.By[0:1], self.By, self.By[-1:]))
        interp_x = interp1d(interp_array_r, interp_Bx, **interp1d_kwargs)
        interp_y = interp1d(interp_array_r, interp_By, **interp1d_kwargs)

        interp_Bz = np.concatenate((self.Bz[0:1], self.Bz, self.Bz[-1:]))
        interp_z = interp1d(interp_array_r, interp_Bz, **interp1d_kwargs)

        # populate new values
        self.rcen = 0.5 * (new_redge[1:] + new_redge[:-1])
        self.Bx = interp_x(self.rcen)
        self.By = interp_y(self.rcen)
        self.Bz = interp_z(self.rcen)
        self.r = new_redge[:-1]
        self.deltaL = new_redge[1:] - new_redge[:-1]
        self.B = np.sqrt(self.Bx**2 + self.By ** 2)
        self.phi = np.arctan2(self.Bx, self.By)
        if profile:
            self.ne, _ = self.profile(self.rcen)
        else:
            interp_ne = np.concatenate((self.ne[0:1], self.ne, self.ne[-1:]))
            interp_n = interp1d(interp_array_r, interp_ne, **interp1d_kwargs)
            self.ne = interp_n(self.rcen)

    def create_box_array(self, L, random_seed, coherence,
                         r0=10.0, cell_centered=True):
        '''
        create an array of random magnetic field boxes by drawing
        random angles and box sizes from coherence_func.

        Parameters:
                L				float
                                                size of domain in kiloparsecs

                random_seed 	int
                                                random number seed

                coherence_func	function or float
                                                function that computes coherence length at distance r,
                                                or a single-value floating point number if the coherence
                                                length is constant.

                r0 				float
                                                inner radius of the calculation (used to excide an inner region)
        '''

        if isinstance(coherence, float) == False and callable(
                coherence) == False:
            raise TypeError("kwarg coherence must be callable or a float.")

        # set random number seed
        np.random.seed(random_seed)

        # initialise arrays and counters
        r = r0
        rcen = r0
        rcen_array, r_array = [], []
        deltaL_array = []

        # wonder if there's a better way to do this?
        while r < L:
            # get a coherence length which will be the size of the box
            # this can be a function or a float
            if callable(coherence):
                lc = coherence()
            else:
                lc = coherence

            if self.coherence_r0 is not None:
                lc *= (1.0 + (r / (self.coherence_r0)))

            # ensure the simulation is truncated at distance L
            if (r + lc) > L:
                lc = (L - r) + 1e-10

            # if rcen == r0:
            #	rcen += lc / 2.0
            # else:
            #	rcen += lc

            # rcen_array.append(rcen)
            r_array.append(r)
            deltaL_array.append(lc)

            r += lc
            rcen = r - (lc / 2.0)
            rcen_array.append(rcen)

        #  now we have box sizes and radii, get the field and density in each
        # box
        Ncells = len(r_array)
        self.r = np.array(r_array)
        #rcen_array = np.array(0.5 * (self.r[1:] + self.r[:-1]))
        self.rcen = np.array(rcen_array)
        self.deltaL = np.array(deltaL_array)

        # draw random isotropic angles and save phi
        theta, phi = random_angle(size=Ncells)
        #phi = phi

        # get density and magnetic field strength at centre of box
        if cell_centered:
            rprofile = self.rcen
        else:
            rprofile = self.r
        self.ne, Btot = self.profile(rprofile)
        self.cell_centered = cell_centered

        # get the x and y components and increment r
        #Bx_array.append(B * np.sin(theta2))
        #y_array.append(B * np.cos(theta2))
        self.Bx, self.By, self.Bz = self.get_B_comp_from_angles(
            Btot, theta, phi)
        #self.Bx = Btot * np.sin(theta) * np.cos(phi)
        #self.By = Btot * np.sin(theta) * np.sin(phi)
        self.theta = theta

        # note B is actually Bperp
        self.B = np.sqrt(self.Bx**2 + self.By ** 2)
        self.phi = np.arctan2(self.Bx, self.By)
        #self.phi = phi

        self.rm = self.get_rm()
        self.omega_p = omega_p(self.ne)
        #print (self.rm)

    def get_B_comp_from_angles(self, Btot, theta, phi):
        Bx = Btot * np.sin(theta) * np.cos(phi)
        By = Btot * np.sin(theta) * np.sin(phi)
        Bz = Btot * np.cos(theta)
        return (Bx, By, Bz)

    def resonance_prune(self, mass, threshold=0.1, refine=50, required_res=3):
        # first identify any close-to resonances
        delta = np.log10(self.omega_p) - np.log10(mass)
        select = (np.fabs(delta) < threshold)

        # copy the domain to a new class
        domain_to_return = CopyDomain(self)

        # if no close to resonances, nothing to be done
        # also don't worry about cases where the resonance happens off the end of the
        # array
        if (np.sum(select) == 0) or (
                np.argmin(np.fabs(delta)) == len(self.omega_p) - 1):
            return (domain_to_return)

        # find non zero parts of selection
        index = np.asarray(select).nonzero()[0]

        if len(index) > required_res:
            # multiple domains are close to resonance. This means
            # we must be resolving the resonance relatively well,
            # so we just discard the the two points that span the actual resonance
            # Just discard the closest two
            closest1 = np.where(delta > 0, delta, np.inf).argmin()
            closest2 = np.where(-delta > 0, -delta, np.inf).argmin()
            ind1 = np.min((closest1, closest2))
            ind2 = np.max((closest1, closest2))
            if (ind2 - ind1) != 1:
                print(
                    "Warning: resonance_prune: values close to resonance are not adjacent!")

            attrs_to_mod = [
                "ne",
                "Bx",
                "By",
                "Bz",
                "B",
                "phi",
                "deltaL",
                "r",
                "rcen",
                "Bz",
                "omega_p"]
            for a in attrs_to_mod:
                arr = getattr(domain_to_return, a)
                to_concat = (arr[:ind1], arr[ind2 + 1:])
                arr_new = np.concatenate(to_concat)
                setattr(domain_to_return, a, arr_new)

            #self.rm = self.get_rm()

        # there are only a few domains close to resonance point, so we need to
        # resample
        elif len(index) <= required_res:
            # find the point either side of the resonance and find the first
            # one
            closest1 = np.where(delta > 0, delta, np.inf).argmin()
            closest2 = np.where(-delta > 0, -delta, np.inf).argmin()
            ind = np.min((closest1, closest2))
            #print (closest1, closest2)

            # new r array
            r_insert = np.linspace(self.r[ind], self.r[ind + 1], refine + 1)
            rcen_insert = 0.5 * (r_insert[1:] + r_insert[:-1])

            # new r and rcen arrays
            r = np.concatenate((self.r[:ind], r_insert[:-1], self.r[ind + 1:]))
            rcen = np.concatenate(
                (self.rcen[:ind], rcen_insert, self.rcen[ind + 1:]))

            deltaL_insert = r_insert[1:] - r_insert[:-1]
            deltaL = np.concatenate(
                (self.deltaL[:ind], deltaL_insert, self.deltaL[ind + 1:]))

            # get the density
            if self.cell_centered:
                rprofile = rcen
            else:
                rprofile = r
            ne, _ = self.profile(rprofile)
            w_p = omega_p(ne)
            new_delta = np.log10(w_p) - np.log10(mass)

            # find closest two arguments
            closest1 = np.where(new_delta > 0, new_delta, np.inf).argmin()
            closest2 = np.where(-new_delta > 0, -new_delta, np.inf).argmin()
            ind_new = np.min((closest1, closest2))

            if ind_new == (ind + len(rcen_insert) - 1):
                ndiscard = 1
            else:
                ndiscard = 2

            domain_to_return.r = np.concatenate(
                (r[:ind_new], r[ind_new + ndiscard:]))
            domain_to_return.rcen = np.concatenate(
                (rcen[:ind_new], rcen[ind_new + ndiscard:]))
            domain_to_return.ne = np.concatenate(
                (ne[:ind_new], ne[ind_new + ndiscard:]))
            domain_to_return.omega_p = omega_p(domain_to_return.ne)
            domain_to_return.deltaL = np.concatenate(
                (deltaL[:ind_new], deltaL[ind_new + ndiscard:]))
            N = len(domain_to_return.r)
            assert (N == (len(self.r) + refine - ndiscard - 1))

            # things that remain constant across resampling
            attrs_const = ["Bx", "By", "Bz", "B", "phi"]
            #list_const = [domain_to_return.Bx, self.By, self.B, self.phi, self.Bz]
            for a in attrs_const:
                arr = getattr(domain_to_return, a)
                arr_insert = np.ones(len(rcen_insert) - 1) * arr[ind]
                to_concat = (arr[:ind], arr_insert, arr[ind + ndiscard:])
                arr = np.concatenate(to_concat)
                setattr(domain_to_return, a, arr)
                assert (len(getattr(domain_to_return, a)) == N,
                        "incorrect array lengths after pruning -- could mean domain was modified pre-pruning")

        #domain_to_return = DomainTemp(deltaL, B, phi, ne, len(index))

        return (domain_to_return)

    def concat(self, index1, index2, insert_array=None):
        '''
        cut out the middle part of an array, between index1 and index2,
        and stick the two ends back together again. Used to excise problematic
        portions of a domain.


        Parameters:
                index1 		int
                                        the starting point of the excision
                index2    	int
                                        the ending point of the excision.
        '''

        arrays_to_splice = [
            self.r,
            self.rcen,
            self.Bx,
            self.By,
            self.B,
            self.omega_p,
            self.phi,
            self.ne,
            self.Bz]
        for arr in arrays_to_splice:
            arr = np.concatenate((arr[:index1], arr[index2:]))

        return len(self.r)


# this class copies over a different class to a new one without
# trying to write non-writable attributes and without altering the original
# class
class CopyDomain:
    def __init__(self, input_domain):
        attrs_to_copy = [f for f in dir(input_domain) if "__" not in f]
        for a in attrs_to_copy:
            #print (a)
            value = getattr(input_domain, a)
            setattr(self, a, value)
