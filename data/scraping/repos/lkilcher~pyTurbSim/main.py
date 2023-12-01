"""
This module defines two coherence models:
nwtc - The NWTC 'non-IEC' coherence model.
iec  - The IEC coherence model.
"""
from .base import cohereModelBase, np, ts_float, cohereObj, ts_complex
from ..base import tslib, dbg
from ..misc import Lambda


class cohereObjNone(cohereObj):

    """
    This is a 'dummy' coherence object that forces the coherence to
    zero.
    """

    def calc_phases(self, phases):
        return phases

    def calcCoh(self, f, comp, ii, jj):
        return 0 * f


class none(cohereModelBase):

    """
    This is a 'dummy' coherence model that forces the coherence to
    zero.
    """
    cohereObj = cohereObjNone


class cohereObjNWTC(cohereObj):

    def calc_phases(self, phases):
        """
        Compute the `correlated phases` for each grid-point from the
        input `phases` based on the coherence NWTC 'non-IEC coherence
        model.

        Parameters
        ----------
        phases : array_like(np, nf)
                 The input (generally randomized) phases for each
                 point for each frequency.

        Returns
        -------
        phases : array_like(np, nf)
                 The correlated phases according to this cohereObj's
                 coherence model (see :meth:`calcCoh`).

        Notes
        -----

        This method should not be called explicitly.  It is called by
        a cohereObj instance's __call__ method.

        See also
        --------
        calcCoh : computes the coherence for individual grid-point pairs.

        """
        if tslib is not None:
            tmp = np.zeros((self.n_p, self.n_f), dtype=ts_complex, order='F')
            u = self.grid.flatten(self.prof.u).copy(order='F')
            for icomp in range(3):
                tmp[:] = phases[icomp]
                tslib.nonieccoh(tmp, self.grid.f,
                                self.grid.y, self.grid.z, u,
                                self.a[icomp], self.b[icomp], self.CohExp,
                                self.ncore, self.n_f, self.n_y, self.n_z)
                phases[icomp] = tmp
        else:
            phases = cohereObj.calc_phases(self, phases)
        return phases

    def calcCoh(self, f, comp, ii, jj):
        """
        The base function for calculating coherence for non-IEC
        spectral models.

        See the TurbSim documentation for further information.

        This function is only used if the TSlib fortran library is not
        available.
        """
        ii = self.grid.ind2sub(ii)
        jj = self.grid.ind2sub(jj)
        r = self.grid.dist(ii, jj)
        two = ts_float(2)
        zm = (self.grid.z[ii[0]] + self.grid.z[jj[0]]) / two
        um = (self.prof.u[ii] + self.prof.u[jj]) / two
        return np.exp(-self.a[comp] * (r / zm) ** self.CohExp *
                      np.sqrt((f * r / um) ** two + (self.b[comp] * r) ** two))


class nwtc(cohereModelBase):

    """NWTC coherence model.

    This is also known as the 'non-IEC' coherence model.

    Parameters
    ----------
    a : array_like(3)
        The 'a' exponential 'coherence decrement' coefficients of the
        coherence function for each velocity component.
    b : a
        The 'b' exponential 'coherence decrement' coefficients of the
        coherence function for each velocity component.
    CohExp : float
             The 'Coherence Exponent' parameter for the coherence function.

    Notes
    -----
    The NWTC 'non-IEC' coherence model for velocity component 'k' between two
    points (z_i,y_i) and (z_j,y_j) is

    .. math::

       Coh_k=exp(-a_k (r/z_m)^CohExp ((f r / u_m)^2 + (b_k r)^2)  k=u,v,w

    Where,

    f is frequency.

    r is the distance between the two points.

    a_k and b_k are 'coherence decrement' input parameters for each of
    the velocity components.

    u_m is the average velocity of the two points.

    z_m is the average height of the two points.

    CohExp is the 'coherence exponent' input parameter (default is 0).

    """
    __doc__ += cohereModelBase.__doc__

    cohereObj = cohereObjNWTC  # This must be defined for each coherence model.

    def __init__(self, a=[None, None, None], b=[0., 0., 0.], CohExp=0.0):
        """
        Create a NWTC 'non-IEC' coherence model.

        Parameters
        ----------
        a : 3-element array-like of floats (or None for default), optional.
            The first coherence decrement input parameter for each of the
            three velocity components. If an element is None, a default value
            is used. The defaults are:
                 a[0] = uhub
                 a[1] = 0.75*a[0]
                 a[2] = 0.75*a[0]
        b : 3-element array-like of floats, optional.
            The second coherence decrement input parameter for each
            velocity component.  Each element defaults to 0.
        CohExp :  float, optional
        """
        if CohExp is None:
            self.CohExp = 0.0
        else:
            self.CohExp = CohExp
        self.a = a
        self.b = np.array(b, dtype=ts_float)
        if dbg:
            #self.timer=dbg.timer('tslib-cohNWTC')
            self.timer = dbg.timer('roll')

    def _sumfile_string(self, tsrun, ):
        sumstring_format = """
        Coherence model used                             =  {dat.model_desc}
        Coherence Exponent                               =  {dat.CohExp:0.2f}
        Coherence decrements (IncDec):
           U (a, b)                                      =  ({coh.a[0]:0.2f}, {coh.b[0]:0.2f})  # noqa
           V (a, b)                                      =  ({coh.a[1]:0.2f}, {coh.b[1]:0.2f})  # noqa
           W (a, b)                                      =  ({coh.a[2]:0.2f}, {coh.b[2]:0.2f})  # noqa
        """
        return sumstring_format.format(dat=self, coh=tsrun.cohere, )

    def set_coefs(self, cohereObj):
        """
        This method is called by the coherence model __call__ method
        just before returning the coherence instance *cohi*.
        """
        cohereObj.CohExp = self.CohExp
        cohereObj.b = self.b
        cohereObj.a = np.empty((3), dtype=ts_float)
        if self.a[0] is None:
            cohereObj.a[0] = cohereObj.prof.uhub
        else:
            cohereObj.a[0] = self.a[0]
        if self.a[1] is None:
            cohereObj.a[1] = 0.75 * cohereObj.a[0]
        else:
            cohereObj.a[1] = self.a[1]
        if self.a[2] is None:
            cohereObj.a[2] = 0.75 * cohereObj.a[0]
        else:
            cohereObj.a[2] = self.a[2]


class cohereObjIEC(cohereObj):

    def calcCoh(self, f, comp, ii, jj):
        """
        Calculate the coherence for a velocity component, between two points.

        Parameters
        ----------
        f : array_like(nf,dtype=float)

        comp : int {0,1,2}
               indicating the velocity component for which to compute
               the coherence.

        ii,jj : array_like(int,2)
                indicating the grid-points between which to calculate
                the coherence.  for example: ii=(1,3),jj=(2,3)

        Returns
        -------
        coh : array_like(nf,dtype=float)
              The coherence between the two points as a function of frequency.

        Notes
        -----

        This method is only used if tslib is not available.


        """
        if comp == 0:
            r = self.grid.dist(ii, jj)
            return np.exp(-self.a * np.sqrt((f * r / self.prof.uhub) ** 2 +
                                            (0.12 * r / self.Lc) ** 2))
        else:
            return 0

    def calc_phases(self, phases):
        """
        Compute and set the full cross-coherence matrix for component
        *comp* for 'coherence calculator' instance *cohi*.

        Compute the coherence array for coherence instance cohi for
        the IEC model.

        """
        if tslib is not None:
            out = phases[0].copy(order='F')
            tslib.ieccoh(out, self.grid.f,
                         self.y, self.z, self.prof.uhub,
                         self.a, self.Lc,
                         self.ncore, self.n_f, self.n_y, self.n_z)
            phases[0] = out
            return phases
        else:
            return cohereObj.calc_phases(self, phases)


class iec(cohereModelBase):

    """IEC coherence model.

    This coherence model only include u-component coherence.

    Parameters
    ----------
    IECedition : int {2, 3},
                 Different IEC editions have slightly different
                 coefficients to the spectral model.

    Notes
    -----
    The IEC coherence is zero for the v- and w-components (coherence matrix
    is the identity matrix so that auto-spectra are retained).

    The form of this model is for the u-component is,

    .. math::

       Coh=exp(-a*((f*r/uhub)^2+(0.12*r/Lc)^2)^0.5)

    Where,

    f is frequency.

    r is the distance between the two points.

    uhub is the hub-height mean velocity.

    If IECedition<=2:
    a  = 8.8,
    Lc = 2.45*min(30m,HubHt)

    If IECedition>=3:
    a  = 12,
    Lc = 5.67*min(60m,HubHt)

    """
    __doc__ += cohereModelBase.__doc__

    cohereObj = cohereObjIEC

    def _L(self, zhub):
        """
        Calculate the 'coherence scale parameter' for hub-height *zhub*.
        """
        return self._Lfactor * Lambda(zhub, self.IECedition)

    def __init__(self, IECedition=3, ):
        self.IECedition = IECedition
        if IECedition <= 2:
            # The Lambda function includes a factor of 0.7 (_Lfactor*0.7=2.45).
            self._Lfactor = 3.5
            self.a = 8.8
        else:  # 3rd edition IEC standard:
            # The Lambda function includes a factor of 0.7 (_Lfactor*0.7=5.67)
            self._Lfactor = 8.1
            self.a = 12.
        if dbg:
            self.timer = dbg.timer('tslib-cohIEC')

    def _sumfile_string(self, tsrun, ):
        sumstring_format = """
        Coherence model used                             =  {dat.model_desc}
        IEC Edition                                      =  {dat.IECedition:d}
        Coherence length scale parameter                 =  {L:0.2f} [m]
        """
        return sumstring_format.format(dat=self, L=self._L(tsrun.grid.zhub))

    def set_coefs(self, cohereObj):
        """
        Initialize a coherence instance for the IEC coherence model.

        """
        cohereObj.Lc = self._L(cohereObj.grid.zhub)
        cohereObj.a = self.a
