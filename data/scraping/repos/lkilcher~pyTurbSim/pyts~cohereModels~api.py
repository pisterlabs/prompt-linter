r"""
A central purpose of TurbSim is to produce time-series of velocity
that are correlated spatially and in a way that is consistent with
observations.  In particular, TurbSim produces time-series with
specific frequency-dependent correlations between all pairs of points;
this is the 'coherence'.

Loosely defined, coherence is the frequency dependence of the
correlation between velocity signals at different points in space.
Specically, for two signals `u1` and `u2` (e.g. velocity signals at points
`z1,y1` and `z2,y2`), the coherence is defined as:

.. math::
  Coh_{u1,u2} = \frac{|C_{u1,u2}|^2}{S_{u1}S_{u2}}

Where :math:`C_{u1,u2}` is the cross-spectral density between signals u1 and
u2, and :math:`S_{u1}` is the auto-spectral density of signal u1 (similar for
:math:`S_{u2}`).

This module provides 'coherence models' which are specific spatial
coherence forms (functions of frequency) for each component of
velocity for all pairs of points in the grid.

Available coherence models
--------------------------

:class:`.main.iec` (alias: iec)
  The IEC spatial coherence model

:class:`.main.nwtc` (alias: nwtc)
  The NWTC 'non-IEC' coherence model.

:class:`.main.none` (alias: none)
  A 'no coherence' model.

:class:`.cohereModelBase`
  This is the base class for coherence models. To create a new one,
  subclass this class or subclass and modify an existing coherence
  model.

:class:`~.base.cohereObj`
  This is the 'coherence object' class.  All coherence model `__call__`
  methods must take a :class:`tsrun <pyts.main.tsrun>` as input and
  return this class.

Further details on creating your own coherence model, can be found in
:mod:`pyts.cohereModels.base` documentation.

"""
from .base import cohereObj, cohereModelBase
import main

iec = main.iec
nwtc = main.nwtc
none = main.none
