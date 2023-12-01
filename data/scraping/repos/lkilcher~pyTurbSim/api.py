"""
This is the PyTurbSim python advanced programming interface
(API). This module provides a fully-customizable high-level
object-oriented interface to the PyTurbSim program.

The four components of this API are:

1) The :class:`tsrun <pyts.main.tsrun>` class, which is the
controller/run object for PyTurbSim simulations.

2) The :class:`tsGrid <pyts.base.tsGrid>` class, which is used to
define the TurbSim grid.

3) The 'model' classes, which include:

   a) :mod:`profModels <pyts.profModels>` (aliased here as 'pm'), contains the mean velocity
   profile models.

   b) :mod:`.specModels` (aliased here as 'sm'), contains the TKE spectral
   models.

   c) :mod:`.stressModels` (aliased here as 'rm'), contains the Reynold's
   stress profile models.

   d) :mod:`.cohereModels` (aliased here as 'cm'), contains the spatial
   coherence models.

4) The :mod:`io` module, which supports reading and writing of TurbSim
input (.inp) and output files (e.g. .bl, .wnd, etc.)

Example usage of this API can be found in the <pyts_root>/Examples/api.py file.

.. literalinclude:: ../../../examples/api.py


"""
from main import tsrun
from base import tsGrid
import profModels.api as profModels
import specModels.api as specModels
import cohereModels.api as cohereModels
import stressModels.api as stressModels
import io

# Set aliases to the model modules:
pm=profModels
sm=specModels
cm=cohereModels
rm=stressModels
