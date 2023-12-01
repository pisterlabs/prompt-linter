#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                                 Jiao Lin   
#                      California Institute of Technology
#                      (C)   2007    All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


from .KernelNode import KernelNode as base, debug


class Phonon_CoherentInelastic_PolyXtal_Kernel(base):


    tag = "Phonon_CoherentInelastic_PolyXtal_Kernel"
    

    def createKernel( self, **kwds ):
        max_omega = self._parse( kwds['max-omega'] )
        from mccomponents.sample.phonon \
             import coherentinelastic_polyxtal_kernel as f
        return f(None, max_omega = max_omega)
    
    
    def onDispersion(self, dispersion):
        self.element.dispersion = dispersion
        return
    
    
    onPeriodicDispersion = onDispersion
    
    pass # end of Phonon_CoherentInelastic_PolyXtal_Kernel


from .HomogeneousScatterer import HomogeneousScatterer
HomogeneousScatterer.onPhonon_CoherentInelastic_PolyXtal_Kernel = HomogeneousScatterer.onKernel


# version
__id__ = "$Id$"

# End of file 
