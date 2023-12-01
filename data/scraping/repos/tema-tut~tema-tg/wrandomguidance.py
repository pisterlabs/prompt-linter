# Copyright (c) 2006-2010 Tampere University of Technology
# 
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
# 
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
WGuidance (Weighted Random Test Selection) reads the following parameter
values:

- randomseed (any hashable object, default: None)

  seed for random number generator

WRandom requires that Transition objects of the model implement
getProbability() method.

"""

from tema.guidance.guidance import Guidance as GuidanceBase
import random
import time

# The sum of probabilities of transitions must not differ from 1 more
# than FLOAT_ERR, otherwise a warning is printed to log.  For those
# who did not know, (0.3 + 0.3 + 0.3 + 0.1) != 1.
FLOAT_ERR=.0000001

version='0.1'

class GuidanceException(Exception): pass

class WGuidance(GuidanceBase):

    def __init__(self):
        GuidanceBase.__init__(self)
        self.setParameter('randomseed',time.time())

    def setParameter(self,parametername,parametervalue):
        if not parametername in ['randomseed']:
            print __doc__
            raise Exception("Invalid parameter '%s' for gameguidance." % parametername)
        GuidanceBase.setParameter(self,parametername,parametervalue)
        if parametername=='randomseed':
            self._rndchoose=random.Random(parametervalue).choice

    def suggestAction(self,state_object):
        # check that getProbability is implemented
        try:
            probabilities = [t.getProbability() for t in state_object.getOutTransitions()]
        except AttributeError:
            self.log("getProbability not implemented in a transition from state %s"
                     % state_object)
            raise GuidanceException("getProbability not implemented")
        maxvalue=sum(probabilities)
        if not (1-FLOAT_ERR < maxvalue < 1+FLOAT_ERR):
            self.log("Warning: weights not normalized to 1 (sum = %s) in state %s" % (maxvalue,state_object))
        r=random.random()*maxvalue
        integral=0
        for i,p in enumerate(probabilities):
            integral+=p
            if r<=integral: return state_object.getOutTransitions()[i].getAction()
        # this line should never be reached
        raise GuidanceException("Failed to pick an action. Is this a deadlock: %s?"
                                % state_object)

Guidance=WGuidance
