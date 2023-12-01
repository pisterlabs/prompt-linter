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
RandomGuidance reads the following parameter values:

- randomseed (any hashable object, default: None)

  seed for random number generator

"""

from tema.guidance.guidance import Guidance as GuidanceBase
import random
import time # for random seed initialization


version='0.1 random walk'

class Guidance(GuidanceBase):

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
        return self._rndchoose(state_object.getOutTransitions()).getAction()
