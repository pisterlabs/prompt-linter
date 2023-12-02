#!/usr/bin/env python
# coding: iso-8859-1
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
guiguidance let's the user decide (via ModelGui) what to execute next.
"""

from tema.guidance.guidance import Guidance as GuidanceBase
from tema.validator.simulation.modelgui import getTheModelGui

class Guidance(GuidanceBase):
    def __init__(self):
        GuidanceBase.__init__(self)

    def setParameter(self,paramname,paramvalue):
        print __doc__
        raise Exception("Invalid parameter '%s' for guiguidance." % paramname)
#        GuidanceBase.setParameter(self,paramname,paramvalue)

    def prepareForRun(self):
        self._path = []

    def markExecuted(self, transition):
        self._path = self._path[1:]
        GuidanceBase.markExecuted(self,transition)

    def suggestAction(self, from_state):
        if not self._path or not self._path[0].getSourceState()==from_state:
            # Ask the gui (=user) for a path
            self._path = getTheModelGui().selectPath(from_state)
        else:
            # we still have path left, just drawing the current position
            getTheModelGui().stepTo(from_state)
        return self._path[0].getAction()

    def isThreadable(self):
        # Afaik, Tkinter gui doesn't tolerate calls from threads
        # other than the one that created it. That's why we can't use
        # modelgui from another thread.
        return False



