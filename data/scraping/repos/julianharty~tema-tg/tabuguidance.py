# -*- coding: utf-8 -*-
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
A guidance that tries to find actions/states/transitions that are not in
a tabulist.

Params:
    'numtabuactions' (nonnegative integer of "infinite"):
        Size of action tabu list.
    'numtabustates' (nonnegative integer of "infinite"):
        Size of state tabu list.
    'numtabutransitions' (nonnegative integer of "infinite):
        Size of transition tabu list.

If none of these are given, using infinite state tabulist.

If more than one numtabu params given, tabuguidance'll search the tabulists
in this order: action tabulist -> state tabulist -> transition tabulist.
When an outgoing transition whose action/state/transition is not in the
tabulist is found, the corresponding action is suggested.
If no such transition is found after searching through all the tabulists,
a random action is suggested.

If there are many possible actions to execute, one of them is chosen randomly.
"""

version="tabuguidance 0.21"

from tema.guidance.guidance import Guidance as GuidanceBase
from tema.coverage.tabulist import TabuList
import random

INFINITY = () # () is a good choice for INFINITY since () > anything...


class Guidance(GuidanceBase):
    def __init__(self):
        GuidanceBase.__init__(self)
        self._NUM_TABU_ACTIONS = None
        self._NUM_TABU_STATES = None
        self._NUM_TABU_TRANSITIONS = None
        self._tabulist_action = None
        self._tabulist_state = None
        self._tabulist_transition = None

    def setParameter(self,paramname,paramvalue):
        accepted = ("numtabuactions","numtabustates","numtabutransitions")
        if paramname=='numtabuactions':
            self._NUM_TABU_ACTIONS = self._parseSize(paramname,paramvalue)
        elif paramname=='numtabustates':
            self._NUM_TABU_STATES = self._parseSize(paramname,paramvalue)
        elif paramname=='numtabutransitions':
            self._NUM_TABU_TRANSITIONS = self._parseSize(paramname,paramvalue)
        else:
            print __doc__
            raise Exception("Invalid parameter '%s' for tabuguidance. Accepted parameters: %s" % paramname, accepted)
        GuidanceBase.setParameter(self,paramname,paramvalue)

    def _parseSize(self,paramname,paramvalue):
        if paramvalue == float("infinity") or paramvalue is INFINITY or \
           paramvalue in ("inf","infinite","infinity"):
            return INFINITY
        try:
            return int(paramvalue)
        except ValueError:
            raise Exception("Tabuguidance: invalid '%s' value: %s. It should be a positive integer or 'infinite'." % (paramname,paramvalue))

    def prepareForRun(self):

        # if no numtabu* params given, use infinite state tabulist
        if (self._NUM_TABU_ACTIONS is None and
            self._NUM_TABU_STATES is None and
            self._NUM_TABU_TRANSITIONS is None):
            self.log("Using default: 'numtabustates:infinite'")
            self._NUM_TABU_STATES = INFINITY

        self._suggesters = [] # the funcs that suggest an action

        # order: action, state, transition
        if self._NUM_TABU_ACTIONS is not None:
            self._tabulist_action = TabuList(self._NUM_TABU_ACTIONS)
            self._suggesters.append(self._newAction)
        if self._NUM_TABU_STATES is not None:
            self._tabulist_state = TabuList(self._NUM_TABU_STATES)
            self._suggesters.append(self._newStateAction)
        if self._NUM_TABU_TRANSITIONS is not None:
            self._tabulist_transition = TabuList(self._NUM_TABU_TRANSITIONS)
            self._suggesters.append(self._newTransitionAction)

    def markExecuted(self, transition):

        # special case: add the very first (source) state to the tabu-list
        statelist = self._tabulist_state
        if statelist and len(statelist) == 0:
            statelist.add( transition.getSourceState() )

        # add actions/states/transitions to tabulists if given tabulist exists
        if self._tabulist_action is not None:
            self._tabulist_action.add( str(transition.getAction()) )
        if self._tabulist_state is not None:
            self._tabulist_state.add( str(transition.getDestState()) )
        if self._tabulist_transition is not None:
            self._tabulist_transition.add( str(transition) )

        GuidanceBase.markExecuted(self,transition)

    def suggestAction(self, from_state):

        out_trans = from_state.getOutTransitions()
        random.shuffle(out_trans)

        for suggester in self._suggesters:
            action = suggester(out_trans)
            if action is not None:
                return action

        # no non-tabu actions found, a random action is our best suggestion...
        return out_trans[0].getAction() # out_trans has been shuffled

    def _newAction(self, trans):
        """returns a non-tabu action, or None"""
        for t in trans:
            if str(t.getAction()) not in self._tabulist_action:
                return t.getAction()
        return None

    def _newStateAction(self, trans):
        """returns an action leading to a non-tabu state, or None"""
        for t in trans:
            if str(t.getDestState()) not in self._tabulist_state:
                return t.getAction()
        return None

    def _newTransitionAction(self, trans):
        """returns an action of a non-tabu transition, or None"""
        for t in trans:
            if str(t) not in self._tabulist_transition:
                return t.getAction()
        return None




