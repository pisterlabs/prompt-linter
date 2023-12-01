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
Greedy guidance is a breadth first searching algorithm that
returns the shortest path improving coverage.  If one of the search
limits is reached, a random path is selected.

Greedy guidance reads the following parameter values:

- max_states (positive integer, default: 10000)

  The number of states the breath search algorithm expands in a single
  search round.

- max_second (positive value, default: 3600)

  The maximum amount of time in seconds a single search can last.
"""

version='wormguidance based on greedyguidance: 0.beta'

from tema.guidance.guidance import Guidance as GuidanceBase
from tema.model.model import Transition
import random
import time
import re

GoodState, OnlyJump, UglyState, SelectJorS = range(4)

class StopCondition:
    def __init__(self, prm_src, sized_dict, start_time):
        self._dictionary = sized_dict
        self._start_time = start_time
        self._max_states = prm_src.getParameter("max_states")
        self._time_limit = prm_src.getParameter("max_seconds",3600)

    def __call__(self):
        rval = (time.time()-self._start_time) >= self._time_limit
        if self._max_states :
            rval = rval or (len(self._dictionary) >= self._max_states)
        return rval
        

class Guidance(GuidanceBase):
    def __init__(self):
        GuidanceBase.__init__(self)
        self._stored_path=[]
        self._random_select=random.Random(time.time()).choice
        self._sleep_ts_re = re.compile(r"SLEEPts.*")
        
    def _search_transition_by_name(self, from_state, a_name):
        for trs in from_state.getOutTransitions() :
            if str( trs.getAction()) == a_name :
                return trs
        return None

    def _get_select_set(self, state, closed):
        rval=[]
        for trs in state.getOutTransitions():
            if str(trs.getDestState()) not in closed:
                rval.append(trs)
        return rval

    def _construct_path_to(self, transition, closed):
        rval=[transition]
        s=rval[0].getSourceState()
        while s :
            rval[0:0]=[closed[str(s)]]
            s=rval[0].getSourceState()
        return rval[1:]

    def _breadth_first_search(self, from_state, target_actions):
        self.setParameter("max_states",self.getParameter("max_states",10000))
        closed={}
        waiting=[Transition(None,None,from_state)]
        stop_condition=StopCondition(self,closed,self._start_time)
        
        while waiting and not stop_condition() :
            current_trans = waiting.pop(0)
            current_state = current_trans.getDestState()
            if  not closed.has_key(str(current_state)) :
                closed[str(current_state)] = current_trans
                for trs in current_state.getOutTransitions():
                    if str(trs.getAction()) in target_actions :
                        self._forbiden_set=set()
                        return (self._construct_path_to(trs, closed), True)
                    elif str(trs.getDestState()) in self._forbiden_set:
                        pass
                    elif closed.has_key(str(trs.getDestState())) :
                        pass
                    else:
                        waiting.append(trs)

        if waiting :
            trs=self._random_select(waiting)
            #self._forbiden_set = self._forbiden_set | set(closed.keys())
            self._forbiden_set = set(closed.keys())
            self.log("Forbiden set: %s" % len(self._forbiden_set))
            return (self._construct_path_to(trs, closed), False)

        self._forbiden_set=set()
        return (None, False)

    def _search_engine(self, from_state, target_actions):
        
        self._stored_path, success = self._breadth_first_search (from_state,\
                                                        target_actions)
        if success :
            self._search_state = GoodState
        elif ( self._search_state == UglyState and random.random() < 0.25) \
          or not self._stored_path :
            back_path, success = self._breadth_first_search (from_state,\
                                          self._to_sleep_actions)
            if success :
                self._stored_path = back_path
                self._search_state = GoodState
                self.log("Moves backwards")
        else :
            self._search_state = UglyState
        if self._search_state == UglyState :
            self.log("Jumps randomly forward")

    def prepareForRun(self):
        nonexit="Nonexisting string"
        if self.getParameter("help", nonexit) != nonexit:
            print __doc__
            raise Exception("Asked only for help")
        GuidanceBase.prepareForRun(self)
        if len(self._requirements) != 1 :
            raise Exception("Needs exactly one requirement")
        if not self._testmodel :
            raise Exception("Model should be given")
        self._stored_path=[]
        self._to_sleep_actions =\
                   self._testmodel.matchedActions(set([self._sleep_ts_re]))
        self._last_go_back = False
        self._search_state = GoodState
        self._forbiden_set = set()
        self.log("Wormguidance ready for rocking")

    def _trslist_to_str(self,path):
        return str([ str(t.getAction()) for t in path])

    def suggestAction(self, from_state):
        # self.log("DEBUG: new search beginning")
        self._start_time=time.time()
        if self._stored_path :
            if str(self._stored_path[0].getSourceState()) != str(from_state) :
                self.log("Throw away: %s"\
                      % self._trslist_to_str(self._stored_path) )
                self._stored_path=[]
                self._forbiden_set=set()
                
        # self.log("DEBUG: Ok, käynnistellään etsintää")

        if not self._stored_path :
            cov_obj=self._requirements[0]
            test_model=self._testmodel
            # self.log("DEBUG: about to hint")            
            rex, d = cov_obj.getExecutionHint()
            # self.log("DEBUG: about to degrypt")            
            actions = test_model.matchedActions(rex)
            # self.log("DEBUG: tapahtumanimet  "+str(actions))
            if len(actions) > 0 :
                self._search_engine(from_state, actions)
                test_model.clearCache()
                self.log("Path: %s"\
                      % self._trslist_to_str(self._stored_path) )

        if self._stored_path :
            trs = self._stored_path.pop(0)
            self.log("Search has been ended")
            return trs.getAction()
        else:
            raise Exception ("Next action can not be found")
