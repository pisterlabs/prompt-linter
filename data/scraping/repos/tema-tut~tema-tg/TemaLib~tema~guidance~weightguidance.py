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
TODO

Parameters:
    'searchdepth':
        The maximum length of a path to search.
        default: 10

    'searchorder':
        'shortestfirst': Always search the shortest unfinished path next.
        'bestfirst': Always search the best unfinished path next.
        default: 'shortestfirst'

    'maxtransitions':
        The maximum number of transitions to go through in the search.
        default: 10000

    'greedy':
        0: Search until it's time to stop. Then select the best path.
        1: Choose the very first coverage-improving path found.
        default: 0

    'searchconstraint':

        searchconstraint sets some limits on to which paths will be searched.

             ^
            B|
             o<---,
             ^    |E
            A| C  |
          -->o--->o
             ^    |
            F|    |D
             o<---'

        'nocrossingpaths':
            Breadth-first kind of search. If a transition has already been
            searched, it won't be searched again as a part of any other path.
            Eg. if a path A-B has already been searched, C-E-B won't be searched

        'noloops':
            A transition can appear only once in a path.
            Eg. C-D-F-C won't not searched (C-D-F will be though).

        'noconstraint':
            No constraints, all the paths can be searched.
            Eg. C-D-F-C-D-F-C-E is possible

        default: 'noloops'


    'transitionweight':
        TODO




"""

import random
from heapq import heappush, heappop

from tema.guidance.guidance import Guidance as GuidanceBase

version = '0.1'

SEARCH_CONSTRAINTS = (NONE,NO_LOOPS,NO_CROSSING_PATHS) = range(3)

class Guidance(GuidanceBase):
    def __init__(self):
        GuidanceBase.__init__(self)
        # default parameters:
        self.setParameter("transitionweight",0)
        self.setParameter("searchdepth",10)
        self.setParameter("searchorder","shortestfirst")
        self.setParameter("maxtransitions",10000)
        self.setParameter("greedy",0)
        self.setParameter("searchconstraint","noloops")

    def setParameter(self,name,value):
        accepted = ("transitionweight","searchorder","searchdepth",
                    "maxtransitions","searchconstraint")
        if name == "transitionweight":
            if isinstance(value,str) and value.startswith('kw:'):
                kww = float(value[3:])
                self._transitionweight = \
                    lambda t: kww if 'kw_' in str(t.getAction()) else 0
            else:
                if value < 0:
                    self.log("WARNING! Negative transition weight "+
                             "doesn't make sense!")
                self._transitionweight = lambda t: value
        elif name == "searchorder":
            if value == "bestfirst":
                self._toHeap = lambda p,badness: (badness,len(p),p)
                self._fromHeap = lambda values: (values[2],values[0])
            elif value == "shortestfirst":
                self._toHeap = lambda p,badness: (len(p),badness,p)
                self._fromHeap = lambda values: (values[2],values[1])
            else:
                raise ValueError("Invalid searchorder: '%s'" % (value,))
        elif name in ("searchdepth", "searchradius"):
            self._searchDepth = value
        elif name == "maxtransitions":
            self._maxTransitions = value
        elif name == "greedy":
            self._greedy = value
        elif name == "searchconstraint":
            if value == "nocrossingpaths":
                self._seco = NO_CROSSING_PATHS
            elif value == "noloops":
                self._seco = NO_LOOPS
            elif value == "noconstraint":
                self._seco = NONE
            else:
                raise ValueError("Invalid searchconstraint '%s'"%value)
        else:
            print __doc__
            raise ValueError("Invalid parameter '%s' for newguidance. "%name +
                             "Accepted parameters: %s" % ",".join(accepted))
        GuidanceBase.setParameter(self,name,value)

    def _kwWeight(self,transition):
        if "kw_" in str(transition.getAction()):
            return 1
        return 0

    def prepareForRun(self):
        self._thePlan = []

    def suggestAction(self, fromState):
        if not self._thePlan:
            self.log("Computing a new path...")
            # reverse the path so we can pop() the next transition...
            self._thePlan = [t for t in reversed(self._search(fromState))]
            self._testmodel.clearCache()

        nextTrans = self._thePlan.pop()

        # != operator not defined for States!
        if not nextTrans.getSourceState() == fromState:
            # we ended up in a state that wasn't in _thePlan.
            # usually (always?) this happens when we suggested a path with
            # action A but ~A was actually executed, or vice versa.
            # TODO: something to deal with this "nondetermism" in the search,
            # or no?
            self.log("We've fallen off the path I once suggested! "+\
                     "I'll suggest a new path.")
            self._thePlan = []
            return self.suggestAction(fromState)
        return nextTrans.getAction()

    def _search(self,fromState):
        """ Searches from the given state until:
            - all the paths with length 'searchdepth' have been searched
            - OR 'maxtransitions' transitions seen
            - OR 'greedy' is enabled and any path improving coverage is found.

            Returns the best path found.

            Goodness of a path =
                covreq.transitionPoints(t) - _transitionweight(t)
            for each transition t in the path.

            If not transitionPoints defined in covreq, then goodness =
            covreq.getPercentage() difference between the end and the beginning.

            'searchorder'=='shortestfirst':
                always check the shortest unfinished path next
            'searchorder'=='bestfirst':
                always check the best unfinished path found so far
        """

        if len(self._requirements) > 1:
            raise NotImplementedError("Only one requirement, please.")

        req = self._requirements[0]
        startCov = req.getPercentage()

        # If the req has transitionPoints method, we'll use that.
        # Otherwise, using getPercentage()
        useTP = hasattr(req,"transitionPoints")

        # pathHeap contains the paths whose search is in progress.
        # the goodness of the last transition of each of the paths has not been
        # determined yet.

        startingTrans = [t for t in fromState.getOutTransitions()]
        pathHeap = [self._toHeap((t,),0) for t in startingTrans]
        seenTrans = set(startingTrans)

        # because heapq is smallest-first, measuring the badness instead of
        # goodness of path...
        bestPaths = []
        leastBadness = 0

        SEARCH_TRANSITIONS = self._maxTransitions
        MAX_LENGTH = self._searchDepth

        # the paths whose length is max. their search has been thus stopped.
        maxLenPaths = []
        # the paths that can't be continued, even though their length < max.
        deadEnds = []

        transitionsSearched = 0

        while True: # searching until there's some reason to stop (break).

            if not pathHeap or MAX_LENGTH==0:
                self.log("Search ended: searched all the paths "+
                         "up to length %i." % MAX_LENGTH)
                break
            if transitionsSearched >= SEARCH_TRANSITIONS:
                self.log("Search ended: hit the maximum transitions limit "+
                         "of %i transitions" % SEARCH_TRANSITIONS)
                break

            # always taking one path from pathHeap and increasing its length by
            # the outgoing transitions of its last state. the increased paths
            # are again put to pathHeap.

            path,badness = self._fromHeap( heappop(pathHeap) )

            # push the req and mark the path executed.

            req.push()

            last = path[-1]

            # If the req has transitionPoints method, we'll use that.
            # Otherwise, using getPercentage (all reqs should have that).
            if useTP:
                for t in path[:-1]:
                    req.markExecuted(t)
                badness -= req.transitionPoints(last)
                badness += self._transitionweight(last)
            else:
                for t in path:
                    req.markExecuted(t)
                # adding a nonpositive number
                badness = startCov - req.getPercentage()

            # popping the req resets the changes done after push.
            req.pop()

            # is this the best path so far?
            if badness < leastBadness:
                leastBadness = badness
                bestPaths = [path]
                if self._greedy:
                    # we've found a path that's better than nothing.
                    # if we're greedy, that's all we need.
                    break
            elif badness == leastBadness:
                # this is equally good as the best path
                bestPaths.append(path)

            if len(path) < MAX_LENGTH:
                isDeadEnd = True # dead end until proven otherwise
                for t in last.getDestState().getOutTransitions():
                    if self._tranShouldBeSearched(t,path,seenTrans):
                        # add an one-transition-longer path to pathHeap
                        heappush(pathHeap, self._toHeap(path+(t,),badness))
                        seenTrans.add(t)
                        isDeadEnd = False
                if isDeadEnd:
                    deadEnds.append(path)

            else:
                maxLenPaths.append(path)

            transitionsSearched += 1

        if leastBadness == 0:
            # no good paths found...
            if pathHeap:
                self.log("Returning a random unsearched path.")
                p,unused = self._fromHeap(random.choice(pathHeap))
                return p
            elif maxLenPaths:
                self.log("Returning a random max_len path (len = %i)" % (MAX_LENGTH,))
                return random.choice( maxLenPaths )
            elif deadEnds:
                self.log("Returning a random dead end path.")
                return random.choice( deadEnds )
        else:
            # found one or more good paths
            shortestBestPathLen = min([len(q) for q in bestPaths])
            shortestBestPaths = [p for p in bestPaths
                                 if len(p) == shortestBestPathLen]
            bestPath = random.choice(shortestBestPaths)
            self.log("Returning a path whose length is %i, badness = %f" % (
                len(bestPath),leastBadness) )
            return bestPath

    def _tranShouldBeSearched(self,t,path,seenTrans):
        return (self._seco == NO_CROSSING_PATHS and t not in seenTrans
                or
                self._seco == NO_LOOPS and t not in path
                or
                self._seco == NONE)

    def _newPathCanBeCreated(self,pathHeap):
        return not self._maxNumPaths or len(pathHeap) < self._maxNumPaths
