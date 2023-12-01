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
GameGuidance reads the following parameter values:

- lookahead (natural number, default: 15)

  the search depth to which the algorithm explores the state space
  before answering.

- randomseed (any hashable object, default: None)

  seed for random number generator

- rerouteafter (natural number, default: 1)

  route will be recalculated when rerouteafter steps have been taken
  (or when execution has run out of the previous route in any case)

"""

# TODO
#
# Do not reroute at all when:
#
# - transitions from the current state cause the same function call in
# keyword adapter (vwVerifyText, ~vwVerifyText)
#
# Do not reroute too long when:
#
# - all transitions are about verification, not sending real input
# events


from tema.guidance.guidance import Guidance as GuidanceBase
import random
import time # for random seed initialization


version='0.15 very simple player, chooses a route with max points with min steps'

class Guidance(GuidanceBase):

    FINISHPOINTS="found goal"

    def __init__(self):
        GuidanceBase.__init__(self)
        self.setParameter('lookahead',15)
        self.setParameter('randomseed',time.time())
        self.setParameter('rerouteafter',1)
        self._lastroute=[]
        self._steps_to_reroute=0

    def setParameter(self,parametername,parametervalue):
        if not parametername in ['lookahead','randomseed','rerouteafter']:
            print __doc__
            raise Exception("Invalid parameter '%s' for gameguidance." % parametername)
        GuidanceBase.setParameter(self,parametername,parametervalue)
        if parametername=='randomseed':
            self._rndchoose=random.Random(parametervalue).choice
        elif parametername=='rerouteafter':
            self._steps_to_reroute=self.getParameter(parametername)

    def suggestAction(self,state_object):
        if self._steps_to_reroute<=0 \
               or self._lastroute==[] \
               or not state_object==self._lastroute[-1].getSourceState():
            # We need to calculate new route. It will be written to
            # self._lastroute.

            if len(state_object.getOutTransitions())==1:
                # If there is only one possible transition, forget the
                # routing for now. Next suggestAction call causes
                # rerouting anyway, because then self._lastroute will
                # be empty
                self._lastroute=[state_object.getOutTransitions()[0]]
                self.log("There is only one possible action: %s" % self._lastroute[-1].getAction())
            else:
                self.log("Rerouting...")
                points,self._lastroute = self._plan_route(state_object,self.getParameter('lookahead'))
                self._steps_to_reroute=self.getParameter('rerouteafter')
                
                log_actions=[t.getAction().toString() for t in self._lastroute[::-1]]
                self.log("New route: points: %s, route: %s" % (points,log_actions))
        else:
            self.log("Using the next action in the planned route: %s" %
                     self._lastroute[-1].getAction())

        next_transition=self._lastroute.pop()
        self._steps_to_reroute-=1
        return next_transition.getAction()

    def _plan_route(self,state_object,depth):
        """Returns a pair (points, path) where length of path is the
        parameter depth+1 and points is a pair
        (points_in_the_end_of_path,
        number_of_unnecessary_depth_in_the_end_of_the_path).
        The unnecessary steps do not increase the points.
        """
        # if no look-ahead, return zero points and any out transition
        if depth<=0:
            try:
                transition=[self._rndchoose(
                    state_object.getOutTransitions() )]
                return ([sum([r.getPercentage() for r in self._requirements]),0],
                        transition)
            except:
                self.log("Deadlock detected, gameguidance cannot continue.")
                self.log("Deadlock state: %s" % state_object)
                raise Exception("Unexpected deadlock in the test model.")

        outtrans=state_object.getOutTransitions()

        # Initialize transition point table of length of
        # outtransitions with pairs of zeros. The table contains the
        # coverage points after execution the transition.
        points=[ [0.0,0] for t in outtrans]
        nonfinishing_routes=[None]*len(outtrans)
        finishing_routes=[]
        shortest_finishing_length=depth
        
        for transition_index,t in enumerate(outtrans):

            # mark transition t executed in every requirement and calc points
            for r in self._requirements:
                r.push()
                r.markExecuted(t)
                points[transition_index][0]+=r.getPercentage()
                
            if int(points[transition_index][0])>=len(self._requirements):
                # every requirement fulfilled
                finishing_routes.append([t])
                shortest_finishing_length=0
            elif shortest_finishing_length>0:
                future_points,route = self._plan_route(t.getDestState(),shortest_finishing_length-1)
                route.append(t)
                if future_points[0]==Guidance.FINISHPOINTS:
                    finishing_routes.append(route)
                    shortest_finishing_length=min(shortest_finishing_length,len(route))
                else:
                    if points[transition_index][0]==future_points[0]:
                        # there will be no increase in points in the future =>
                        # the search depth after which nothing happens increases
                        points[transition_index][1]=depth
                    else:
                        # future looks bright, wasted depth does not increase
                        # copy points and the depth
                        points[transition_index]=future_points
                    nonfinishing_routes[transition_index]=route
            
            # restore the transition execution status in every requirement
            for r in self._requirements:
                r.pop()

        # if there are finishing routes, return one of the shortest:
        if finishing_routes:
            route_lengths=[ len(r) for r in finishing_routes ]
            minlen=min(route_lengths)
            best_route_indexes=[ i for i,rl in enumerate(route_lengths) if rl==minlen ]
            chosen_route_index=self._rndchoose(best_route_indexes)
            return [Guidance.FINISHPOINTS,0],finishing_routes[ chosen_route_index ]
        else:
            # return any of the routes with maximum points
            # that give the maximum points with the smallest number of steps
            maximumpoints=max(points) # max ([ [1,9], [2,8], [2,8], [2,1] ]) == [2,8]
            best_route_indexes=[i for i,p in enumerate(points) if p==maximumpoints]
        return maximumpoints, nonfinishing_routes[ self._rndchoose(best_route_indexes) ]
