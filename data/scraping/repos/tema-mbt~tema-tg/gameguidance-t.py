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
GameGuidance-Threading Guidance module

Notes:

- Do not use this guidance with test models that include deadlocks!

GameGuidance-Threading reads the following parameter values:

- maxdepth (natural number, default: 100)

  the search depth to which the algorithm explores the state space
  at maximum.

- mindepth (natural number, default: 0)

  the lower bound for the search depth before an action can be
  suggested.

- randomseed (any hashable object, default: None)

  seed for random number generator.

"""

from tema.guidance.guidance import Guidance as GuidanceBase
import random

import time
import thread
import copy

version='0.1 very simple player'




class GuidanceT(GuidanceBase):

    FINISHPOINTS="found goal"

    def __init__(self):
        GuidanceBase.__init__(self)
        self.setParameter('maxdepth',100)
        self.setParameter('mindepth',1)
        self.setParameter('randomseed',time.time())
        self._lastroute=[]
        self._steps_to_reroute=0

        # search front is a list of triplets: (score, path, coverage)
        # where score is a pair: (coverage_percentage, steps since
        # last change coverage_percentage). Thus, the bigger the
        # number of steps, the faster the final coverage_percentage is
        # achieved.
        self.search_front=[]
        self._thread_id=None
        self._front_shortened=0 # msg from markExecuted to thread

    def setParameter(self,parametername,parametervalue):
        if not parametername in ['maxdepth','mindepth','randomseed']:
            print __doc__
            raise Exception("Invalid parameter '%s' for gameguidance." % parametername)
        GuidanceBase.setParameter(self,parametername,parametervalue)
        if parametername=='randomseed':
            self._rndchoose=random.Random(parametervalue).choice

    def prepareForRun(self):
        GuidanceBase.prepareForRun(self)
        self.search_front_lock=thread.allocate_lock()

    def markExecuted(self,transition_object):
        locked=0 # keeps if this function has acquired search_front_lock

        # Update advances to the 'official' coverage object
        GuidanceBase.markExecuted(self,transition_object)
        
        # Then cleanup search front: remove every entry from the
        # search front if it starts with some other than the executed
        # transition_object, and shorten entries starting with the
        # transition.

        # Shortening cannot be done if paths are too short, therefore,
        # if the thread is running, let it do its job 
        while 1:
            self.search_front_lock.acquire()
            locked=1
            if len(self.search_front[0][1])<2:
                # There is just one action in the search front, it can
                # be safely removed only if the thread is no more
                # running, that is, time_to_quit signal has been
                # given.
                if self.time_to_quit: break
                else:
                    self.search_front_lock.release()
                    locked=0
                    time.sleep(1) # give some time to the thread
                    continue
                    
                # NOTE: This may cause a livelock if there are
                # deadlocks in the model: search front is not getting
                # any deeper. There should not be deadlocks!
            else:
                break
            
        # If the thread is quitting, there is no reason to
        # cleanup the search front
        #if self.time_to_quit:
        #    if locked: self.search_front_lock.release()
        #    return

        # This function must own the lock now, search_front can be
        # edited.
        new_search_front=[]
        for points,path,reqs in self.search_front:
            if path[0]==transition_object:
                self._front_shortened=1 # message to the thread
                new_search_front.append([points,path[1:],reqs])
        self.search_front=new_search_front
        self.log("Search front reduced to length %s and depth %s" %
                 (len(self.search_front),len(self.search_front[0][1])))
        self.search_front_lock.release()

    def suggestAction(self,state_object):
        # If a thread has not been started yet, start it now and give
        # it some time to find something. The first depth is reached
        # very fast.
        if self._thread_id==None:
            self.time_to_quit=0
            self._thread_id=thread.start_new_thread(
                self._route_planner_thread,(state_object,))

            time.sleep(1)

        # Choose randomly one the transitions that start the paths
        # with the best score.
        self.search_front_lock.acquire()

        if len(self.search_front)==0:
            # The search front should not be empty, because
            # suggestAction and markExecuted are executed one after
            # another, and markExecuted never finishes with an empty
            # search front.
            self.log("Strange! Search front should never be empty, but it is.")
            raise Exception("suggestAction found an empty search front")

        # If necessary, give the algorithm some time to reach the
        # minimal search depth.
        if self.search_front[0][0][0] < 1.0: # not finished yet
            while len(self.search_front[0][1]) < self._params['mindepth']:
                self.search_front_lock.release()
                time.sleep(1)
                self.search_front_lock.acquire()
    
        max_points=self.search_front[-1][0]
        best_transition=self._rndchoose(
            [path[0] for po,path,tr in self.search_front if po==max_points])
        
        self.search_front_lock.release()
        return best_transition.getAction()

    def _route_planner_thread(self,starting_state):
        self.log("Route planner thread started")

        # initialize search front
        self.search_front_lock.acquire()
        for t in starting_state.getOutTransitions():
            reqs=copy.deepcopy(self._requirements)
            for r in reqs: r.markExecuted(t)
            self.search_front.append(
                [(sum([r.getPercentage() for r in reqs]),0),[t],reqs])
        self.search_front_lock.release()

        while not self.time_to_quit:
            # let someone else use search_front...
            time.sleep(0.05)
            
            # Increase search front depth by one level:

            # 1. Go through a copy of the search front
            self.search_front_lock.acquire()
            shallow_copy_of_search_front=copy.copy(self.search_front)
            self.search_front_lock.release()

            if len(shallow_copy_of_search_front[0][1])>=self.getParameter('maxdepth'):
                time.sleep(1)
                continue # maximum depth reached, do not calculate more
            
            new_search_front=[]
            for points,path,reqs in shallow_copy_of_search_front:
                if self._front_shortened==1:
                    # markExecuted touched the front, forget
                    # the update of this front....
                    break 
                for t in path[-1].getDestState().getOutTransitions():
                    nreqs=copy.deepcopy(reqs)
                    npath=copy.copy(path)
                    npath.append(t)
                    for r in nreqs: r.markExecuted(t)
                    new_perc=sum([r.getPercentage() for r in nreqs])
                    if new_perc==points[0]: new_steps=points[1]+1
                    else: new_steps=0 # the percentage has grown!
                    new_search_front.append([
                        (new_perc,new_steps),
                        npath,
                        nreqs])
            new_search_front.sort()

            # 2. If the search front has not been changed during the
            # search, it can be updated. Otherwise, forget the results
            # and try to update the new search front.
            
            self.search_front_lock.acquire()
            if self._front_shortened==0:
                self.search_front=new_search_front
                self.log("New search front length %s, depth %s, score %s" %
                         (len(new_search_front),
                          len(new_search_front[-1][1]),
                          new_search_front[-1][0]))
            else:
                self._front_shortened=0
                self.log("Throwing away depth %s, rerouting from depth %s" %
                         (len(shallow_copy_of_search_front[0][1]),
                          len(self.search_front[0][1])))

            if self.search_front[0][0][0]>=1.0:
                self.time_to_quit=1
                self.log("Nothing can possibly go wrong anymore")
            self.search_front_lock.release()
            
Guidance=GuidanceT
