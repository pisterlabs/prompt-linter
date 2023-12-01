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
sharedtabuguidance - a guidance that shares a tabulist between processes.

Uses multiprocessing -> requires python 2.6+

The tabulist is in a separate process which can be started
by giving 'startandconnect:PORT' arg. It listens to localhost:PORT.
After that, other sharedtabuguidances with arg 'connect:PORT' will
use the same tabulist.

When the starter sharedtabuguidance stops, the tabulist process stops also. :(

Accepted guidance-args:

    'startandconnect:PORT'
        Starts a shared tabulist on localhost:PORT, and connects to it.

    'connect:PORT'
        Connects to an existing shared tabulist on localhost:PORT.

Other guidance-args, only accepted with 'startandconnect'.
('connect'ing guidances will use those same args)
    
    'tabuitems:TABUITEMTYPE'
        One of the following:
            state (default)
            statecomponent
            transition


NOTE:

The tabulist is updated only after a transition has been actually
executed (= markExecuted(t) is called).

Example: (processes P1 and P2)
P1: non-tabu transition T is suggested.
P2: the same non-tabu transition T is suggested.
P1: T is executed
P1: T is added to the tabulist.
P2: T is executed (it's actually tabu now, but P2 doesn't know it)
P2: T is added to the tabulist. (useless but harmless)

Don't think this is a serious issue. Could be fixed by implementing some
kind of nextToBeExecutedTransitions tabulist alongside the main tabulist.
"""

# TODO: maybe get rid of this and
# do a more general kinda shared guidance thing or something...

version="0.01"

from tema.guidance.guidance import Guidance as GuidanceBase
from tema.coverage.tabulist import TabuList
import random

try:
    from multiprocessing.managers import SyncManager
    from multiprocessing import Lock
except ImportError,e:
    from processing.managers import SyncManager
    from processing import Lock


class TabuListManager(SyncManager):
    pass

class TabuListUser:
    # The tabulist.
    # There's only 1 tabulist per process (it's a class variable).
    _THE_TABULIST = TabuList() 
    # Locked when somebody's using the tabulist.
    _TABULIST_LOCK = Lock()
    # Number of connected TabuListUsers.
    _CONNECTED = 0
    _CONN_LOCK = Lock()
    _PARAMS = None
    def __init__(self,params=None):
        TabuListUser._CONN_LOCK.acquire()
        TabuListUser._CONNECTED += 1
        self._connNum = TabuListUser._CONNECTED
        if params is not None:
            TabuListUser._PARAMS = params
        TabuListUser._CONN_LOCK.release()

    def getParameters(self):
        return TabuListUser._PARAMS

    def connNum(self):
        return self._connNum

    def len(self):
        TabuListUser._TABULIST_LOCK.acquire()
        le = len(TabuListUser._THE_TABULIST)
        TabuListUser._TABULIST_LOCK.release()
        return le

    def add(self, item):
        TabuListUser._TABULIST_LOCK.acquire()
        TabuListUser._THE_TABULIST.add(item)
        TabuListUser._TABULIST_LOCK.release()

    def addMany(self, items):
        TabuListUser._TABULIST_LOCK.acquire()
        for item in items:
            TabuListUser._THE_TABULIST.add(item)
        TabuListUser._TABULIST_LOCK.release()

    def tabunessOf(self, items):
        """ Eg. If the 3 first items are tabu and the last one is not,
            returns: (True,True,True,False)
        """
        TabuListUser._TABULIST_LOCK.acquire()
        to = tuple([i in TabuListUser._THE_TABULIST for i in items])
        TabuListUser._TABULIST_LOCK.release()
        return to

TabuListManager.register('TabuList',TabuListUser)

def _getTabuListManager(port):
    manager = TabuListManager( ('127.0.0.1', port),
                              authkey='tema_shared_tabulist_%s'%(version,) )
    return manager


class Guidance(GuidanceBase):
    def __init__(self):
        GuidanceBase.__init__(self)
        self._port = None
        self._manager = None
        self._iAmTheManagerStarter = False
        self._sgParams = []

    def setParameter(self,name,value):
        if name == 'help':
            print __doc__
            raise Exception()
        elif name == 'connect':
            self._port = value
        elif name == 'startandconnect':
            self._port = value
            self._iAmTheManagerStarter = True
        else:
            self._sgParams.append( (name,value) )
#        GuidanceBase.setParameter(self,name,value)

    def _setParameterForReal(self,name,value):
        if name in ('tabuitems','tabuitem'):
            if value.startswith('statecomp'):
                self.markExecuted = self._markExecuted_destStateComps
                self.suggestAction = self._suggestAction_destStateComps
            elif value in ('state','states'):
                self.markExecuted = self._markExecuted_destState
                self.suggestAction = self._suggestAction_destState
            elif value in ('transition','transitions'):
                self.markExecuted = self._markExecuted_transition
                self.suggestAction = self._suggestAction_transition
            else:
                raise ValueError("Invalid tabuitems: %s" % (value,))
        else:
            raise ValueError("Invalid argument: %s" % (name,))

    def prepareForRun(self):
        if self._port is None:
            raise ValueError("'connect' or 'startandconnect' must be given!")

        if self._sgParams and not self._iAmTheManagerStarter:
            raise ValueError("Setting parameters are only allowed "+
                             "with 'startandconnect'. When connecting, "+
                             "we just use existing params.")

        self._manager = _getTabuListManager(self._port)
        if self._iAmTheManagerStarter:
            for (n,v) in self._sgParams:
                self._setParameterForReal(n,v)
            self.log("Starting a new shared tabulist on port %i."%(self._port))
            self._manager.start()
            self.log("Started.")
            self._remoteTabuList = self._manager.TabuList(self._sgParams)
        else:
            self.log("Connecting to an existing shared tabulist on port %i"%(
                     self._port))
            self._manager.connect()
            self.log("Connected.")
            self._remoteTabuList = self._manager.TabuList()
            self._sgParams = self._remoteTabuList.getParameters()
            for (n,v) in self._sgParams:
                self._setParameterForReal(n,v)
        self.log("The guidance params are: %s" % (self._sgParams,))

        le = self._remoteTabuList.len()
        connNum = self._remoteTabuList.connNum()
        self.log(("I was the guidance number %i to connect to this tabulist."+
                  " It already contains %i items.")%(connNum,le))


    def _markExecuted_destState(self, transition):
        s = str(transition.getDestState())
        self._remoteTabuList.add(s)

    def _suggestAction_destState(self, from_state):
        trans = from_state.getOutTransitions()
        acts = [t.getAction() for t in trans]
        dests = [str(t.getDestState()) for t in trans]
        tabus = self._remoteTabuList.tabunessOf(dests)
        nonTabuActs = [a for i,a in enumerate(acts) if not tabus[i]]
        self.log("%i/%i of possible actions are non-tabu."%(
                 len(nonTabuActs),len(acts)))
        if nonTabuActs:
            a = random.choice(nonTabuActs)
            self.log("Returning a non-tabu action %s"%(a,))
        else:
            a = random.choice(acts)
            self.log("Returning a tabu action %s"%(a,))
        return a

    markExecuted = _markExecuted_destState
    suggestAction = _suggestAction_destState


    def _markExecuted_destStateComps(self, transition):
        self._remoteTabuList.addMany(_compStates(transition))

    def _suggestAction_destStateComps(self, from_state):
        actNont = [(t.getAction(),self._nontabunessOfDestStateComps(t)) for
                   t in from_state.getOutTransitions()] 
        maxNont = max([nont for a,nont in actNont])
        bestActions = [a for (a,nont) in actNont if nont==maxNont]
        self.log("There are %i actions with %i non-tabuness."%(
                 len(bestActions),maxNont))
        a = random.choice(bestActions)
        return a

    def _nontabunessOfDestStateComps(self,transition):
        tabunesses = self._remoteTabuList.tabunessOf(_compStates(transition))
        return tabunesses.count(False)


    def _markExecuted_transition(self, transition):
        self._remoteTabuList.add(_transitionAsPicklable(transition))

    def _suggestAction_transition(self, from_state):
        trans = from_state.getOutTransitions()
        picklTrans = [_transitionAsPicklable(t) for t in trans]
        acts = [t.getAction() for t in trans]
        tabus = self._remoteTabuList.tabunessOf(picklTrans)
        nonTabuActs = [a for i,a in enumerate(acts) if not tabus[i]]
        self.log("%i/%i of possible actions are non-tabu."%(
                 len(nonTabuActs),len(acts)))
        if nonTabuActs:
            a = random.choice(nonTabuActs)
            self.log("Returning a non-tabu action %s"%(a,))
        else:
            a = random.choice(acts)
            self.log("Returning a tabu action %s"%(a,))
        return a


    def isThreadable(self):
        # sharedtabuguidance won't work as threaded! (don't really know why)
        return False


def _compStates(transition):
    jeje = [s._id for s in transition.getDestState()._id]
    return tuple([(i,s) for i,s in enumerate(jeje)])

def _transitionAsPicklable(transition):
    return (str(transition.getSourceState()),
            str(transition.getAction()),
            str(transition.getDestState()))
