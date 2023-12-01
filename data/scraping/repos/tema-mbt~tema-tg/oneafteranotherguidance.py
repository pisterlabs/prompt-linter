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
oneafteranotherguidance executes multiple Guidances. After one guidance
finishes (= reaches 100% coverage), another guidance takes over.

The guidances and coverage requirements are defined in a file given as
a 'file' parameter to oneafteranotherguidance. Example of such a file:

# First, find all 'keke' actions with gameguidance. Then start random walk.
--guidance=gameguidance --coverage=clparser --coveragereq='actions .*keke.*'
--guidance=randomguidance

That is, one line per one set of arguments. The guidances are executed in the
same order as the lines in the file. The syntax is the same as with testengine
command line arguments. Allowed arguments are:

--guidance (REQUIRED)
--guidance-args
--coverage
--coveragereq
--coveragereq-args

NOTE: some of the coverages (dummycoverage, some findnew coverages, etc.) may
never reach 100%. That means that they'll never stop, and the guidances after
them won't be executed.
"""

from tema.guidance.guidance import Guidance as GuidanceBase

import shlex # for parsing a '--XXX=yyy --ZZZ="a b"' kind of argument string
import getopt # for parsing a list of args given by shlex

version='0.1 oneafteranotherguidance'

class Guidance(GuidanceBase):

    def __init__(self):
        GuidanceBase.__init__(self)
        self._guidances = []
        self._guidanceOpts = []
        self._covs = []
        self._currIndex = -1
        self._model = None
        self._filename = None

    def setTestModel(self,model):
        self._model = model
        for g in self._guidances:
            g.setTestModel(model)

    def setParameter(self,parametername,parametervalue):
        if parametername=='file':
            self._filename = parametervalue
        else:
            print __doc__
            raise Exception("Invalid param '%s' for oneafteranotherguidance."
                            % (parametername,))
#        GuidanceBase.setParameter(self,parametername,parametervalue)

    def prepareForRun(self):
        if self._filename is None:
            raise ValueError("oneafteranotherguidance needs a 'file' param.")
        self._addGuidancesFromFile(self._filename)

        # This is a bit ugly...
        # Setting the getPercentage method of the covreq that was given
        # to me by addRequirement. After this, _totalReq will simply answer
        # what I want it to answer (= my getPercentage()). It doesn't
        # matter what the covreq was originally.
        # i.e: we don't use the _totalReq to guide us, we'll just use it to
        # report the current progress of the actually used covreqs (= those
        # read from the file given as a 'file' parameter) to testengine.
        self._totalReq.getPercentage = lambda: self.getPercentage()

        self._startNextGuidance()

    def markExecuted(self, t):
        self._currGuidance.markExecuted(t)
        if self._currCov.getPercentage() == 1:
            self.log("Reached 100%% coverage: %s"
              %(_guidanceOptsAsStr(self._guidanceOpts[self._currIndex],)))
            if self._currIndex+1 < len(self._covs):
                self._startNextGuidance()

    def suggestAction(self, s):
        return self._currGuidance.suggestAction(s)

    def addRequirement(self, req):
        self._totalReq = req

    def getPercentage(self):
        # E.g: the 3rd one of the total of 4 covreqs is 60% covered ->
        # the total coverage is [2 * 100% * (1/4)] + [60% * (1/4)] = 65%
        covsFinished = self._currIndex
        shareOfOneCov = 1.0 / len(self._covs)
        shareOfFinished = covsFinished * shareOfOneCov
        shareOfCurr = self._currCov.getPercentage() * shareOfOneCov
        return shareOfFinished + shareOfCurr

    def _startNextGuidance(self):
        self._currIndex += 1
        self._currGuidance = self._guidances[self._currIndex]
        self._currCov = self._covs[self._currIndex]
        self._currGuidance.prepareForRun()
        self.log("New guidance: %s"
              %(_guidanceOptsAsStr(self._guidanceOpts[self._currIndex],)))

    def _addGuidancesFromFile(self,filename):
        f = file(filename)
        for line in f:
            strippedLine = _stripLine(line)
            if not strippedLine:
                continue
            opts = _parseOptionsLine(strippedLine)
            self._addGuidance(opts)
        f.close()

    def _addGuidance(self,opts):
        cov = _createCoverage(opts['coverage'],opts['coveragereq'],self._model)
        set_parameters(cov,opts['coveragereq-args'])
        self._covs.append(cov)

        guidance = _createGuidance(opts['guidance'])
        set_parameters(guidance,opts['guidance-args'])
        guidance.setTestModel(self._model)
        guidance.addRequirement(cov)
        self._guidances.append(guidance)

        self._guidanceOpts.append(opts)


def _stripLine(line):
    return line.split('#',1)[0].strip() # anything after '#' is comment

ARGS = ['guidance','guidance-args','coverage','coveragereq','coveragereq-args']
def _parseOptionsLine(line):
    opts, rest = getopt.getopt(shlex.split(line),[],['%s='%a for a in ARGS])
    if rest:
        raise Exception("Invalid arguments: %s'" % (rest,))
    opts = [(n[2:],v) for n,v in opts] # remove '--' from argument names
    optsDict = dict(opts)
    if 'guidance' not in optsDict:
        raise Exception("guidance argument required.")
    optsDict.update( dict((a,'') for a in ARGS if a not in optsDict) )
    return optsDict

def _createGuidance(guidance):
    guidancemodule=__import__("tema.guidance."+guidance,globals(),locals(),[''])
    return guidancemodule.Guidance()

def _createCoverage(coverage,coveragereq,model):
    if not coverage:
        from tema.coverage.dummycoverage import CoverageRequirement
        return CoverageRequirement('')
    covmodule=__import__("tema.coverage."+coverage,globals(),locals(),[''])
    return covmodule.requirement(coveragereq,model=model)

def _guidanceOptsAsStr(gopts):
    s = gopts['guidance']
    if gopts['guidance-args']:
        s += '(%s)'%(gopts['guidance-args'],)
    if gopts['coverage']:
        s += " using %s" % (gopts['coverage'],)
        if gopts['coveragereq-args']:
            s += '(%s)'%(gopts['coveragereq-args'],)
        if gopts['coveragereq']:
            s += " with '%s'"%(gopts['coveragereq'],)
    return s


# copy-pasted from testengine.
# couldn't import because importing testengine also runs it... :(
def set_parameters(object,argument_string):
    """Parse argument string and call setParameter-method of the
    object accordingly. For example argument string
    'port:9090,yellowflag,logger:adapterlog' implies calls
    setParameter('port',9090), setParameter('yellowflag',None),
    setParameter('logger',adapterlog_object)."""
    # TODO: implement special object-type parameters (not needed so far)
    for argpair in argument_string.split(","):
        if not argpair: continue
        if ":" in argpair:
            name,value=argpair.split(":",1)
        else:
            name,value=argpair,None
        try: object.setParameter(name,int(value))
        except Exception,e:
            if not (isinstance(e,TypeError) or isinstance(e,ValueError)): raise e
            try: object.setParameter(name,float(value))
            except Exception,e: 
                if not (isinstance(e,TypeError) or isinstance(e,ValueError)): raise e
                object.setParameter(name,value)
