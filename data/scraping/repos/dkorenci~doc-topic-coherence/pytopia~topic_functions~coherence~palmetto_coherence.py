import palmettonew

from org.aksw.palmetto.custom import CoherenceBuilder, CoherenceParams
from palmettonew import JArray

javaVMInitialized = False
def initVM():
    '''initialize java virtual machine'''
    global javaVMInitialized
    if not javaVMInitialized:
        try:
            palmettonew.initVM(vmargs=['-Djava.awt.headless=true',
                                    '-Dsun.arch.data.model=64', '-Dsun.cpu.endian=little'])
        except:
            palmettonew.initVM()
        javaVMInitialized = True

initVM()

from pytopia.tools.IdComposer import IdComposer
from pytopia.context.ContextResolver import resolve

class PalmettoCoherence(IdComposer):

    def __init__(self, measure, index, topWords, windowSize=0, wordTransform=None, standard=True):
        '''
        :param measure: string describing coherence measure
        :param index: pytopia id of Palmetto Lucene index
        :param topWords: number of top topic words used to calculate coherence
        '''
        if measure not in ['umass', 'uci', 'npmi', 'c_a', 'c_p', 'c_v']:
            raise Exception('unknown coherence measure: %s' % measure)
        self.measure = measure
        self.index = index
        self.topWords = topWords
        self.windowSize = windowSize
        self.wordTransform = wordTransform
        self.standard = standard
        IdComposer.__init__(self)
        self.coherence = None

    def __constructMeasure(self):
        if self.coherence: return
        indexFile = resolve(self.index)
        cp = CoherenceParams(self.measure, indexFile, self.standard, self.windowSize)
        self.coherence = CoherenceBuilder.constructCoherence(cp)

    def topic2list(self, topic):
        '''
        Transforms string of whitespace separated words to a list,
         and applies word transformation.
        '''
        words = topic.split()
        if self.wordTransform is None: return words
        return [ self.wordTransform(w) if self.wordTransform(w) != None else w
                        for w in words ]

    def __call__(self, topic):
        '''
        :param topic: (modelId, topicId)
        '''
        mid, tid = topic
        model = resolve(mid)
        return self.calculateCoherence(model.topic2string(tid, topw=self.topWords))

    def calculateCoherence(self, topicStr):
        '''
        :param topicStr: a string of whitespace separated words,
                or a list of such strings
        :return: coherence value, or a list of values
        '''
        self.__constructMeasure()
        batch = isinstance(topicStr, list)
        l = len(topicStr) if batch else 1
        jarray = JArray('object')(l)
        if not batch:
            jarray[0] = strings2JavaArray(self.topic2list(topicStr))
            res = self.coherence.calculateCoherences(jarray)
            return res[0]
        else:
            for i, t in enumerate(topicStr): jarray[i] = strings2JavaArray(self.topic2list(t))
            res = self.coherence.calculateCoherences(jarray)
            return [c for c in res]

def strings2JavaArray(strings):
    '''
    :param strings: iterable of strings
    :return: JArray jcc adapter for java array of strings
    '''
    jarr = JArray('string')(len(strings))
    for i, _ in enumerate(strings): jarr[i] = strings[i]
    return jarr