from __future__ import division
import sys
import pickle
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
sys.path.append('./pyltr/')
import pyltr
sys.path.append('../wikisim/')
from wikipedia import *
from operator import itemgetter
import requests
import json
import nltk
import scipy as sp
import scipy.sparse as sprs
import scipy.spatial
import scipy.sparse.linalg
from calcsim import *
import numpy as np
from pycorenlp import StanfordCoreNLP
scnlp = StanfordCoreNLP('http://localhost:9000')
import copy
from sklearn.preprocessing import OneHotEncoder
from unidecode import unidecode


MIN_MENTION_LENGTH = 3 # mentions must be at least this long
MIN_FREQUENCY = 20 # anchor with frequency below is ignored

# with open('/users/cs/amaral/wikisim/wikification/pos-filter-out-nonmentions.txt', 'r') as srcFile:
#     posFilter = srcFile.read().splitlines()

def get_solr_count(s):
    """ Gets the number of documents the string occurs 
        NOTE: Multi words should be quoted
    Arg:
        s: the string (can contain AND, OR, ..)
    Returns:
        The number of documents
    """

    q='+text:(\"%s\")'%(s,)
    qstr = 'http://localhost:8983/solr/enwiki20160305/select'
    params={'indent':'on', 'wt':'json', 'q':q, 'rows':0}
    r = requests.get(qstr, params=params)
    try:
        if 'response' not in r.json():
            return 0
        else:
            return r.json()['response']['numFound']
    except:
        return 0

def get_mention_count(s):
    """
    Description:
        Returns the amount of times that the given string appears as a mention in wikipedia.
    Args:
        s: the string (can contain AND, OR, ..)
    Return:
        The amount of times the given string appears as a mention in wikipedia
    """
    
    result = anchor2concept(s)
    rSum = 0
    for item in result:
        rSum += item[1]
        
    return rSum

def mentionProb(text):
    """
    Description:
        Returns the probability that the text is a mention in Wikipedia.
    Args:
        text: 
    Return:
        The probability that the text is a mention in Wikipedia.
    """
    
    totalMentions = get_mention_count(text)
    totalAppearances = get_solr_count(text.replace(".", ""))
    
    if totalAppearances == 0:
        return 0 # a mention never used probably is not a good link
    else:
        return totalMentions/totalAppearances
    
def normalize(nums):
    """Normalizes a list of nums to its sum + 1"""
    
    numSum = sum(nums) + 1 # get max
    
    # fill with normalized
    normNums = []
    for num in nums:
        normNums.append(num/numSum)
        
    return normNums

def destroyExclusiveOverlaps(textData):
    """
    Description:
        Removes all overlaps that start at same letter from text data, so that only the best mention in an
        overlap set is left.
    Args:
        textData: [[start, end, text, anchProb],...]
    Return:
        textData minus the unesescary elements that overlap.
    """
    
    newTextData = [] # textData minus the unesescary parts of the overlapping
    overlappingSets = [] # stores arrays of the indexes of overlapping items from textData
    
    # creates the overlappingSets array
    i = 0
    while i < len(textData)-1:
        # even single elements considered overlapping set
        # this is root of overlapping set
        overlappingSets.append([i])
        overlapIndex = len(overlappingSets) - 1
        theBegin = textData[i][0]
        
        # look at next words until not overlap
        for j in range(i+1, len(textData)):
            # if next word starts before endiest one ends
            if textData[j][0] == theBegin:
                overlappingSets[overlapIndex].append(j)
                i = j # make sure not to repeat overlap set
            else:
                # add final word
                if j == len(textData) - 1:
                    overlappingSets.append([j])
                break
        i += 1
                    
    # get only the best overlapping element of each set
    for oSet in overlappingSets:
        bestIndex = 0
        bestScore = -1
        for i in oSet:
            score = mentionProb(textData[i][2])
            if score > bestScore:
                bestScore = score
                bestIndex = i
        
        # put right item in new textData
        newTextData.append(textData[bestIndex])
        
    return newTextData

def destroyResidualOverlaps(textData):
    """
    Description:
        Removes all overlaps from text data, so that only the best mention in an
        overlap set is left.
    Args:
        textData: [[start, end, text, anchProb],...]
    Return:
        textData minus the unesescary elements that overlap.
    """
    
    newTextData = [] # to be returned
    oSet = [] # the set of current overlaps
    rootWIndex = 0 # the word to start looking from for finding root word
    rEnd = 0 # the end index of the root word
    
    # keep looping as long as overlaps
    while True:
        oSet = []
        oSet.append(textData[rootWIndex])
        for i in range(rootWIndex + 1, len(textData)):
            # if cur start before root end
            if textData[i][0] < textData[rootWIndex][1]:
                oSet.append(textData[i])
            else:
                break # have all overlap words

        
        bestIndex = 0
        # deal with the overlaps
        if len(oSet) > 1:
            bestProb = 0
            
            # choose the most probable
            i = 0
            for mention in oSet:
                prob = mentionProb(mention[2])
                if prob > bestProb:
                    bestProb = prob
                    bestIndex = i
                i += 1
        else:
            rootWIndex += 1 # move up one if no overlaps
                
        # remove from old text data all that is not best
        for i in range(0, len(oSet)):
            if i <> bestIndex:
                textData.remove(oSet[i])
                
        # add the best to new
        if not (oSet[bestIndex] in newTextData):
            newTextData.append(oSet[bestIndex])
            
        if rootWIndex >= len(textData):
            break
    
    return newTextData
    
def mentionStartsAndEnds(textData, forTruth = False):
    """
    Description:
        Takes in a list of mentions and turns each of its mentions into the form: [wIndex, start, end]. 
        Or if forTruth is true: [[start,end,entityId]]
    Args:
        textData: {'text': [w1,w2,w3,...] , 'mentions': [[wordIndex,entityTitle],...]}, to be transformed 
            as described above.
        forTruth: Changes form to use.
    Return:
        The mentions in the form [[wIndex, start, end],...]]. Or if forTruth is true: [[start,end,entityId]]
    """
    
    curWord = 0 
    curStart = 0
    for mention in textData['mentions']:
        while curWord < mention[0]:
            curStart += len(textData['text'][curWord]) + 1
            curWord += 1
            
        ent = mention[1] # store entity title in case of forTruth
        mention.pop() # get rid of entity text
        
        if forTruth:
            mention.pop() # get rid of wIndex too
            
        mention.append(curStart) # start of the mention
        mention.append(curStart + len(textData['text'][curWord])) # end of the mention
        
        if forTruth:
            mention.append(title2id(ent)) # put on entityId
    
    return textData['mentions']

posBefDict = {
    'IN':0,
    'DT':1,
    'NNP':2,
    'JJ':3,
    ',':4,
    'CC':5,
    'NN':6,
    'VBD':7,
    'CD':8,
    '(':9,
    'TO':10,
    'FAIL':11
}

posCurDict = {
    'NNP':0,
    'NN':1,
    'JJ':2,
    'NNS':3,
    'CD':4,
    'NNPS':5,
    'FAIL':6
}

posAftDict = {
    ',':0,
    '.':1,
    'IN':2,
    'NNP':3,
    'CC':4,
    'NN':5,
    'VBD':6,
    ':':7,
    'VBZ':8,
    'POS':9,
    'NNS':10,
    'TO':11,
    'FAIL':12
}
     
def getGoodMentions(splitText, mentions, model, overlapFix = False):
    """
    Description: 
        Finds the potential mentions that are deemed good by our classifier.
    Args:
        splitText: The text in split form.
        mentions: All of the potential mentions [word index, start offset, end offset].
        model: The machine learning model to predict with.
        overlapFix: Whether there are overlaps that need to be dealt with.
    Return:
        A subset of arg mentions with each element deemed to be worthy by the classifier.
    """
    
    goodMentions = [] # the mentions to return
    
    #Get POS tags of all text
    postrs = nltk.pos_tag(copy.deepcopy(splitText))

    # get stanford core mentions
    try:
        try:
            atext = " ".join(splitText).encode('utf-8')
        except:
            atext = " ".join(splitText)
        stnfrdMentions0 = scnlp.annotate(atext, properties={
            'annotators': 'entitymentions',
            'outputFormat': 'json'})
    except:
        print 'Error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
        k = 1/0
    stnfrdMentions = []
    for sentence in stnfrdMentions0['sentences']:
        for mention in sentence['entitymentions']:
            stnfrdMentions.append(mention['text'])
            
    enc = OneHotEncoder(n_values = [12,7,13], categorical_features = [0,1,2])
            
    for i in range(len(mentions)):
        aMention = [] # fill with attributes about current mention for prediction
        
        """ 
        Append POS tags of before, on, and after mention.
        """
        if i == 0:
            bef = 'NONE'
        else:
            bef = postrs[i-1][1] # pos tag of before
        if bef in posBefDict:
            bef = posBefDict[bef]
        else:
            bef = posBefDict['FAIL']
            
        on = postrs[i][1] # pos tag of mention
        if on in posCurDict:
            on = posCurDict[on]
        else:
            on = posCurDict['FAIL']
        
        if i == len(splitText) - 1:
            aft = 'NONE'
        else:
            aft = postrs[i+1][1] # pos tag of after
        if aft in posAftDict:
            aft = posAftDict[aft]
        else:
            aft = posAftDict['FAIL']
        
        aMention.extend([bef, on, aft])
        
        """
        Append mention probability.
        """
        aMention.append(mentionProb(splitText[mentions[i][0]]))
        
        """
        Find whether Stanford NER decides the word to be mention.
        """
        if splitText[mentions[i][0]] in stnfrdMentions:
            stnfrdMentions.remove(splitText[mentions[i][0]])
            aMention.append(1)
        else:
            aMention.append(0)
            
        """
        Whether starts with capital.
        """
        if splitText[mentions[i][0]][0].isupper():
            aMention.append(1)
        else:
            aMention.append(0)
            
        """
        Whether there is an exact match in Wikipedia.
        """
        if title2id(splitText[mentions[i][0]]) is not None:
            aMention.append(1)
        else:
            aMention.append(0)
            
        """
        Whether word contains a space.
        """
        if ' ' in splitText[mentions[i][0]]:
            aMention.append(1)
        else:
            aMention.append(0)
            
        """
        Whether the word contains only ascii characters.
        """
        try:
            splitText[mentions[i][0]].decode('ascii')
            aMention.append(1)
        except:
            aMention.append(0)
            
        """
        Get all positive classified instances.
        """
        #aMention = enc.fit_transform([aMention]).toarray()[0]
        if model.predict([aMention])[0] == 1:
            goodMentions.append(mentions[i])
            # put score of prediction
            goodMentions[-1].append(model.predict_proba([aMention])[0][1]) # put score of prediction
            
    
    # get the right amount needed
    if True:
        amount = 5 + int(0.12778 * len(splitText))
    else:
        amount = len(goodMentions)
            
    if overlapFix == False:
        goodMentions = sorted(goodMentions, key = itemgetter(1), reverse = False)[:amount]
        return goodMentions        
    else:
        """
        Remove all overlaps in results.
        """
            
        # sort on prediction probability descending
        goodMentions = sorted(goodMentions, key = itemgetter(-1), reverse = True)

        try:
            goodlen = len(goodMentions[0])
        except:
            return []

        for mention1 in goodMentions:
            if len(mention1) > goodlen:
                continue
            for mention2 in goodMentions:
                # dont do anything with a previous or same one
                if (mention2[0] == mention1[0] or
                        mention1[-1] < mention2[-1] or
                        len(mention2) > goodlen):
                    continue
                # flag 2 if 2 starts before 1 ends and 2 ends after 1 starts
                if mention2[1] < mention1[2] and mention2[2] >= mention1[1]:
                    #print 'Overlap found', str(mention1), str(mention2)
                    mention2.append(0) # just increase length to flag for deletion

        finalMentions = []
        for mention in goodMentions:
            if len(mention) == goodlen:
                finalMentions.append(mention[:3])
                
        finalMentions = sorted(finalMentions, key = itemgetter(1), reverse = False)[:amount]

        return finalMentions
    
def mentionExtract(text, mthd = 'cls2'):
    """
    Description:
        Takes in a text and splits it into the different words/mentions.
    Args:
        text: The text to be split.
        useCoreNLP: Whether to use CoreNLP entity mention annotation.
            Currently severely broken, never set to True.
    Return:
        The text split it into the different words / mentions: 
        {'text':[w1,w2,...], 'mentions': [[wIndex,begin,end],...]}
    """
    
    if mthd == 'cnlp': # use CoreNLP's entity mention annotator
        output = scnlp.annotate(text, properties={
            'annotators': 'entitymentions',
            'outputFormat': 'json'
        })
        
        # get all tokens together and all mentions together
        tokens = []
        mentions0 = []
        for sentence in output['sentences']:
            for token in sentence['tokens']:
                tokens.append(token['originalText'])
            for em in sentence['entitymentions']:
                mentions0.append(em)
        
        # put it all into splitText and mentions in the right way
        splitText = []
        mentions = []
        curT = 0 # token index
        curM = 0 # mention index
        while(curT < len(tokens)):
            if curM < len(mentions0) and curT == mentions0[curM]['docTokenBegin']:
                # put in entity mention
                splitText.append(mentions0[curM]['text'])
                mentions.append([len(splitText) - 1,
                                 mentions0[curM]['characterOffsetBegin'], 
                                 mentions0[curM]['characterOffsetEnd']])
                curT = mentions0[curM]['docTokenEnd']
                curM += 1
                continue
            else:
                # put in next token
                splitText.append(tokens[curT])
                curT += 1
        
    elif mthd == 'cls1': # this one lest solr deal with overlaps
        addr = 'http://localhost:8983/solr/enwikianchors20160305/tag'
        params={'overlaps':'LONGEST_DOMINANT_RIGHT', 'tagsLimit':'5000', 'fl':'id','wt':'json','indent':'on'}
        try:
            tmp1 = text.encode('utf-8')
        except:
            tmp1 = text
        r = requests.post(addr, params=params, data=tmp1)
        textData0 = r.json()['tags']
        splitText = [] # the text now in split form
        mentions = [] # mentions before remove inadequate ones
        textData = [] # [[begin,end,word,anchorProb],...]
        i = 0 # for wordIndex
        # get rid of extra un-needed Solr data
        for item in textData0:
            mentions.append([i, item[1], item[3]])
            i += 1
            # also fill split text
            splitText.append(text[item[1]:item[3]])
        if 'gbc-er' not in mlModels:
            mlModels['gbc-er'] = pickle.load(open(mlModelFiles['gbc-er'], 'rb'))
        mentions = getGoodMentions(splitText, mentions, mlModels['gbc-er'])
        
    elif mthd == 'cls2': # this one we deal with overlaps
        addr = 'http://localhost:8983/solr/enwikianchors20160305/tag'
        params={'overlaps':'ALL', 'tagsLimit':'5000', 'fl':'id','wt':'json','indent':'on'}
        
        try:
            tmp1 = text.encode('utf-8')
        except:
            tmp1 = text
        r = requests.post(addr, params=params, data=tmp1)
        textData0 = r.json()['tags']
        splitText = [] # the text now in split form
        mentions = [] # mentions before remove inadequate ones
        textData = [] # [[begin,end,word,anchorProb],...]
        i = 0 # for wordIndex
        # get rid of extra un-needed Solr data
        for item in textData0:
            mentions.append([i, item[1], item[3]])
            i += 1
            # also fill split text
            splitText.append(text[item[1]:item[3]])
        if 'gbc-er' not in mlModels:
            mlModels['gbc-er'] = pickle.load(open(mlModelFiles['gbc-er'], 'rb'))
        mentions = getGoodMentions(splitText, mentions, mlModels['gbc-er'], True)
    
    # filter out mentions
#     filters = []
#     with open('/users/cs/amaral/wikisim/wikification/mentions-filter.txt', 'r') as f:
#         for line in f:
#             filters.append(line.strip())
            
    goodMentions = []
    for mention in mentions:
            goodMentions.append(mention)
    
    return {'text':splitText, 'mentions':goodMentions}

def getMentionsInSentence(textData, mainWord):
    """
    Description:
        Finds all mentions that are in the same sentence as mainWord.
    Args:
        textData: A text in split form along with its suspected mentions.
        mainWord: The index of the word that is in the wanted sentence
    Return:
        A list of mention texts that are in the same sentence as mainWord
    """
    
    sents = nltk.sent_tokenize(" ".join(textData['text']))
    
    # start and end of sentences (absolute)
    sStart = 0
    sEnd = 0
    
    mentionStrs = [] # the mentions
    
    curEnd = 0
    for sent in sents:
        curEnd += len(sent)
        # if sentence ends after mention starts
        if curEnd > mainWord[1]:
            sEnd = curEnd
            sStart = sEnd - len(sent)
            mWIndex = textData['mentions'].index(mainWord) # index of mainWord
            
            # add every mention before main in sent to mentionsStr
            for i in range(mWIndex-1, -1, -1):
                if textData['mentions'][i][2] > sStart:
                    mentionStrs.append(textData['text'][textData['mentions'][i][0]])
                else:
                    break
                    
            # add every mention after main in sent to mentionsStr
            for i in range(mWIndex+1, len(textData['mentions'])):
                if textData['mentions'][i][1] < sEnd:
                    mentionStrs.append(textData['text'][textData['mentions'][i][0]])
                else:
                    break
            
            break
    
    return " ".join(mentionStrs).strip()

def generateCandidates(textData, maxC, hybrid = False):
    """
    Description:
        Generates up to maxC candidates for each possible mention word in phrase.
    Args:
        textData: A text in split form along with its suspected mentions.
        maxC: The max amount of candidates to accept.
        Hybrid: Whether to include best context fitting results too.
    Return:
        The top maxC candidates for each possible mention word in textData. Each 
        mentions has its candidates of the form: [(wikiId, popularity),...]
    """
    
    candidates = []
    
    ctxC0 = 0 # the amount of candidates to fill from best context.
    if hybrid == True:
        popC = int(maxC/2) + 1 # get ceil
        ctxC0 = maxC - popC
    else:
        popC = maxC
    
    for mention in textData['mentions']:
        resultT = sorted(anchor2concept(textData['text'][mention[0]]), key = itemgetter(1), 
                          reverse = True)[:popC]
        results = [list(item) for item in resultT]
        
        # get the right amount to fill with context 
        if len(results) < popC and hybrid == True:
            # fill in rest with context
            ctxC = maxC - len(results)
        elif hybrid == True:
            ctxC = ctxC0
        else:
            ctxC = 0
            
        # get some context results from solr
        if ctxC > 0:
            mentionStr = escapeStringSolr(textData['text'][mention[0]])
            ctxStr = escapeStringSolr(getMentionsInSentence(textData, mention))
            
            strIds = ['-id:' +  str(res[0]) for res in results]
            
            # select all the docs from Solr with the best scores, highest first.
            addr = 'http://localhost:8983/solr/enwiki20160305/select'

            try:
                tmp1 = mentionStr.encode('utf-8')
            except:
                tmp1 = mentionStr
            
            if len(ctxStr) >= 0:
                params={'fl':'id', 'indent':'on', 'fq':" ".join(strIds),
                        'q':'title:(' + tmp1+')^5',
                        'wt':'json', 'rows': str(ctxC)}
            else:
                
                try:
                    tmp2 = ctxStr.encode('utf-8')
                except:
                    tmp2 = ctxStr
                
                params={'fl':'id', 'indent':'on', 'fq':" ".join(strIds),
                        'q':'title:(' + tmp1 + ')^5'
                        + ' text:(' + tmp2 + ')',
                        'wt':'json', 'rows':str(ctxC)}
            
            r = requests.get(addr, params = params)
            try:
                if ('response' in r.json() 
                        and 'docs' in r.json()['response']
                        and len(r.json()['response']['docs']) > 0):
                    for doc in r.json()['response']['docs']:
                        # get popularity of entity given the mention
                        popularity = 0
                        thingys = anchor2concept(textData['text'][mention[0]])
                        for thingy in thingys:
                            if thingy[0] == long(doc['id']):
                                popularity = thingy[1]
                                break
                        
                        results.append([long(doc['id']), popularity])
            except:
                pass
            
        candidates.append(results[:maxC]) # take up to maxC of the results
    
    return candidates

def precision(truthSet, mySet):
    """
    Description:
        Calculates the precision of mySet against the truthSet.
    Args:
        truthSet: The 'right' answers for what the entities are. [[start,end,id],...]
        mySet: My code's output for what it thinks the right entities are. [[start,end,id],...]
    Return:
        The precision: (# of correct entities)/(# of found entities)
    """
    
    numFound = len(mySet)
    numCorrect = 0 # incremented in for loop
    
    truthIndex = 0
    myIndex = 0
    
    while truthIndex < len(truthSet) and myIndex < len(mySet):
        if mySet[myIndex][0] < truthSet[truthIndex][0]:
            if mySet[myIndex][1] > truthSet[truthIndex][0]:
                # overlap with mine behind
                if truthSet[truthIndex][2] == mySet[myIndex][2]:
                    numCorrect += 1
                    truthIndex += 1
                    myIndex += 1
                elif truthSet[truthIndex][1] < mySet[myIndex][1]:
                    # truth ends first
                    truthIndex += 1
                else:
                    # mine ends first
                    myIndex += 1
            else:
                # mine not even reach truth
                myIndex += 1
                
        elif mySet[myIndex][0] == truthSet[truthIndex][0]:
            # same mention (same start atleast)
            if truthSet[truthIndex][2] == mySet[myIndex][2]:
                numCorrect += 1
                truthIndex += 1
                myIndex += 1
            elif truthSet[truthIndex][1] < mySet[myIndex][1]:
                # truth ends first
                truthIndex += 1
            else:
                # mine ends first
                myIndex += 1
                  
        elif mySet[myIndex][0] > truthSet[truthIndex][0]:
            if mySet[myIndex][0] < truthSet[truthIndex][1]:
                # overlap with truth behind
                if truthSet[truthIndex][2] == mySet[myIndex][2]:
                    numCorrect += 1
                    truthIndex += 1
                    myIndex += 1
                elif truthSet[truthIndex][1] < mySet[myIndex][1]:
                    # truth ends first
                    truthIndex += 1
                else:
                    # mine ends first
                    myIndex += 1
            else:
                # mine beyond mention, increment truth
                truthIndex += 1

    #print 'correct: ' + str(numCorrect) + '\nfound: ' + str(numFound)
    if numFound == 0:
        return 0
    else:
        return (numCorrect/numFound)

def recall(truthSet, mySet):
    """
    Description:
        Calculates the recall of mySet against the truthSet.
    Args:
        truthSet: The 'right' answers for what the entities are. [[start,end,id],...]
        mySet: My code's output for what it thinks the right entities are. [[start,end,id],...]
    Return:
        The recall: (# of correct entities)/(# of actual entities)
    """
    
    numActual = len(truthSet)
    numCorrect = 0 # incremented in for loop)
    
    truthIndex = 0
    myIndex = 0
    
    while truthIndex < len(truthSet) and myIndex < len(mySet):
        if mySet[myIndex][0] < truthSet[truthIndex][0]:
            if mySet[myIndex][1] > truthSet[truthIndex][0]:
                # overlap with mine behind
                if truthSet[truthIndex][2] == mySet[myIndex][2]:
                    numCorrect += 1
                    truthIndex += 1
                    myIndex += 1
                elif truthSet[truthIndex][1] < mySet[myIndex][1]:
                    # truth ends first
                    truthIndex += 1
                else:
                    # mine ends first
                    myIndex += 1
            else:
                # mine not even reach truth
                myIndex += 1
                
        elif mySet[myIndex][0] == truthSet[truthIndex][0]:
            # same mention (same start atleast)
            if truthSet[truthIndex][2] == mySet[myIndex][2]:
                numCorrect += 1
                truthIndex += 1
                myIndex += 1
            elif truthSet[truthIndex][1] < mySet[myIndex][1]:
                # truth ends first
                truthIndex += 1
            else:
                # mine ends first
                myIndex += 1
                  
        elif mySet[myIndex][0] > truthSet[truthIndex][0]:
            if mySet[myIndex][0] < truthSet[truthIndex][1]:
                # overlap with truth behind
                if truthSet[truthIndex][2] == mySet[myIndex][2]:
                    numCorrect += 1
                    truthIndex += 1
                    myIndex += 1
                elif truthSet[truthIndex][1] < mySet[myIndex][1]:
                    # truth ends first
                    truthIndex += 1
                else:
                    # mine ends first
                    myIndex += 1
            else:
                # mine beyond mention, increment truth
                truthIndex += 1
                
    if numActual == 0:
        return 0
    else:
        return (numCorrect/numActual)
    
def mentionPrecision(trueMentions, otherMentions):
    """
    Description:
        Calculates the precision of otherMentions against the trueMentions.
    Args:
        trueMentions: The 'right' answers for what the mentions are.
        otherMentions: Our mentions obtained through some means.
    Return:
        The precision: (# of correct mentions)/(# of found mentions)
    """
    
    numFound = len(otherMentions)
    numCorrect = 0 # incremented in for loop
    
    trueIndex = 0
    otherIndex = 0
    
    while trueIndex < len(trueMentions) and otherIndex < len(otherMentions):
        # if mentions start and end on the same
        if (trueMentions[trueIndex][0] == otherMentions[otherIndex][0]
               and trueMentions[trueIndex][1] == otherMentions[otherIndex][1]):
            #print ('MATCH: [' + str(trueMentions[trueIndex][0]) + ',' + str(trueMentions[trueIndex][1]) + ']' + trueMentions[trueIndex][2] 
            #       + ' <===> [' + str(otherMentions[otherIndex][0]) + ',' + str(otherMentions[otherIndex][1]) + ']' + otherMentions[otherIndex][2])
            numCorrect += 1
            trueIndex += 1
            otherIndex += 1
        # if true mention starts before the other starts
        elif trueMentions[trueIndex][0] < otherMentions[otherIndex][0]:
            #print ('FAIL: [' + str(trueMentions[trueIndex][0]) + ',' + str(trueMentions[trueIndex][1]) + ']' + trueMentions[trueIndex][2] 
            #       + ' <XXX> [' + str(otherMentions[otherIndex][0]) + ',' + str(otherMentions[otherIndex][1]) + ']' + otherMentions[otherIndex][2])
            trueIndex += 1
        # if other mention starts before the true starts (same doesnt matter)
        elif trueMentions[trueIndex][0] >= otherMentions[otherIndex][0]:
            #print ('FAIL: [' + str(trueMentions[trueIndex][0]) + ',' + str(trueMentions[trueIndex][1]) + ']' + trueMentions[trueIndex][2] 
            #       + ' <XXX> [' + str(otherMentions[otherIndex][0]) + ',' + str(otherMentions[otherIndex][1]) + ']' + otherMentions[otherIndex][2])
            otherIndex += 1
        else:
            print 'AAAAAAAHHHHHHHHHHHHHHHHHHHHHHHHHHHHH!!!!!!!!!!!!!!!!!!!'

    #print 'correct: ' + str(numCorrect) + '\nfound: ' + str(numFound)
    if numFound == 0:
        return 0
    else:
        return (numCorrect/numFound)

def mentionRecall(trueMentions, otherMentions):
    """
    Description:
        Calculates the recall of otherMentions against the trueMentions.
    Args:
        trueMentions: The 'right' answers for what the mentions are.
        otherMentions: Our mentions obtained through some means.
    Return:
        The recall: (# of correct entities)/(# of actual entities)
    """
    
    numActual = len(trueMentions)
    numCorrect = 0 # incremented in for loop)
    
    trueIndex = 0
    otherIndex = 0
    
    while trueIndex < len(trueMentions) and otherIndex < len(otherMentions):
        # if mentions start and end on the same
        if (trueMentions[trueIndex][0] == otherMentions[otherIndex][0]
               and trueMentions[trueIndex][1] == otherMentions[otherIndex][1]):
            numCorrect += 1
            trueIndex += 1
            otherIndex += 1
        # if true mention starts before the other starts
        elif trueMentions[trueIndex][0] < otherMentions[otherIndex][0]:
            trueIndex += 1
        # if other mention starts before the true starts (same doesnt matter)
        elif trueMentions[trueIndex][0] >= otherMentions[otherIndex][0]:
            otherIndex += 1
        
    print 'correct: ' + str(numCorrect) + '\nactual: ' + str(numActual)
    if numActual == 0:
        return 0
    else:
        return (numCorrect/numActual)
    
def getSurroundingWords(text, mIndex, window, asList = False):
    """
    Description:
        Returns the words surround the given mention. Expanding out window elements
        on both sides.
    Args:
        text: A list of words.
        mIndex: The index of the word that is the center of where to get surrounding words.
        window: The amount of words to the left and right to get.
        asList: Whether to return the words as a list, otherwise just a string.
    Return:
        The words that surround the given mention. Expanding out window elements
        on both sides.
    """
    
    imin = mIndex - window
    imax = mIndex + window + 1
    
    # fix extreme bounds
    if imin < 0:
        imin = 0
    if imax > len(text):
        imax = len(text)
        
    if asList == True:
        words = (text[imin:mIndex] + text[mIndex+1:imax])
    else:
        words = " ".join(text[imin:mIndex] + text[mIndex+1:imax])
    
    # return surrounding part of word minus the mIndex word
    return words

def getMentionSentence(text, mention, asList = False):
    """
    Description:
        Returns the sentence of the mention, minus the mention.
    Args:
        text: The text to get the sentence from.
        index: The mention.
        asList: Whether to return the words as a list, otherwise just a string.
    Return:
        The sentence of the mention, minus the mention.
    """
    
    # the start and end indexes of the sentence
    sStart = 0
    sEnd = 0
    
    # get sentences using nltk
    sents = nltk.sent_tokenize(text)
    
    # find sentence that mention is in
    curLen = 0
    for s in sents:
        curLen += len(s)
        # if greater than begin of mention
        if curLen > mention[1]:
            # remove mention from string to not get bias from self referencing article
            if asList == True:
                sentence = (s.replace(text[mention[1]:mention[2]],"")).split(" ")
            else:
                sentence = s.replace(text[mention[1]:mention[2]],"")
            
            return sentence
        
    # in case it missed
    if asList == True:
        return []
    else:
        return ""

def escapeStringSolr(text):
    """
    Description:
        Escapes a given string for use in Solr.
    Args:
        text: The string to escape.
    Return:
        The escaped text.
    """
    
    text = text.replace("\\", "\\\\\\")
    text = text.replace('+', r'\+')
    text = text.replace("-", "\-")
    text = text.replace("&&", "\&&")
    text = text.replace("||", "\||")
    text = text.replace("!", "\!")
    text = text.replace("(", "\(")
    text = text.replace(")", "\)")
    text = text.replace("{", "\{")
    text = text.replace("}", "\}")
    text = text.replace("[", "\[")
    text = text.replace("]", "\]")
    text = text.replace("^", "\^")
    text = text.replace("\"", "\\\"")
    text = text.replace("~", "\~")
    text = text.replace("*", "\*")
    text = text.replace("?", "\?")
    text = text.replace(":", "\:")
    
    return text

def getContext1Scores(mentionStr, context, candidates):
    """
    Description:
        Uses Solr to find the relevancy scores of the candidates based on the context.
    Args:
        mentionStr: The mention as it appears in the text
        context: The words that surround the target word.
        candidates: A list of candidates that each have the entity id and its frequency/popularity.
    Return:
        The score for each candidate in the same order as the candidates.
    """
    
    candScores = []
    for i in range(len(candidates)):
        candScores.append(0)
    
    # put text in right format
    context = escapeStringSolr(context)
    mentionStr = escapeStringSolr(mentionStr)
    
    strIds = ['id:' +  str(strId[0]) for strId in candidates]
    
    try:
        tmp1 = context.encode('utf-8')
    except:
        tmp1 = context
        
    try:
        tmp2 = mentionStr.encode('utf-8')
    except:
        tmp2 = mentionStr
    
    # select all the docs from Solr with the best scores, highest first.
    addr = 'http://localhost:8983/solr/enwiki20160305/select'
    params={'fl':'id score', 'fq':" ".join(strIds), 'indent':'on',
            'q':'text:('+tmp1+')^1 title:(' + tmp2+')^1.35',
            'wt':'json'}
    r = requests.get(addr, params = params)
    
    try:
        # assign the scores
        for doc in r.json()['response']['docs']:
            # find candidate of doc
            i = 0
            for cand in candidates:
                if cand[0] == long(doc['id']):
                    candScores[i] = doc['score']
                    break
                i += 1
    except:
        # keep zero scores
        pass
            
    return candScores

def bestContext1Match(mentionStr, context, candidates):
    """
    Description:
        Uses Solr to find the candidate that gives the highest relevance when given the context.
    Args:
        mentionStr: The mention as it appears in the text
        context: The words that surround the target word.
        candidates: A list of candidates that each have the entity id and its frequency/popularity.
    Return:
        The index of the candidate with the best relevance score from the context.
    """
    
    # put text in right format
    context = escapeStringSolr(context)
    mentionStr = escapeStringSolr(mentionStr)
    
    strIds = ['id:' +  str(strId[0]) for strId in candidates]
    
    try:
        tmp1 = context.encode('utf-8')
    except:
        tmp1 = context
        
    try:
        tmp2 = mentionStr.encode('utf-8')
    except:
        tmp2 = mentionStr
    
    # select all the docs from Solr with the best scores, highest first.
    addr = 'http://localhost:8983/solr/enwiki20160305/select'
    params={'fl':'id score', 'fq':" ".join(strIds), 'indent':'on',
            'q':'text:('+tmp1+')^1 title:(' + tmp2+')^1.35',
            'wt':'json'}
    r = requests.get(addr, params = params)
    
    if 'response' not in r.json():
        return 0 # default to most popular
    
    if 'docs' not in r.json()['response']:
        return 0
    
    results = r.json()['response']['docs']
    if len(results) == 0:
        return 0 # default to most popular
    
    bestId = long(r.json()['response']['docs'][0]['id'])
    
    #for doc in r.json()['response']['docs']:
        #print '[' + id2title(doc['id']) + '] -> ' + str(doc['score'])
    
    # find which index has bestId
    bestIndex = 0
    for cand in candidates:
        if cand[0] == bestId:
            return bestIndex
        else:
            bestIndex += 1
            
    return bestIndex # in case it was missed

def getContext2Scores(mentionStr, context, candidates):
    """
    Description:
        Uses Solr to find the relevancy scores of the candidates based on the context.
    Args:
        mentionStr: The mention as it appears in the text
        context: The words that surround the target word.
        candidates: A list of candidates that each have the entity id and its frequency/popularity.
    Return:
        The score for each candidate in the same order as the candidates.
    """
    
    candScores = []
    for i in range(len(candidates)):
        candScores.append(0)
    
    # put text in right format
    context = escapeStringSolr(context)
    mentionStr = escapeStringSolr(mentionStr)
    
    strIds = ['entityid:' +  str(strId[0]) for strId in candidates]
    
    # dictionary to hold scores for each id
    scoreDict = {}
    for cand in candidates:
        scoreDict[str(cand[0])] = 0
        
    try:
        tmp1 = context.encode('utf-8')
    except:
        tmp1 = context
        
    try:
        tmp2 = mentionStr.encode('utf-8')
    except:
        tmp2 = mentionStr
    
    # select all the docs from Solr with the best scores, highest first.
    addr = 'http://localhost:8983/solr/enwiki20160305_context/select'
    params={'fl':'entityid', 'fq':" ".join(strIds), 'indent':'on',
            'q':'_context_:('+tmp1+') entity:(' + tmp2 + ')^1',
            'wt':'json'}
    r = requests.get(addr, params = params)
    
    try:
        # get count for each id
        for doc in r.json()['response']['docs']:
            scoreDict[str(doc['entityid'])] += 1
    except:
        # keep zero scores
        pass
    
    # give scores to each cand
    for j in range(0, len(candidates)):
        candScores[j] = scoreDict[str(candidates[j][0])]
            
    return candScores

def bestContext2Match(mentionStr, context, candidates):
    """
    Description:
        Uses Solr to find the candidate that gives the highest relevance when given the context.
    Args:
        context: The words that surround the target word.
        candidates: A list of candidates that each have the entity id and its frequency/popularity.
    Return:
        The index of the candidate with the best relevance score from the context.
    """
    
    # put text in right format
    context = escapeStringSolr(context)
    mentionStr = escapeStringSolr(mentionStr)
    strIds = ['entityid:' +  str(strId[0]) for strId in candidates]
    
    # dictionary to hold scores for each id
    scoreDict = {}
    for cand in candidates:
        scoreDict[str(cand[0])] = 0
        
    try:
        tmp1 = context.encode('utf-8')
    except:
        tmp1 = context
        
    try:
        tmp2 = mentionStr.encode('utf-8')
    except:
        tmp2 = mentionStr
    
    # select all the docs from Solr with the best scores, highest first.
    addr = 'http://localhost:8983/solr/enwiki20160305_context/select'
    params={'fl':'entityid', 'fq':" ".join(strIds), 'indent':'on',
            'q':'_context_:('+tmp1+') entity:(' + tmp2 + ')^1',
            'wt':'json'}
    r = requests.get(addr, params = params)
    
    if 'response' not in r.json():
        return 0 # default to most popular
    
    if 'docs' not in r.json()['response']:
        return 0
    
    results = r.json()['response']['docs']
    if len(results) == 0:
        return 0 # default to most popular
    
    for doc in r.json()['response']['docs']:
        scoreDict[str(doc['entityid'])] += 1
    
    # get the index that has the best score
    bestScore = 0
    bestIndex = 0
    curIndex = 0
    for cand in candidates:
        if scoreDict[str(cand[0])] > bestScore:
            bestScore = scoreDict[str(cand[0])]
            bestIndex = curIndex
        curIndex += 1
            
    return bestIndex

def getWord2VecScores(context, candidates):
    """
    Description:
        Uses word2vec to find the similarity scores of each mention to the context vector.
    Args:
        context: The words that surround the target word as a list.
        candidates: A list of candidates that each have the entity id and its frequency/popularity.
    Return:
        The scores of eac candidate.
    """
    
    candScores = []
    for i in range(len(candidates)):
        candScores.append(0)
        
    ctxVec = pd.Series(sp.zeros(300)) # default zero vector
    # add all context words together
    for word in context:
        ctxVec += getword2vector(word)
        
    # compare context vector to each of the candidates
    i = 0
    for cand in candidates:
        eVec = getentity2vector(str(cand[0]))
        score = 1-sp.spatial.distance.cosine(ctxVec, eVec)
        if math.isnan(score):
            score = 0
        candScores[i] = score
        i += 1 # next index
        
    return candScores

def bestWord2VecMatch(context, candidates):
    """
    Description:
        Uses word2vec to find the candidate with the best similarity to the context.
    Args:
        context: The words that surround the target word as a list.
        candidates: A list of candidates that each have the entity id and its frequency/popularity.
    Return:
        The index of the candidate with the best similarity score with the context.
    """
    
    ctxVec = pd.Series(sp.zeros(300)) # default zero vector
    # add all context words together
    for word in context:
        ctxVec += getword2vector(word)
        
    # compare context vector to each of the candidates
    bestIndex = 0
    bestScore = 0
    i = 0
    for cand in candidates:
        eVec = getentity2vector(str(cand[0]))
        score = 1-sp.spatial.distance.cosine(ctxVec, eVec)
        #print '[' + id2title(cand[0]) + ']' + ' -> ' + str(score)
        # update score and index
        if score > bestScore: 
            bestIndex = i
            bestScore = score
            
        i += 1 # next index
            
    return bestIndex
    
def wikifyPopular(textData, candidates):
    """
    Description:
        Chooses the most popular candidate for each mention.
    Args:
        textData: A text in split form along with its suspected mentions.
        candidates: A list of list of candidates that each have the entity id and its frequency/popularity.
    Return:
        All of the proposed entities for the mentions, of the form: [[start,end,entityId],...].
    """
    
    topCandidates = []
    i = 0 # track which mention's candidates we are looking at
    # for each mention choose the top candidate
    for mention in textData['mentions']:
        if len(candidates[i]) > 0:
            topCandidates.append([mention[1], mention[2], candidates[i][0][0]])
        i += 1 # move to list of candidates for next mention
            
    return topCandidates

def wikifyContext(textData, candidates, oText, useSentence = False, window = 7, method2 = False):
    """
    Description:
        Chooses the candidate that has the highest relevance with the surrounding window words.
    Args:
        textData: A textData in split form along with its suspected mentions.
        candidates: A list of candidates that each have the entity id and its frequency/popularity.
        oText: The original text to be used for getting sentence.
        useSentence: Whether to set use whole sentence as context, or just windowsize.
        window: How many words on both sides of a mention to search for context.
    Return:
        All of the proposed entities for the mentions, of the form: [[start,end,entityId],...].
    """
    
    topCandidates = []
    i = 0 # track which mention's candidates we are looking at
    # for each mention choose the top candidate
    for mention in textData['mentions']:
        if len(candidates[i]) > 0:
            if not useSentence:
                context = getSurroundingWords(textData['text'], mention[0], window)
            else:
                #context = getMentionSentence(oText, mention)
                context = getMentionsInSentence(textData, mention)
            #print '\nMention: ' + textData['text'][mention[0]]
            #print 'Context: ' + context
            if method2 == False:
                bestIndex = bestContext1Match(textData['text'][mention[0]], context, candidates[i])
            else:
                bestIndex = bestContext2Match(textData['text'][mention[0]], context, candidates[i])
            topCandidates.append([mention[1], mention[2], candidates[i][bestIndex][0]])
        i += 1 # move to list of candidates for next mention
        
    return topCandidates

def wikifyWord2Vec(textData, candidates, oText, useSentence = False, window = 5):
    """
    Description:
        Chooses the candidates that have the highest similarity to the context.
    Args:
        textData: A textData in split form along with its suspected mentions.
        candidates: A list of candidates that each have the entity id and its frequency/popularity.
        oText: The original text to be used for getting sentence.
        useSentence: Whether to set use whole sentence as context, or just windowsize.
        window: How many words on both sides of a mention to search for context.
    Return:
        All of the proposed entities for the mentions, of the form: [[start,end,entityId],...].
    """
    
    topCandidates = []
    i = 0 # track which mention's candidates we are looking at
    # for each mention choose the top candidate
    for mention in textData['mentions']:
        if len(candidates[i]) > 0:
            if not useSentence:
                context = getSurroundingWords(textData['text'], mention[0], window, asList = True)
            else:
                context = getMentionSentence(oText, mention, asList = True)
            #print '\nMention: ' + textData['text'][mention[0]]
            #print 'Context: ' + " ".join(context)
            bestIndex = bestWord2VecMatch(context, candidates[i])
            topCandidates.append([mention[1], mention[2], candidates[i][bestIndex][0]])
        i += 1 # move to list of candidates for next mention
        
    return topCandidates

def wikifyCoherence(textData, candidates, ws = 5):
    """
    Description:
        Chooses the candidates that have the highest coherence according to rvs pagerank method.
    Args:
        textData: A textData in split form along with its suspected mentions.
        candidates: A list of candidates that each have the entity id and its frequency/popularity.
        ws: How many words on both sides of a mention to search for context.
    Return:
        All of the proposed entities for the mentions, of the form: [[start,end,entityId],...].
    """
    
    topCands = [] # the top candidate from each candidate list
    candsScores = coherence_scores_driver(candidates, ws, method='rvspagerank', direction=DIR_BOTH, op_method="keydisamb")
    i = -1 # track what mention we are on
    for cScores in candsScores:
        i += 1
        
        if len(cScores) == 0:
            continue # nothing to do with this one
            
        bestScore = sorted(cScores, reverse = True)[0]
        curIndex = 0
        for score in cScores:
            if score == bestScore:
                topCands.append([textData['mentions'][i][1], textData['mentions'][i][2], candidates[i][curIndex][0]])
                break
            curIndex += 1
            
    return topCands

mlModels = {} # dictionary of different models
mlModelFiles = {
    'abc': '/users/cs/amaral/wikisim/wikification/ml-models/model-abc-10000-hyb.pkl',
    'bgc': '/users/cs/amaral/wikisim/wikification/ml-models/model-bgc-10000-hyb.pkl',
    'etc': '/users/cs/amaral/wikisim/wikification/ml-models/model-etc-10000-hyb.pkl',
    'gbc': '/users/cs/amaral/wikisim/wikification/ml-models/model-gbc-10000-hyb.pkl',
    'rfc': '/users/cs/amaral/wikisim/wikification/ml-models/model-rfc-10000-hyb.pkl',
    'lsvc': '/users/cs/amaral/wikisim/wikification/ml-models/model-lsvc-10000-hyb.pkl',
    'svc': '/users/cs/amaral/wikisim/wikification/ml-models/model-svc-10000-hyb.pkl',
    'lmart': '/users/cs/amaral/wikisim/wikification/ml-models/model-lmart-10000-pop-no-w2v.pkl',
    'gbc-er': '/users/cs/amaral/wikisim/wikification/ml-models/er/er-model-gbc-30000.pkl',
    'bgc-er': '/users/cs/amaral/wikisim/wikification/ml-models/er/er-model-bgc-30000.pkl'}

def wikifyMulti(textData, candidates, oText, model, useSentence = True, window = 7):
    """
    Description:
        Disambiguates each of the mentions with their given candidates using the desired
        machine learned model.
    Args:
        textData: A textData in split form along with its suspected mentions.
        candidates: A list of candidates that each have the entity id and its frequency/popularity.
        oText: The original text, unsplit.
        model: The machine learned model to use for disambiguation: 
            'gbc' (gradient boosted classifier), 'etr' (extra trees regression), 
            'gbr' (gradient boosted regression), 'lmart' (LambdaMART (a learning to rank method)),
            and 'rfr' (random forest regression).
        useSentence: Whether to use windo size of sentence (for context methods)
        window: How many words on both sides of a mention to search for context.
    Return:
        All of the proposed entities for the mentions, of the form: [[start,end,entityId],...].
    """
    
    mlModel = mlModels[model] # get reference to model
    
    # get score from coherence
    cohScores = coherence_scores_driver(candidates, 5, method='rvspagerank', direction=DIR_BOTH, op_method="keydisamb")
    
    i = 0
    # get scores from each disambiguation method for all mentions
    for mention in textData['mentions']:
        if len(candidates[i]) > -1: # stub
            # get the scores from each basic method.
            
            # normalize popularity scores
            cScrs = []
            for cand in candidates[i]:
                cScrs.append(cand[1])
            cScrs = normalize(cScrs)
            j = 0
            for cand in candidates[i]:
                cand[1] = cScrs[j]
                j += 1
            
            contextMInS = getMentionsInSentence(textData, textData['mentions'][i])
            contextS = getMentionSentence(oText, textData['mentions'][i], asList = True)
            
            # context 1 scores
            cScrs = getContext1Scores(textData['text'][mention[0]], contextMInS, candidates[i])
            cScrs = normalize(cScrs)
            # apply score to candList
            for j in range(0, len(candidates[i])):
                candidates[i][j].append(cScrs[j])
            
            # context 2 scores
            cScrs = getContext2Scores(textData['text'][mention[0]], contextMInS, candidates[i])
            cScrs = normalize(cScrs)
            # apply score to candList
            for j in range(0, len(candidates[i])):
                candidates[i][j].append(cScrs[j])

            # get score from word2vec
            #cScrs = getWord2VecScores(contextS, candidates[i])
            #cScrs = normalize(cScrs)
            # apply score to candList
            for j in range(0, len(candidates[i])):
                #candidates[i][j].append(cScrs[j])
                candidates[i][j].append(0)

            # get score from coherence
            for j in range(0, len(candidates[i])):
                candidates[i][j].append(cohScores[i][j])
            
        i += 1
    
    topCandidates = []
    
    i = 0
    # go through all mentions again to disambiguate with ml model
    for mention in textData['mentions']:
        try:
            Xs = [cand[1:] for cand in candidates[i]]
            if len(Xs) == 0:
                i += 1
                continue
            pred = mlModel.predict(Xs)
        except:
            try:
                Xs = [cand[1:] for cand in candidates[i]]
                pred = mlModel.predict(np.array(candidates[i][1:]).reshape(1, -1))
            except:
                i += 1
                continue
        cur = 0
        best = 0
        bestI = 0
        for j in range(len(pred)):
            if pred[j] > best:
                best = pred[j]
                bestI = j
        
        topCandidates.append([mention[1], mention[2], candidates[i][bestI][0]])
        
        i += 1
        
    return topCandidates

def wikifyEval(text, mentionsGiven, maxC = 20, method='popular', 
               strict = False, hybridC = True, model = 'lmart', erMethod = 'cls1'):
    """
    Description:
        Takes the text (maybe text data), and wikifies it for evaluation purposes using the desired method.
    Args:
        text: The string to wikify. Either as just the original string to be modified, or in the 
            form of: [[w1,w2,...], [[wid,entityId],...] if the mentions are given.
        mentionsGiven: Whether the mentions are given to us and the text is already split.
        maxC: The max amount of candidates to extract.
        method: The method used to wikify.
        strict: Whether to use such rules as minimum metion length, or minimum frequency of concept.
        hybridC: Whether to split generated candidates between best of most frequent of most context related.
        model: What model to use if using machine learning based method. LambdaMART as 'lmart' is default.
            Other options are: 'gbc' (gradient boosted classifier), 'etr' (extra trees regression), 
            'gbr' (gradient boosted regression), and 'rfr' (random forest regression).
        erMethod: The method to use for ER. 'cls1', 'cls2', 'cnlp'.
    Return:
        All of the proposed entities for the mentions, of the form: [[start,end,entityId],...].
    """
    
    if not(mentionsGiven): # if words are not in pre-split form
        text = text.replace(u'\u2010', '-')
        text = text.replace(u'\u2011', '-')
        text = text.replace(u'\u2012', '-')
        text = text.replace(u'\u2013', '-')
        text = text.replace(u'\u2014', '-')
        text = text.replace(u'\u2015', '-')
        textData = mentionExtract(text, mthd = erMethod) # extract mentions from text
        oText = text # the original text
    else: # if they are
        textData = text
        textData['mentions'] = mentionStartsAndEnds(textData) # put mentions in right form
        oText = " ".join(text['text'])
    
    # get rid of small mentions
    if strict:
        textData['mentions'] = [item for item in textData['mentions']
                    if  len(textData['text'][item[0]]) >= MIN_MENTION_LENGTH]
    
    if method == 'popular':
        maxC = 1 # only need one cand for popular
    
    candidates = generateCandidates(textData, maxC, hybridC)
    
    if method == 'popular':
        wikified = wikifyPopular(textData, candidates)
    elif method == 'context1':
        wikified = wikifyContext(textData, candidates, oText, useSentence = True, window = 7)
    elif method == 'context2':
        wikified = wikifyContext(textData, candidates, oText, useSentence = True, window = 7, method2 = True)
    elif method == 'word2vec':
        wikified = wikifyWord2Vec(textData, candidates, oText, useSentence = False, window = 5)
    elif method == 'coherence':
        wikified = wikifyCoherence(textData, candidates, ws = 5)
    elif method == 'multi':
        if model not in mlModels:
            mlModels[model] = pickle.load(open(mlModelFiles[model], 'rb'))
        wikified = wikifyMulti(textData, candidates, oText, model, useSentence = True, window = 7)
    
    # get rid of very unpopular mentions
    if strict:
        wikified = [item for item in wikified
                    if item[3] >= MIN_FREQUENCY]
    
    return wikified

def doWikify(text, maxC = 20, hybridC = False, method = 'multi', erMethod = 'cls2'):
    """
    Description:
        Takes in text, and returns the location of mentions as well as the
        entities they refer to.
    Args:
        text: The text to be wikified.
    Return:
        A list of mentions where each element contains the character offset
        start and end, as well as the corresponding wikipedia page.
    """
    # find the mentions
    # text data now has text in split form and, the mentions
    textData = mentionExtract(text, mthd = erMethod)
    
    
    # generate candidates
    candidates = generateCandidates(textData, maxC, hybridC)
    
    # disambiguate each mention to its candidates
    if method == 'popular':
        wikified = wikifyPopular(textData, candidates)
    elif method == 'context1':
        wikified = wikifyContext(textData, candidates, text, useSentence = True, window = 7)
    elif method == 'context2':
        wikified = wikifyContext(textData, candidates, text, useSentence = True, window = 7, method2 = True)
    elif method == 'word2vec':
        try:
            word2vec
        except:
            word2vec = gensim_loadmodel('/users/cs/amaral/cgmdir/WikipediaClean5Negative300Skip10.Ehsan/WikipediaClean5Negative300Skip10')
        wikified = wikifyWord2Vec(textData, candidates, text, useSentence = False, window = 5)
    elif method == 'coherence':
        wikified = wikifyCoherence(textData, candidates, ws = 5)
    elif method == 'multi':
        try:
            #word2vec
            pass
        except:
            #word2vec = gensim_loadmodel('/users/cs/amaral/cgmdir/WikipediaClean5Negative300Skip10.Ehsan/WikipediaClean5Negative300Skip10')
            pass
        if 'lmart' not in mlModels:
            mlModels['lmart'] = pickle.load(open(mlModelFiles['lmart'], 'rb'))
        wikified = wikifyMulti(textData, candidates, text, 'lmart', useSentence = True, window = 7)
    
    return wikified

def annotateText(text, maxC = 20, hybridC = False, method = 'multi', erMethod = 'cls2'):
    """
    Description:
        Annotates text with html anchor tags linking to the wikipedia pages
        of the suspected entities.
    Args:
        text: The text to be annotated.
    Return:
        The text where the mentions are in anchor tags that link to the 
        corresponding wikipedia page.
    """
    
    # get the annotations
    ants = doWikify(text, maxC = maxC, hybridC = hybridC, method = method, erMethod = erMethod)
    
    # get title and intro of each entity
    strIds = ['id:' +  str(ant[2]) for ant in ants]
    addr = 'http://localhost:8983/solr/enwiki20160305/select'
    params={'fl':'title opening_text id', 'fq':" ".join(strIds), 
            'indent':'on', 'q':'*:*', 'wt':'json', 'rows':str(len(ants))}
    r = requests.get(addr, params = params)
    try:
        for doc in r.json()['response']['docs']:
            # find ant with same id as doc
            for ant in ants:
                if len(ant) == 3 and str(ant[2]) == doc['id']:
                    ant.extend([doc['title'], doc['opening_text'][:250] + '...'])
    except:
        pass
    
    # fill in all unfilled ants with thing
    for ant in ants:
        if len(ant) == 3:
            ant.extend([id2title(ant[2]).replace('_', ' '), 'Description Not Found.'])
            tmp = ant[3]
            ant[3] = ''.join([i if ord(i) < 128 else '' for i in tmp]) # filter out non ascii, https://stackoverflow.com/questions/20078816/replace-non-ascii-characters-with-a-single-space
    
    newText = '' # the text to return with anchor tags
    skip = 0
    curM = 0 # cur mention index
    for i in range(len(text)):
        if skip > 0:
            skip -= 1
            continue
        if curM < len(ants) and i == ants[curM][0]:
            skip = ants[curM][1] - ants[curM][0] - 1
            newText += ('<a class="toooltip" target="_blank" href="https://en.wikipedia.org/wiki/'
                       + id2title(ants[curM][2]) + '">' 
                       + text[ants[curM][0]:ants[curM][1]] 
                       + '<span class="toooltiptext"><strong>' + ants[curM][3].encode('utf-8') + '</strong><br/>' 
                       + ants[curM][4].encode('utf-8')
                       + '</span></a>')
            curM += 1
        else:
            newText += text[i]
            
    return newText
    