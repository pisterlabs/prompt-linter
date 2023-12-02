# -*- coding:utf-8 -*-
'''
Created on Apr 16, 2018

@author: yingc
'''

from gensim import corpora, models, similarities
from pprint import pprint
import matplotlib.pyplot as plt
import math
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
import numpy as np

from gensim.models.coherencemodel import CoherenceModel

import logging
from textblob.classifiers import _get_document_tokens

#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

documents0 = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
            "The EPS user interface management system",
            "System and human system engineering testing of EPS",
            "Relation of user perceived response time to error measurement",
            "The generation of random binary unordered trees",
            "The intersection graph of paths in trees",
            "Graph minors IV Widths of trees and well quasi ordering",
            "Graph minors A survey"]

documents1 = ["The concept of the observable universe. The fact that the reason we can't see a certain range into the space being because the light hasn't had time to get to earth since the beginning of the universe is crazy to me.",
             "So you mean the universe is buffering for us?",
             "Wow, now your analogy blew my mind!",
             "I want it now godamit! gluurrraAA grrraAAA",
             "Fucking pop-up.",
             "Nah it's more like the draw distance.",
             "this comment literally made the whole observable universe thing actually make sense to me for the first time. cheers.",
             "Your comment just blew my mind into milky way chunks.",
             "Oh. Damn.",
             "Holy shit o.o",
             "I guarantee the universe is gonna put itself behind a paywall very soon",
             "There is an horizon beyond which we will never be able to see no matter how long the universe runs for. It is one of the unsolved cosmological problems. If there are boundaries beyond which no information will ever pass then how did the universe end up homogeneous?",
             "Not really."]

documents2 = ["Holy shit is that what that means? I never gave it much thought but always assumed 'observable universe' just to be the furthest we can accurately see before the information becomes too small or distorted by the great distance.",
              "Its even crazier than that. Due to the expansion of the Universe, everthing outside the observable Universe is now moving faster than the speed of light away from us. That means that the light from the rest of the Universe will never reach us. We live in an ever shrinking bubble of local galaxies. Everything around us will literally fade out of existence (since by definition, if you can't ever observe it, it doesn't exist) as the relative speed between us and the rest of the galaxies passes the speed of light. EDIT There is a ton of responses to this thread. I've posted this link elsewhere, but I figured I'd put here as well. It explained this way, way better than I could ever hope to.https://www.youtube.com/watch?v=XBr4GkRnY04 There are differences between what we can see, how far away those objects currently are, how long its been going on, how wide our visible universe is, etc. But the basic point is the same. Outside some radius about our place in the universe, the rest of the universe is expanding away from us greater than the speed of light. Not only will the light from that part of the universe never reach us, we can never reach them. We are forever isolated in our bubble. If you think about the simulation theory of the universe, it's an ingenious way to contain us without having walls.",
              "This baffles me. My knowledge of this stuff is severely limited to things I've only been told/ read/ watched, but I remember on this one episode of Cosmos (the new one), NDT mentioned that nothing could ever go faster than light. I think the situation they portrayed was if you could travel to 99.9999~% the speed of light on a bike, then switch on the headlight, the light leaving the headlight would still be traveling at the speed of light. Since you brought this up though, I was wondering about it as well. If the universe is expanding more rapidly, could it expand at such a rate where light couldn't overcome the rate of expansion? And if it could, what happens to the light traveling in the opposite direction? I mean if I'm in a car going at 25 mph and throw a ball out the window going 25 mph in the opposite direction, it'd appear like the ball is standing still to someone on the outside, right (not taking gravity into account)? So could light theoretically be standing still somewhere in the universe? I'm sorry for the babbling on my part, but this screws with my mind in all sorts of ways. EDIT: Holy expletive, this is why I love the reddit community. Thanks for all the helpful answers everyone!",
              "The galaxies and other things that are 'moving faster than the speed of light away from us' are not moving through space/time faster than the speed of light, but are moving that fast because of space/time itself expanding. The individual stars, planets, asteroids and such aren't moving through space at a faster than light speed, the very fabric of space/time is expanding faster than light. Although I'm not entirely sure if space actually IS expanding that fast right now, I just know that it is continually increasing its rate of expansion and will eventually (if it hasn't already) break that barrier. So the 'nothing travels faster than light' rule still holds, because that rule is talking about things moving through space, not space itself. Hopefully I explained that adequately.",
              "Very informative and detailed answer. I think I understand your explanation of space being the container, which is itself expanding. The light, or contents in the container still adhere to the rules of the inside of the container, but different rules apply to the container itself? Sorry I keep reverting to comparisons, only way I can sort of make sense of things.",
              "Yeah, you pretty much have it. The analogy that gets used lots is to blow up a balloon and draw dots on it. With the dots representing galaxies and the balloon surface representing space itself. If you blow the balloon up further, it expands, and the dots (galaxies) get farther away from one another. However, the dots themselves haven't actually moved."
              ]

d1 = ["The concept of the observable universe. The fact that the reason we can't see a certain range into the space being because the light hasn't had time to get to earth since the beginning of the universe is crazy to me. So you mean the universe is buffering for us? Wow, now your analogy blew my mind! I want it now godamit! gluurrraAA grrraAAA. Fucking pop-up.Nah it's more like the draw distance. this comment literally made the whole observable universe thing actually make sense to me for the first time. cheers.Your comment just blew my mind into milky way chunks. Oh. Damn. Holy shit o.o I guarantee the universe is gonna put itself behind a paywall very soon There is an horizon beyond which we will never be able to see no matter how long the universe runs for. It is one of the unsolved cosmological problems. If there are boundaries beyond which no information will ever pass then how did the universe end up homogeneous? Not really."]

d2 = ["Holy shit is that what that means? I never gave it much thought but always assumed 'observable universe' just to be the furthest we can accurately see before the information becomes too small or distorted by the great distance. Its even crazier than that. Due to the expansion of the Universe, everthing outside the observable Universe is now moving faster than the speed of light away from us. That means that the light from the rest of the Universe will never reach us. We live in an ever shrinking bubble of local galaxies. Everything around us will literally fade out of existence (since by definition, if you can't ever observe it, it doesn't exist) as the relative speed between us and the rest of the galaxies passes the speed of light. EDIT There is a ton of responses to this thread. I've posted this link elsewhere, but I figured I'd put here as well. It explained this way, way better than I could ever hope to.https://www.youtube.com/watch?v=XBr4GkRnY04 There are differences between what we can see, how far away those objects currently are, how long its been going on, how wide our visible universe is, etc. But the basic point is the same. Outside some radius about our place in the universe, the rest of the universe is expanding away from us greater than the speed of light. Not only will the light from that part of the universe never reach us, we can never reach them. We are forever isolated in our bubble. If you think about the simulation theory of the universe, it's an ingenious way to contain us without having walls. This baffles me. My knowledge of this stuff is severely limited to things I've only been told/ read/ watched, but I remember on this one episode of Cosmos (the new one), NDT mentioned that nothing could ever go faster than light. I think the situation they portrayed was if you could travel to 99.9999~% the speed of light on a bike, then switch on the headlight, the light leaving the headlight would still be traveling at the speed of light. Since you brought this up though, I was wondering about it as well. If the universe is expanding more rapidly, could it expand at such a rate where light couldn't overcome the rate of expansion? And if it could, what happens to the light traveling in the opposite direction? I mean if I'm in a car going at 25 mph and throw a ball out the window going 25 mph in the opposite direction, it'd appear like the ball is standing still to someone on the outside, right (not taking gravity into account)? So could light theoretically be standing still somewhere in the universe? I'm sorry for the babbling on my part, but this screws with my mind in all sorts of ways. EDIT: Holy expletive, this is why I love the reddit community. Thanks for all the helpful answers everyone! The galaxies and other things that are 'moving faster than the speed of light away from us' are not moving through space/time faster than the speed of light, but are moving that fast because of space/time itself expanding. The individual stars, planets, asteroids and such aren't moving through space at a faster than light speed, the very fabric of space/time is expanding faster than light. Although I'm not entirely sure if space actually IS expanding that fast right now, I just know that it is continually increasing its rate of expansion and will eventually (if it hasn't already) break that barrier. So the 'nothing travels faster than light' rule still holds, because that rule is talking about things moving through space, not space itself. Hopefully I explained that adequately. Very informative and detailed answer. I think I understand your explanation of space being the container, which is itself expanding. The light, or contents in the container still adhere to the rules of the inside of the container, but different rules apply to the container itself? Sorry I keep reverting to comparisons, only way I can sort of make sense of things. Yeah, you pretty much have it. The analogy that gets used lots is to blow up a balloon and draw dots on it. With the dots representing galaxies and the balloon surface representing space itself. If you blow the balloon up further, it expands, and the dots (galaxies) get farther away from one another. However, the dots themselves haven't actually moved."
              ]

documents3 = ["Texas serial bomber made video confession before blowing himself up",
              "What are the chances we ever see the video?",
              "About the same as the chances of the Browns winning the Super Bowl.",
              "I take the browns to the super bowl every morning.",
              "I have to applaud your regularity",
              "I thought at first you meant he posts that comment regularly. But now I get it. Healthy colon.",
              "Pshh I'm taking the browns to the super bowl as we speak",
              "Consistency is the key.",
              "Seriously. Well done.",
              "Zero, videos like this are locked down and used for training purposes. There are a host of confessions and tapes of crimes the public will never see and some have caused agents in training to kill themselves because they are so vile.",
              "Holy fuck, here I am thinking 'just transcripts? How bad can it be' Bad, guys. Very fucking bad.",
              "I want to know what kind of phone he has. I have had one break from a 3 foot fall, and his survived a fucking explosion?!",
              "Nokia brick",
              "God those old analog phones from the 90's were amazingly durable. They also had great reception (Way better than what I have now).",
              "Yes but the old phones had the drawback of having to be charged every two weeks."
              
    ]

documents33 = ["Texas serial bomber made video confession before blowing himself up",
              "What are the chances we ever see the video?",
              "About the same as the chances of the Browns winning the Super Bowl.",
              "I take the browns to the super bowl every morning.",
              "I have to applaud your regularity",
              "I thought at first you meant he posts that comment regularly. But now I get it. Healthy colon.",
              "Pshh I'm taking the browns to the super bowl as we speak",
              "Consistency is the key.",
              "Seriously. Well done.",
              "Zero, videos like this are locked down and used for training purposes. There are a host of confessions and tapes of crimes the public will never see and some have caused agents in training to kill themselves because they are so vile.",
              "here I am thinking 'just transcripts? How bad can it be' Bad, guys. Very bad.",
              "I want to know what kind of phone he has. I have had one break from a 3 foot fall, and his survived an explosion?!",
              "Nokia brick",
              "God those old analog phones from the 90's were amazingly durable. They also had great reception (Way better than what I have now).",
              "Yes but the old phones had the drawback of having to be charged every two weeks."
              
    ]

documents3_normal = ["Texas serial bomber made video confession before blowing himself up",
              "What are the chances we ever see the video?",
              "About the same as the chances of the Browns winning the Super Bowl.",
              "every morning.",
              "I have to applaud your regularity",
              "I thought at first you meant he posts that comment regularly. But now I get it. Healthy colon.",
              "Pshh I'm taking the browns to the super bowl as we speak",
              "Consistency is the key.",
              "Seriously. Well done.",
              "Zero, videos like this are locked down and used for training purposes. There are a host of confessions and tapes of crimes the public will never see and some have caused agents in training to kill themselves because they are so vile.",
              "here I am thinking 'just transcripts? How bad can it be' Bad, guys. Very bad.",
              "I want to know what kind of phone he has. I have had one break from a 3 foot fall, and his survived an explosion?!",
              "Nokia brick",
              "God those old analog phones from the 90's were amazingly durable. They also had great reception (Way better than what I have now).",
              "Yes but the old phones had the drawback of having to be charged every two weeks."
              
    ]

documents333 = ["Texas serial bomber made video confession before blowing himself up",
              "Texas serial bomber made video confession before blowing himself up What are the chances we ever see the video?",
              "Texas serial bomber made video confession before blowing himself up What are the chances we ever see the video? About the same as the chances of the Browns winning the Super Bowl.",
              "Texas serial bomber made video confession before blowing himself up What are the chances we ever see the video? About the same as the chances of the Browns winning the Super Bowl. I take the browns to the super bowl every morning",
              "Texas serial bomber made video confession before blowing himself up What are the chances we ever see the video? About the same as the chances of the Browns winning the Super Bowl. I take the browns to the super bowl every morning I have to applaud your regularity",
              "Texas serial bomber made video confession before blowing himself up What are the chances we ever see the video? About the same as the chances of the Browns winning the Super Bowl. I take the browns to the super bowl every morning I have to applaud your regularity I thought at first you meant he posts that comment regularly. But now I get it. Healthy colon.",
              "Texas serial bomber made video confession before blowing himself up What are the chances we ever see the video? About the same as the chances of the Browns winning the Super Bowl. I take the browns to the super bowl every morning I have to applaud your regularity Pshh I'm taking the browns to the super bowl as we speak",
              "Texas serial bomber made video confession before blowing himself up What are the chances we ever see the video? About the same as the chances of the Browns winning the Super Bowl. I take the browns to the super bowl every morning I have to applaud your regularity Consistency is the key.",
              "Texas serial bomber made video confession before blowing himself up What are the chances we ever see the video? About the same as the chances of the Browns winning the Super Bowl. I take the browns to the super bowl every morning I have to applaud your regularity Seriously. Well done.",
              "Texas serial bomber made video confession before blowing himself up What are the chances we ever see the video? Zero, videos like this are locked down and used for training purposes. There are a host of confessions and tapes of crimes the public will never see and some have caused agents in training to kill themselves because they are so vile.",
              "Texas serial bomber made video confession before blowing himself up What are the chances we ever see the video? Zero, videos like this are locked down and used for training purposes. There are a host of confessions and tapes of crimes the public will never see and some have caused agents in training to kill themselves because they are so vile. here I am thinking 'just transcripts? How bad can it be' Bad, guys. Very bad.",
              "Texas serial bomber made video confession before blowing himself up I want to know what kind of phone he has. I have had one break from a 3 foot fall, and his survived an explosion?!",
              "Texas serial bomber made video confession before blowing himself up I want to know what kind of phone he has. I have had one break from a 3 foot fall, and his survived an explosion?! Nokia brick",
              "Texas serial bomber made video confession before blowing himself up I want to know what kind of phone he has. I have had one break from a 3 foot fall, and his survived an explosion?! Nokia brick God those old analog phones from the 90's were amazingly durable. They also had great reception (Way better than what I have now).",
              "Texas serial bomber made video confession before blowing himself up I want to know what kind of phone he has. I have had one break from a 3 foot fall, and his survived an explosion?! Nokia brick God those old analog phones from the 90's were amazingly durable. They also had great reception (Way better than what I have now). Yes but the old phones had the drawback of having to be charged every two weeks."
              
    ]

edges3 = {0:0,
         1:0,
         2:1,
         3:2,
         4:3,
         5:4,
         6:4,
         7:4,
         8:4,
         9:1,
         10:9,
         11:0,
         12:11,
         13:12,
         14:13
         }

#document4= ["Parents can help their children be successful in school by encouraging them. Children usually enjoy playing games instead of studying their boring lessons, so parents have to take the responsibility to monitor their studying and to remind them to do their homework at home after school.  Parents should also encourage their children to study by buying story books with pictures, or they can buy text books or tapes that help children learn to spell or read.  The best way to encourage children to study efficiently is to spell or read.  The best way to encourage children to study efficiently is to reward them when they get an 'A.'  As a child, I experienced this.  My parents gave me a gift if I had studied well, and then I was very excited.  So, if parents really want their children to succeed in school, they need to pay attention to their children's studies and encourage them."]
document4= ["Parents can help their children be successful in school by encouraging them. Children usually enjoy playing games instead of studying their boring lessons, so parents have to take the responsibility to monitor their studying and to remind them to do their homework at home after school.Parents should also encourage their children to study by buying story books with pictures, or they can buy text books or tapes that help children learn to spell or read. The best way to encourage children to study efficiently is to spell or read."]

document5= ["lBJ LBJ LBJ LBJ LBJ Lakers Lakers Lakers Lakers Lakers",
            "Warriors Warriors Warriors Warriors Warriors Championship Championship Championship Championship Championship"]

document6= ["lBJ LBJ LBJ LBJ LBJ Warriors Warriors Warriors Warriors Warriors Lakers Lakers Lakers Lakers Lakers Championship Championship Championship Championship Championship "]
document7= ["lBJ LBJ LBJ LBJ LBJ",
            " Lakers Lakers Lakers Lakers Lakers",
            " Warriors Warriors Warriors Warriors Warriors",
            " Championship Championship Championship Championship Championship "]

#document6= ["lBJ LBJ LBJ LBJ LBJ Lakers Lakers Lakers Lakers Lakers"]
document8= ["lBJ LBJ LBJ LBJ LBJ LBJ LBJ LBJ Warriors Championship basketball Lakers Lakers Lakers Lakers Lakers Lakers Lakers Lakers curry"]

document9= ["lBJ LBJ Lakers Lakers",
            "Warriors  Warriors Championship  Championship"]

document10 =["What concept completely blows your mind?",
             "The concept of the observable universe. The fact that the reason we can't see a certain range into the space being because the light hasn't had time to get to earth since the beginning of the universe is crazy to me.",
             "So you mean the universe is buffering for us?",
             "Wow, now your analogy blew my mind!",
             "I want it now godamit! gluurrraAA grrraAAA",
             "Nah it's more like the draw distance.",
             "this comment literally made the whole observable universe thing actually make sense to me for the first time. cheers.",
             "Your comment just blew my mind into milky way chunks.",
             "Oh. Damn.",
             "Holy shit o.o",
             "I guarantee the universe is gonna put itself behind a paywall very soon",
             "There is an horizon beyond which we will never be able to see no matter how long the universe runs for. It is one of the unsolved cosmological problems. If there are boundaries beyond which no information will ever pass then how did the universe end up homogeneous?",
             "Not really.",
             
             "That until the invention of the train, no one had been able to travel faster than a horse on land.",
             "Also, until trains no one really need a consistent time. The difference between 1:30 and 1:50 was largely inconsequential. Well, until you have several tons of steel hurtling down a track and two of them try to occupy the same space and time. It wasn't uncommon for different clocks in town to display different times until the rail road came through.EDIT: Yes, I get that maritime needed accurate clocks to navigate. That's not what I'm talking about. What I'm talking about is synchronized clocks. Clock A has the same time as place Clock B 200 miles away. For maritime stuff that doesn't matter as long as everyone can accurately judge that X amount of time has passed. Example: If my clock reads 10:10 and your's read 10:15 and 20 minutes later mine reads 10:30 and yours reads 10:35, you will not get lost at sea. Also fixed an auto correct word.",
             "a lot of my friends apparently think the very same thing.",
             "It seems to be cultural. My wife is a wedding photographer and some clients will tell her, oh, it says 1pm but nobody will show up until 1:45. We call it 'X people time.' X has been black, Latin, Indian, southern, Greek...probably a half dozen others.I couldn't stand that. I keep German people time.",
             "German time is showing up 10 minutes early",
             "Like working at a fast food joint. It's 2pm! Why are you just getting here?! Because I start at 2. You need to be 15 minutes early! Can I punch in 15 minutes early then? No! You sit in back and wait till your start time. Okay. Then I'll be here at my start time. Fuck your shit.",
             "Yeah all I need to do is put my bag away and put my apron/hat on. I was once 2 minutes late and got bitched out because of it. So I wasn't even needed there if my manager had the time to delay me for another 3 minutes",
             "You should wash your hands too.",
             "Yeah I do usually but i don't make the food I just take orders"
              
    ]

document11 =[            
             "That until the invention of the train, no one had been able to travel faster than a horse on land.",
             "Also, until trains no one really need a consistent time. The difference between 1:30 and 1:50 was largely inconsequential. Well, until you have several tons of steel hurtling down a track and two of them try to occupy the same space and time. It wasn't uncommon for different clocks in town to display different times until the rail road came through.EDIT: Yes, I get that maritime needed accurate clocks to navigate. That's not what I'm talking about. What I'm talking about is synchronized clocks. Clock A has the same time as place Clock B 200 miles away. For maritime stuff that doesn't matter as long as everyone can accurately judge that X amount of time has passed. Example: If my clock reads 10:10 and your's read 10:15 and 20 minutes later mine reads 10:30 and yours reads 10:35, you will not get lost at sea. Also fixed an auto correct word.",
             "a lot of my friends apparently think the very same thing.",
             "It seems to be cultural. My wife is a wedding photographer and some clients will tell her, oh, it says 1pm but nobody will show up until 1:45. We call it 'X people time.' X has been black, Latin, Indian, southern, Greek...probably a half dozen others.I couldn't stand that. I keep German people time.",
             "German time is showing up 10 minutes early",
             "Like working at a fast food joint. It's 2pm! Why are you just getting here?! Because I start at 2. You need to be 15 minutes early! Can I punch in 15 minutes early then? No! You sit in back and wait till your start time. Okay. Then I'll be here at my start time. Fuck your shit.",
             "Yeah all I need to do is put my bag away and put my apron/hat on. I was once 2 minutes late and got bitched out because of it. So I wasn't even needed there if my manager had the time to delay me for another 3 minutes",
             "You should wash your hands too.",
             "Yeah I do usually but i don't make the food I just take orders"
              
    ]

edges10 = {0:0,
         1:0,
         2:1,
         3:2,
         4:2,
         5:2,
         6:2,
         7:2,
         8:2,
         9:2,
         10:2,
         11:2,
         12:2,
         
         13:13,
         14:13,
         15:14,
         16:15,
         17:16,
         18:17,
         19:18,
         20:19,
         21:20
         }

#documents = documents1+documents2
#documents= documents1
documents = documents3
edges= edges3
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

print 'got', len(documents), 'documents'    # got 9 documents
#pprint(documents)

class MyTexts(object):
    """Construct generator to avoid loading all docs
    
    """
    def __init__(self):
        #stop word list
        #self.stoplist = set('for a of the and to in'.split())
        pass

    def __iter__(self):
        for doc in documents:
            #remove stop words from docs
            stop_free = [i for i in doc.lower().split() if i not in stop]
            punc_free = [ch for ch in stop_free if ch not in exclude]
            normalized = [lemma.lemmatize(word) for word in punc_free]
        
            #yield [word for word in doc.lower().split() if word not in stop]
            yield  normalized


def get_dictionary(texts, min_count=1):
    """Construct dictionary 
    
    """
    dictionary = corpora.Dictionary(texts)
    lowfreq_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() 
                    if docfreq < min_count]
    # remove stop words and low frequence words
    dictionary.filter_tokens(lowfreq_ids)
    # remove gaps in id sequence after words that were removed
    dictionary.compactify()
    
    #dictionary.save('docs.dict')
    return dictionary


def corpus2bow(texts,dictionary):
    """represent docs into a list with bag of words model
       bow: bag of words
    
    """
    corpus=[dictionary.doc2bow(text) for text in texts]
    #pprint(corpus)
    
    # save corpus
    #corpora.MmCorpus.serialize('corpus.mm', corpus)
    # load corpus
    #corpus = corpora.MmCorpus('corpus.mm')
    
    return corpus

def bow2tfidf(corpus):
    """represent docs  with TF*IDF model
    
    """
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus] # wrap the old corpus to tfidf
    
    #print tfidf, '\n' # TfidfModel(num_docs=9, num_nnz=51) 
    #print corpus_tfidf, '\n'
    #print tfidf[corpus[0]], '\n' # convert first doc from bow to tfidf
    
    #for doc in corpus_tfidf: # convert the whole corpus on the fly
    #    print doc
    
    return corpus_tfidf
        
def topic_models(corpus,dictionary,num_topics=2,edges=None):
    """modelling the corpus with LDA, LSI and HDP
    
    """

    LDA_model = models.LdaModel(corpus = corpus, id2word = dictionary, num_topics=num_topics,edges = edges)
    #LDA_model.save('LDA.model')
    #LDA_model = models.LdaModel.load('LDA.model')
    topics =  LDA_model.show_topics( num_words=15, log=False, formatted=False)
    for t in topics:
        print t
    

    i=0
    for c in corpus:
        doc_t =  LDA_model.get_document_topics(c)
        print i, doc_t
        i+=1
    
    #LDA_model.bound(corpus, gamma, subsample_ratio)
    
    #In order to compare perplexities you need to convert gensim's perplexity
    #np.exp(-1. * LDA_model.log_perplexity(train_corpus)).
    '''
    hdp = models.HdpModel(corpus_tfidf, T=100,id2word=dictionary)
    hdp.save("HDP.model")

    # initialize a fold-in LSI transformation
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=num_topics) 

    # create a double wrapper over the original corpus:bow->tfidf->fold-in-lsi
    corpus_lsi = lsi[corpus_tfidf] 

    # save model
    lsi.save('model.lsi')
    # load model
    lsi = models.LsiModel.load('model.lsi')
    
    '''
    
    '''
    nodes = list(corpus_lsi)
    print nodes
    ax0 = [x[0][1] for x in nodes] 
    ax1 = [x[1][1] for x in nodes]
    
    plt.plot(ax0,ax1,'o')
    plt.show()
    '''
    
    return LDA_model


def doc_similarity(doc, corpus):

    
    ver_bow=dictionary.doc2bow(doc.lower().split())#return bags-of-word[(tokenid,count)....]
    print(ver_bow)
    
    lsi = models.LsiModel.load('model.lsi')        
    vec_lsi=lsi[ver_bow]
    print(vec_lsi)
    
    index = similarities.MatrixSimilarity(lsi[corpus]) # transform corpus to LSI space and index it
    
    sims=index[vec_lsi]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    return (sims)

def perplexity(ldamodel, testset, dictionary, size_dictionary, num_topics):
    """calculate the perplexity of a lda-model
    
    """
    
    # dictionary : {7822:'deferment', 1841:'circuitry',19202:'fabianism'...]
    #print ('the info of this ldamodel: \n')
    print ('num of testset: %s; size_dictionary: %s; num of topics: %s'%(len(testset), size_dictionary, num_topics))
    prep = 0.0
    prob_doc_sum = 0.0
    topic_word_list = [] # store the probablity of topic-word:[(u'business', 0.010020942661849608),(u'family', 0.0088027946271537413)...]
    for topic_id in range(num_topics):
        topic_word = ldamodel.show_topic(topic_id, size_dictionary)
        
        dic = {}
        for word, probability in topic_word:
            dic[word] = probability
        topic_word_list.append(dic)
    doc_topics_ist = [] #store the doc-topic tuples:[(0, 0.0006211180124223594),(1, 0.0006211180124223594),...]
    for doc in testset: 
        #doc_topics_ist.append(ldamodel.get_document_topics(doc, minimum_probability=0))
        doc_topics_ist.append(ldamodel[doc])
    testset_word_num = 0
   
    for i in range(len(testset)):
        prob_doc = 0.0 # the probablity of the doc
        doc = testset[i]
        doc_word_num = 0 # the num of words in the doc
        for word_id, num in doc:
            prob_word = 0.0 # the probablity of the word 
            doc_word_num += num
            word = dictionary[word_id]
            for topic_id in range(num_topics):
                # cal p(w) : p(w) = sumz(p(z)*p(w|z))
                prob_topic = doc_topics_ist[i][topic_id][1]
                prob_topic_word = topic_word_list[topic_id][word]
                prob_word += prob_topic*prob_topic_word
            prob_doc += math.log(prob_word) # p(d) = sum(log(p(w)))
        prob_doc_sum += prob_doc
        testset_word_num += doc_word_num
    prep = math.exp(-prob_doc_sum/testset_word_num) # perplexity = exp(-sum(p(d)/sum(Nd))
    print ("the perplexity of this ldamodel is : %s"%prep)
    return prep

def test_perplexity(testset,num_topics):
    
    ldamodel_path = 'LDA.model'
    dictionary = corpora.Dictionary.load('docs.dict')
    lda_model = models.ldamodel.LdaModel.load(ldamodel_path)
    hdp = models.hdpmodel.HdpModel.load("HDP.model")
    # sample 1/300
    #for i in range(corpus.num_docs/300):
    #    testset.append(corpus[i*300])

    return perplexity(lda_model, testset, dictionary, len(dictionary.keys()), num_topics)
    
if __name__ == '__main__':
    texts = MyTexts()
    
    dictionary = get_dictionary(texts, min_count=1)
  
    # save and load dictionary
    '''
    dictionary.save('docs.dict')
    dictionary = corpora.Dictionary.load('docs.dict')
    print dictionary
    '''
    corpus = corpus2bow(texts,dictionary)

    corpus_tfidf = bow2tfidf(corpus)
    #doc="Human computer interaction"
    #print doc_similarity(doc, corpus)
    num_topics = 3
    
    
    lda_model = topic_models(corpus=corpus, dictionary=dictionary,num_topics=num_topics,edges=edges)
    
    #ldamodel_path = 'LDA.model'
    #lda_model = models.ldamodel.LdaModel.load(ldamodel_path)
    #doc=['mean','universe' ,'buffering' ,'us']
    #doc = ['Not ', 'really']
    '''
    for t in texts:
        doc = lda_model.id2word.doc2bow(t)
        #doc_topics, word_topics, phi_values = lda_model.get_document_topics(doc, per_word_topics=True)
        results = lda_model.get_document_topics(doc, per_word_topics=True)
        print results
        '''
    '''
    for i in range(1,num_topics):
        topic_models(corpus=corpus_tfidf, dictionary=dictionary,num_topics=i)
        lda_model = models.ldamodel.LdaModel.load(ldamodel_path)
        '''
        #test_perplexity(corpus_tfidf, i)
        #coherence =  CoherenceModel(model=lda_model, corpus=corpus_tfidf, texts=texts, dictionary=dictionary, coherence='u_mass').get_coherence()      
        #print CoherenceModel(model=lda_model, corpus=corpus_tfidf, texts=texts, dictionary=dictionary, coherence='u_mass').get_coherence()
        #print CoherenceModel(model=lda_model, corpus=corpus, texts=new_docs, dictionary=dictionary, coherence='c_uci').get_coherence()
        #print CoherenceModel(model=lda_model, corpus=corpus, texts=new_docs, dictionary=dictionary, coherence='c_npmi').get_coherence()
        #print coherence
    
       
    
          
