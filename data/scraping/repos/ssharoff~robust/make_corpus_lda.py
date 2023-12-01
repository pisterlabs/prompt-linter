#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Derived from Radim Rehurek's Wiki processor 
# Serge Sharoff, 2019, 

"""
Processes a corpus in one-line format to produce (X is the output prefix):
 1. X_wordids.txt.bz2 and X_corpus.pkl.bz2 - the dictionary and the corpus (re-loaded if processed earlier)
 2. X_tfidf.mm - serialized TFIDF model
 3. 
Example:
  python3 make_corpus_lda.py Corpus.ol outprefix [dict_size] [workers]
"""


import logging
import os.path
import sys

from gensim.corpora import Dictionary, HashDictionary, MmCorpus, TextCorpus
from gensim.models import TfidfModel
import gensim
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel


DEFAULT_DICT_SIZE = 300000
DEFAULT_WORKERS = 40
ntopics=100

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s", ' '.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) < 3:
        print(globals()['__doc__'] % locals())
        sys.exit(1)
    inp, outp = sys.argv[1:3]

    if len(sys.argv) > 3:
        keep_words = int(sys.argv[3])
    else:
        keep_words = DEFAULT_DICT_SIZE
    if len(sys.argv) > 4:
        workers = int(sys.argv[4])
    else:
        workers = DEFAULT_WORKERS

    if os.path.exists(outp + '_wordids.txt.bz2') and os.path.exists(outp + '_corpus.pkl.bz2'):
        dictionary = Dictionary.load_from_text(outp + '_wordids.txt.bz2')
        wiki=TextCorpus.load(outp + '_corpus.pkl.bz2')
    else:
        wiki = TextCorpus(inp)  
        # only keep the most frequent words
        wiki.dictionary.filter_extremes(no_below=20, no_above=0.1, keep_n=DEFAULT_DICT_SIZE)
        # save dictionary and bag-of-words (term-document frequency matrix)
        #MmCorpus.serialize(outp + '_bow.mm', wiki, progress_cnt=10000, metadata=True)  # another ~9h
        wiki.dictionary.save_as_text(outp + '_wordids.txt.bz2')
        wiki.save(outp + '_corpus.pkl.bz2')
        # load back the id->word mapping directly from file
        # this seems to save more memory, compared to keeping the wiki.dictionary object from above
        dictionary = Dictionary.load_from_text(outp + '_wordids.txt.bz2')

    # build tfidf, ~50min
    if os.path.exists(outp+'_tfidf.mm'):
        mm = gensim.corpora.MmCorpus(outp+'_tfidf.mm')
    else:
        tfidf = TfidfModel(wiki, id2word=dictionary, normalize=True)
        #tfidf.save(outp + '.tfidf_model')

        # save tfidf vectors in matrix market format
        # ~4h; result file is 15GB! bzip2'ed down to 4.5GB
        mm=tfidf[wiki]
        MmCorpus.serialize(outp + '_tfidf.mm', mm, progress_cnt=10000)

    logger.info("finished pre-processing, starting LDA %s", program)

    lda = LdaMulticore(mm, id2word=dictionary, workers=workers, num_topics=ntopics)
    lda.save(outp+str(ntopics)+'_lda.model')
    topics=lda.show_topics(num_topics=ntopics, num_words=30)
    print(topics)
    logger.info("finished LDA %s", program)

    toptopics=lda.top_topics(corpus=wiki, dictionary=lda.id2word, coherence='u_mass')
    logger.info("top topicsL %s", 'u_mass')
    print(toptopics)
    # the following never produced useful info:
    #toptopics=lda.top_topics(texts=wiki, dictionary=lda.id2word, coherence='c_v')
    #logger.info("top topicsL %s", 'c_v')
    #print(toptopics)

