import glob
import re

from cltk.stem.latin.j_v import JVReplacer
from cltk.stem.lemma import LemmaReplacer
from gensim import corpora
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim

STOPS_LIST = []
with open('stopwords/stopwords_latin.txt') as fp:
    for line in fp:
        if not line.startswith('#'):
            STOPS_LIST.append(line.strip())


def import_pliny():

    letters = []
    files = glob.glob('letters/*.txt')
    for file in files:
        fp = open(file, 'r')
        input = fp.read()
        letters.append((input), file.split('/')[:-1])
    return letters


def pre_process(letters):

    pre_processed = []
    for letter in letters:
        NO_PUNCT_RE = re.compile(r'[?!\.\'\"<>():;,]')
        replacer = JVReplacer()
        letter = replacer.replace(letter)
        words = re.sub(NO_PUNCT_RE, '', letter).lower().split()

        for i, word in enumerate(words):
            if word.endswith('-'):
                words[i+1] = '%s%s' % (word.strip('-'), words[i+1])
        words = [w for w in words if not w.endswith('-')]
        words = [w for w in words if w not in STOPS_LIST]
        words = ' '.join(words)
        lemmatizer = LemmaReplacer('latin')
        words = lemmatizer.lemmatize(words)
        # very common words that seemed to be cofounding the topic model
        words = [w for w in words if w not in ['magnus', 'bonus', 'ago', 'valeo']]
        pre_processed.append(words)
    return pre_processed


def make_ldamodel(pre_processed, num_topics=5, pylda=False):

    dictionary = corpora.Dictionary(pre_processed)
    corpus = [dictionary.doc2bow(text) for text in pre_processed]
    model = LdaMulticore(corpus, num_topics=num_topics, id2word=dictionary, passes=30, random_state=1)
    if pylda:
        lda_display = pyLDAvis.gensim.prepare(model, corpus, dictionary)
        with open('topic_associations.txt', 'w') as outfile:
            for f in sorted(glob.glob('letters/*.txt')):
                fp = open(f, 'r')
                text = pre_process([fp.read()])
                dictionary = corpora.Dictionary(text)
                outfile.write('%s - %s\n' % (f, model.get_document_topics(dictionary.doc2bow(text[0]))))
                fp.close()

        pyLDAvis.show(lda_display)


def make_ldamodels(pre_processed, max=6):

    perplex_coherence = []
    dictionary = corpora.Dictionary(pre_processed)
    corpus = [dictionary.doc2bow(text) for text in pre_processed]

    for num in range(5, max + 1):
        model = LdaMulticore(corpus, num_topics=num, id2word=dictionary, passes=30, random_state=1)
        coherence_model = CoherenceModel(model=model, texts=pre_processed, dictionary=dictionary, coherence='c_v')
        perplex_coherence.append((num, model.log_perplexity(corpus), coherence_model.get_coherence()))

    for val in perplex_coherence:
        print(val)


letters = import_pliny()
pre_processed = pre_process(letters)
make_ldamodel(pre_processed, pylda=True)
