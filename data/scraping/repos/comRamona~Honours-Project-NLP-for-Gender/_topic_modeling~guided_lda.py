from _storage.storage import FileDir
import os
from gensim.models.ldamodel import LdaModel
from gensim.models import LdaMulticore
# from gensim.models import CoherenceModel
from gensim.corpora import MmCorpus
from gensim.corpora import Dictionary
import numpy as np
import random
import logging

random.seed(1)
logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO

v = 10

fd = FileDir()
dictionary = Dictionary.load(os.path.join(fd.models, "dict10.pkl"))
_ = dictionary[0]
n_words = len(dictionary)
a = dictionary.token2id
id2word = dictionary.id2token
del dictionary
corpus = MmCorpus(os.path.join(fd.models, "acl_bow10.mm"))

seed_words ="""resolution anaphora pronoun discourse antecedent pronouns coreference reference definite algorithm
string state set finite context rule algorithm strings language symbol
medical protein gene biomedical wkh abstracts medline patient clinical biological
call caller routing calls destination vietnamese routed router destinations gorin
proof formula graph logic calculus axioms axiom theorem proofs lambek
centering cb discourse cf utterance center utterances theory coherence entities local
japanese method case sentence analysis english dictionary figure japan word
features data corpus set feature table word tag al test
vowel phonological syllable phoneme stress phonetic phonology pronunciation vowels phonemes
semantic logical semantics john sentence interpretation scope logic form set
user dialogue system speech information task spoken human utterance language
discourse text structure relations rhetorical relation units coherence texts rst
segment segmentation segments chain chains boundaries boundary seg cohesion lexical
event temporal time events tense state aspect reference relations relation
de le des les en une est du par pour
generation text system language information knowledge natural figure domain input
genre stylistic style genres fiction humor register biber authorship registers
system text information muc extraction template names patterns pattern domain
document documents query retrieval question information answer term text web
semantic relations domain noun corpus relation nouns lexical ontology patterns
slot incident tgt target id hum phys type fills perp
metaphor literal metonymy metaphors metaphorical essay metonymic essays qualia analogy
word morphological lexicon form dictionary analysis morphology lexical stem arabic
entity named entities ne names ner recognition ace nes mentions mention
paraphrases paraphrase entailment paraphrasing textual para rte pascal entailed dagan
parsing grammar parser parse rule sentence input left grammars np
plan discourse speaker action model goal act utterance user information
model word probability set data number algorithm language corpus method
prosodic speech pitch boundary prosody phrase boundaries accent repairs intonation
semantic verb frame argument verbs role roles predicate arguments
knowledge system semantic language concept representation information network concepts base
subjective opinion sentiment negative polarity positive wiebe reviews sentence opinions
speech recognition word system language data speaker error test spoken
errors error correction spelling ocr correct corrections checker basque corrected detection
english word alignment language source target sentence machine bilingual mt
dependency parsing treebank parser tree parse head model al np
sentence text evaluation document topic summary summarization human summaries score
verb noun syntactic sentence phrase np subject structure case clause
tree node trees nodes derivation tag root figure adjoining grammar
feature structure grammar lexical constraints unification constraint type structures rule
word senses wordnet disambiguation lexical semantic context similarity dictionary
chinese word character segmentation corpus dictionary korean language table system
synset wordnet synsets hypernym ili wordnets hypernyms eurowordnet hyponym ewn wn
convolution recurrent lstm neural network backprop backpropagation deep layer
embedding continuous representation distributional semantics vec vector space"""

n_topics = 100
import numpy as np
eta = np.zeros((n_topics, n_words))
topic_words = seed_words.split("\n")
for i, topic in enumerate(topic_words):
    words = topic.split(" ")
    for w in words:
        if(w in a):
            eta[i,a[w]] = 0.1
            print(w)
            print(i,a[w])
            
already_assigned = np.sum(eta, axis=0)
m,n = eta.shape
for i in range(m):
    for j in range(n):
        if(eta[i,j] ==0 ):
            eta[i,j] = (1 - already_assigned[i]) / n_topics
            
sums = np.sum(eta, axis=0)
for i in range(m):
    for j in range(n):
        eta[i,j] /= sums[i] 
#model = LdaMulticore(corpus=corpus, workers=3, id2word=id2word, num_topics=100, iterations=500, eta=eta, passes=200)
model = LdaModel(corpus, id2word=id2word, num_topics=100, passes=500, iterations=200, alpha="auto", eta="auto")

model.save(os.path.join(fd.models, 'ldaseed' + str(v) + 'lda'))


