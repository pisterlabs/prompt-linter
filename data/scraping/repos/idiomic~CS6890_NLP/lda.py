import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from nltk import pos_tag
import re
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
import multiprocessing.pool
import os

class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess

lemmatizer = WordNetLemmatizer()

# Keep adjectives, nouns, verbs, and adverbs.
keep_pos = {
	'J': wordnet.ADJ,
	'N': wordnet.NOUN,
	'V': wordnet.VERB,
	'R': wordnet.ADV
}

en_stopwords = stopwords.words('english')
n_noun = 0
n_adjective = 0
n_verb = 0
n_adverb = 0
unique = {}
n_unique = 0
def filter(text):
	global n_noun
	global n_adjective
	global n_verb
	global n_adverb
	global unique
	global n_unique

	text = text.lower()
	text = re.sub('[\n\'\"]+', '', text)
	text = re.sub('http[s]*[^\s]+', '', text)
	text = re.sub('[@&]\w+', '', text)
	text = re.sub('rt', '', text)
	text = re.sub('\.\.\.', '', text)

	all_words = []

	for sent in re.findall('[^.?!;:]+', text):
		if sent == '':
			continue;
		words = re.split('[\s()\[\],^+*`/]', sent.strip())
		words = [
			word
			for word in words
			if not word is '' and not word in en_stopwords
		]
		pos = pos_tag(words)
		pos = [
			(pair[0], keep_pos[pair[1][0].upper()])
			for pair in pos
			if pair[1][0].upper() in keep_pos
		]
		for pair in pos:
			if pair[1] == wordnet.ADJ:
				n_adjective += 1
			elif pair[1] == wordnet.NOUN:
				n_noun += 1
			elif pair[1] == wordnet.VERB:
				n_verb += 1
			elif pair[1] == wordnet.ADV:
				n_adverb += 1
			if not pair[0] in unique:
				unique[pair[0]] = True
				n_unique += 1
		all_words = all_words + [
			lemmatizer.lemmatize(pair[0], pair[1])
			for pair in pos
		]
	return all_words

def viewLDA(times, filtered, original, num_topics):
	id2word = corpora.Dictionary(filtered)
	print('built id2word')
	corpus = [id2word.doc2bow(text) for text in filtered]
	print('built doc2bow')
	lda_model = gensim.models.ldamodel.LdaModel(
		corpus=corpus,
		id2word=id2word,
		num_topics=num_topics,
		random_state=np.random.randint(1024),
		update_every=10,
		chunksize=10,
		passes=100,
		alpha='auto',
		per_word_topics=True)
	print('saving')
	lda_model.save('lda4.model')
	pprint(lda_model.print_topics())

id2word = None
corpus = None
def filterOne(args):
	rnd = args[0]
	num_topics = args[1]
	print(num_topics, rnd)
	lda_model = gensim.models.ldamodel.LdaModel(
		corpus=corpus,
		id2word=id2word,
		num_topics=num_topics,
		random_state=rnd,
		update_every=10,
		chunksize=10,
		passes=50,
		alpha='auto',
		per_word_topics=True)
	coherence_model_lda = CoherenceModel(model=lda_model, texts=filtered, dictionary=id2word, coherence='c_v')
	return coherence_model_lda.get_coherence()

def filterAll(times, filtered, original):
	global id2word
	global corpus
	id2word = corpora.Dictionary(filtered)
	corpus = [id2word.doc2bow(text) for text in filtered]
	s = 2
	f = 20
	t = 20
	n = f - s
	x = list(range(s, f))
	args = []
	for num_topics in x:
		for i in range(t):
			args.append((np.random.randint(1024), num_topics))
	coherence = MyPool(32).map(filterOne, args)
	pprint(coherence)
	coherence = np.array(coherence)
	coherence = np.reshape(coherence, (n, t))
	mean = np.mean(coherence, 1)
	std = np.std(coherence, 1)

	pprint(x)

	pprint(mean)

	pprint(std)

	plt.errorbar(x, mean, std, linestyle='None', marker='^')
	plt.xlabel('Num Topics')
	plt.ylabel('Coherence Score')
	plt.legend(('mean', 'lower_sigma', 'upper_sigma'), loc='best')
	plt.show()

f = open('final.txt', 'r')

print('reading')
times = []
filtered = []
original = []
for line in f.readlines():
	t = re.match('^\d{13}', line)
	if t == None:
		continue
	t = t[0]
	times.append(t)
	line = line[14:]
	filtered.append(filter(line))
print(len(times))
print('processing')
print(n_noun)
print(n_adjective)
print(n_verb)
print(n_adverb)
print(n_unique)

# filterAll(times, filtered, original)
viewLDA(times, filtered, original, 10)


f.close()