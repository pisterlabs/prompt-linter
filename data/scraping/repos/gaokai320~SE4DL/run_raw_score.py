import os
import sys
import random
from gensim import corpora
from gensim.models.wrappers import LdaMallet
from gensim.models.coherencemodel import CoherenceModel

mallet_path = sys.argv[1]
workers = int(sys.argv[2])
model_folder = sys.argv[3]
data_folder = sys.argv[4]
prefix = sys.argv[5]
lda_seed = int(sys.argv[6])
num_topics = int(sys.argv[7])
iterations = int(sys.argv[8])

id2word_path = os.path.join(data_folder, prefix + '_id2word.dict')
raw_path = os.path.join(data_folder, prefix + '_preprocessed')



id2word = corpora.Dictionary.load(id2word_path)
texts = []
with open(raw_path) as f:
    for line in f:
        texts.append(line.strip('\n').split(',')[1].split(' '))
print(f'Total number of words: {len(id2word)}')
print(f'Total number of posts: {len(texts)}')

for epoch in range(10):
    for run in range(10):
        random.shuffle(texts)
        corpus = [id2word.doc2bow(post) for post in texts]
        ldamallet = LdaMallet(mallet_path=mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word, workers=workers, prefix=os.path.join(model_folder, 'epoch' + str(epoch) + '_run' + str(run)), optimize_interval=10, iterations=iterations, random_seed=lda_seed)
