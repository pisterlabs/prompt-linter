# install deepai_nlp
# git clone https://github.com/deepai-solutions/deepai_nlp.git
# pip install -e
# install gensim
# conda install -c anaconda gensim
# or
# pip install gensim
# core i7 6700HQ 2.6GHz
# RAM 8G - Nividia Geforce GTX 950M

import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
import gensim
import numpy
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel

from deepai_nlp.tokenization.crf_tokenizer import CrfTokenizer
from deepai_nlp.word_embedding import word2vec_gensim
from prepare import preprocess_text
from coherence import compute_coherence_values

tokenizer = CrfTokenizer()


def read_data(files, rate_read = 1):
    i = 0
    data = []
    for path in files:
        print('Load file', path)
        with open(path, 'r') as file:
            data.append(file.read())
            file.close()

        files.remove(path)
        # break
        if (i / max_files > rate_read):
            break
        else:
            i = i + 1
            print("Count: ", i)

    print("Clean text")
    return preprocess_text(data, tokenizer)


# open folder
main_dir = 'data'
path, dirs, files = next(os.walk(main_dir))
files = []
for dir in dirs:
    path, sub_dirs, sub_files = next(os.walk('{}/{}/'.format(main_dir, dir)))
    for file in sub_files:
        file = '{}/{}/{}'.format(main_dir, dir, file)
        files.append(file)
max_files = len(files)

num_topics = 35

# and read data
text_data = read_data(files)

# Preprocessing the raw text

# Map text
print("Mapping")
dictionary = corpora.Dictionary(text_data)


# Remove stopwords
print("Remove Stopwords")
dictionary.filter_extremes(no_below=10, no_above=0.6, keep_n=100000)
# dictionary.compactify()
bow_corpus = [dictionary.doc2bow(doc) for doc in text_data]

# training
# parameter LdaMulticore see in https://radimrehurek.com/gensim/models/ldamulticore.html
# --------------------------------------------------------------------------------------
print("Basic Training")
lda_model = gensim.models.LdaMulticore(
    bow_corpus,
    num_topics=num_topics,
    id2word=dictionary,
    passes=10,
    workers=8,
    minimum_probability=0.04,
    random_state=50,
    alpha=1e-2,
    chunksize=4500,
    eta=0.5e-2,
)
coherencemodel = CoherenceModel(
    model=lda_model, texts=text_data, dictionary=dictionary, coherence='c_v')

print("Num Topics =", num_topics, " has Coherence Value of",
      round(coherencemodel.get_coherence(), 4))

print("Done")

for idx, topic in lda_model.print_topics(-1):
    print("Topic: {} \nWords: {}".format(idx, topic))
    print("\n")

# save model

print("Save")
# save corpus
pickle.dump(bow_corpus, open(f'corpus_{num_topics}.pkl', 'wb'))
# save dictionary
dictionary.save(f'dictionary_{num_topics}.gensim')
# save LDA model
lda_model.save(f'model_{num_topics}.gensim')
