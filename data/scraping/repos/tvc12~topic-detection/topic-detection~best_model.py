# install deepai_nlp
# git clone https://github.com/deepai-solutions/deepai_nlp.git
# cd deepai_nlp
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

limit = 70  # max topic
start = 10
step = 14
lda_model, num_topics, coherence_values = compute_coherence_values(
    dictionary, bow_corpus, text_data, start=start, limit=limit, step=step)

x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

# model_list, coherence_values = compute_coherence_values(
#     dictionary=dictionary, bow_corpus=bow_corpus, texts=text_data, start=start, limit=limit, step=step)

print("Done")
for idx, topic in lda_model.print_topics(-1):
    print("Topic: {} \nWords: {}".format(idx, topic))
    print("\n")
# -------------------------------------------------------------------------------
# save model

print("Save")
# save corpus
pickle.dump(bow_corpus, open(f'corpus_{num_topics}.pkl', 'wb'))
# save dictionary
dictionary.save(f'dictionary_{num_topics}.gensim')
# save LDA model
lda_model.save(f'model_{num_topics}.gensim')
