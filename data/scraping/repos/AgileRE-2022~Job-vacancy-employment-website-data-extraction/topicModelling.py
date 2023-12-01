#library
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import gensim.corpora as corpora
import pyLDAvis
import pyLDAvis.gensim_models
from pprint import pprint

import pyLDAvis.sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

#read data
data = pd.read_csv('hasil.csv')
data = data.drop(columns=['Companies','Locations'])

#formating text lowercasing remove symbol
stop_words = set(stopwords.words('english'))
data['Descriptions_without_stopwords'] = data['Descriptions'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
data['Descriptions_without_stopwords'] = data['Descriptions_without_stopwords'].str.replace('[^\w\s]', '')

# data['Descriptions_without_stopwords'] = data['Descriptions_without_stopwords'].map(re.sub(r"(\w)([A-Z])", r"\1 \2", ele) for ele in test_list)
data['Descriptions_without_stopwords'] = data['Descriptions_without_stopwords'].map(lambda x: x.lower())



def sent_to_words (sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence),deacc=True))

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) 
        if word not in stop_words] for doc in texts]

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(w) for w in text.split(' ')])

data['Descriptions_without_stopwords'] = data['Descriptions_without_stopwords'].apply(lemmatize_text)
print(data['Descriptions_without_stopwords'])
dataLDA = data.Descriptions_without_stopwords.tolist()
data_words = list(sent_to_words(dataLDA))
data_words = remove_stopwords(data_words)
print(data_words)

#id2text
id2word = corpora.Dictionary(data_words)
texts = data_words
corpus = [id2word.doc2bow(text) for text in texts]
# print(corpus[:1][0][:30])

lda_model = gensim.models.ldamodel.LdaModel(corpus = corpus, id2word = id2word, num_topics=30, update_every=1, chunksize=100, passes=10, alpha="auto")

# num_topics = 10
# lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word,num_topics=num_topics)
# pprint(lda_model.print_topics())
# doc_lda = lda_model[corpus]

# # Visualize the topics
# pyLDAvis.enable_notebook()LDAvis_data_filepath = os.path.join('./results/ldavis_prepared_'+str(num_topics))
# # # this is a bit time consuming - make the if statement True
# # # if you want to execute visualization prep yourself
# if 1 == 1:
#     LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
#     with open(LDAvis_data_filepath, 'wb') as f:
#         pickle.dump(LDAvis_prepared, f)
        
# # load the pre-prepared pyLDAvis data from disk
# with open(LDAvis_data_filepath, 'rb') as f:
#     LDAvis_prepared = pickle.load(f)
# pyLDAvis.save_html(LDAvis_prepared, './results/ldavis_prepared_'+ str(num_topics) +'.html')

# LDAvis_prepared