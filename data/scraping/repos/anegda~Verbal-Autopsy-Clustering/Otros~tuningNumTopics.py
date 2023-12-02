import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import gensim
import re
import numpy as np
import nltk
import tqdm
from gensim.models import CoherenceModel
import gensim.corpora as corpora
import pandas as pd
pd.options.mode.chained_assignment = None
from nltk import ToktokTokenizer, WordNetLemmatizer, SnowballStemmer
from nltk.corpus import stopwords

from preproceso import stemmer


def compute_coherence_values(corpus, dictionary, k, a, b):
    lda_model = gensim.models.LdaModel(corpus=corpus,
                           id2word=dictionary,
                           num_topics=k,
                           random_state=100,
                           chunksize=100,
                           passes=10,
                           alpha=a,
                           eta=b)

    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')

    return coherence_model_lda.get_coherence()

def limpiar_texto(texto):
    # Eliminamos los caracteres especiales
    texto = re.sub(r'\W', ' ', str(texto))
    # Eliminado las palabras que tengo un solo caracter
    texto = re.sub(r'\s+[a-zA-Z]\s+', ' ', texto)
    # Sustituir los espacios en blanco en uno solo
    texto = re.sub(r'\s+', ' ', texto, flags=re.I)
    # Convertimos textos a minusculas
    texto = texto.lower()
    return texto

def eliminar_stopwords(tokens):
    return [token for token in tokens if token not in STOPWORDS and not token.isdigit()]

def lematizar(tokens):
    return [wnl.lemmatize(token) for token in tokens]

def eliminar_palabras_concretas(tokens):
    palabras_concretas = {"hospit", "die", "death", "doctor", "deceas", "person", "servic", "nurs", "client", "peopl", "patient",                   #ELEMENTOS DE HOSPITAL QUE NO APORTAN INFO SOBRE ENFERMEDAD
                          "brother", "father","respondetn","uncl","famili","member","husband","son", "daughter","marriag",
                          "day", "year", "month", "april", "date", "feb", "jan", "time", "place","later","hour",                                    #FECHAS QUE NO APORTAN INFO SOBRE ENFERMEDD
                          "interview", "opinion", "thousand", "particip", "admit", "document", "inform", "explain", "said", "respond","interviewe",                                                                                                #PALABRAS QUE TIENEN QUE VER CON LA ENTREVISTA
                          "write", "commend", "done", "told", "came", "done", "think", "went", "took", "got",                                       #OTROS VERBOS
                          "brought","becam","start",
                          "even", "also", "sudden", "would", "us", "thank","alreadi","rather","p","none","b",                                       #PALABRAS QUE NO APORTAN SIGNIFICADO
                          "caus", "due", "suffer", "felt", "consequ"}                                                                               #PALABRAS SEGUIDAS POR SINTOMAS


    return [token for token in tokens if token not in palabras_concretas]

def estemizar(tokens):
    return [stemmer.stem(token) for token in tokens]


nltk.download("stopwords")

STOPWORDS = set(stopwords.words("english"))
wnl = WordNetLemmatizer()
stemmer = SnowballStemmer("english")

df = pd.read_csv("../datasets/train.csv")
dfOld = df      #guardamos aqui las columnas que no modificamos pero si necesitamos posteriormente
df = df[["open_response"]]

# 1.- Limpiamos (quitar caracteres especiaes, minúsculas...)
df["Tokens"] = df.open_response.apply(limpiar_texto)

# 2.- Tokenizamos
tokenizer= ToktokTokenizer()
df["Tokens"] = df.Tokens.apply(tokenizer.tokenize)

# 3.- Eliminar stopwords y digitos
df["Tokens"] = df.Tokens.apply(eliminar_stopwords)

# 4.- ESTEMIZAR / LEMATIZAR ???
df["Tokens"] = df.Tokens.apply(estemizar)
#print(df.Tokens[0][0:10])

# 5.- ELIMINAMOS PALABRAS CONCRETAS QUE APARECEN MUCHO PERO NO APORTAN SIGNIFICADO
df["Tokens"] = df.Tokens.apply(eliminar_palabras_concretas)


# A PARTIR DE AQUI ES NUEVO

data_lemmatized = df["Tokens"]

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)
# Create Corpus
texts = data_lemmatized
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# ITERAMOS
grid = {}
grid['Validation_Set'] = {}
# Topics range
min_topics = 10
max_topics = 29
step_size = 2
topics_range = range(min_topics, max_topics, step_size)
# Alpha parameter
alpha = [0.15,0.2,0.5,0.6,0.7,0.9,1.5,1.75,2]
#alpha.append('auto')
#alpha.append('asymmetric')
# Beta parameter
beta = [0.15, 0.2, 0.5, 0.7, 0.9, 1.5,2]
#beta.append('auto')
# Validation sets
num_of_docs = len(corpus)
corpus_sets = [  #gensim.utils.ClippedCorpus(corpus, int(num_of_docs)),
    # gensim.utils.ClippedCorpus(corpus, int(num_of_docs*0.5)),
    # gensim.utils.ClippedCorpus(corpus, int(num_of_docs * 0.75)),
    corpus
    ]
corpus_title = [
    #'1% corpus',
    #'50% corpus',
    #'75% Corpus',
    '100% Corpus'
]
model_results = {'Validation_Set': [],
                 'Topics': [],
                 'Alpha': [],
                 'Beta': [],
                 'Coherence': []
                 }
# Can take a long time to run
if 1 == 1:
    pbar = tqdm.tqdm(total=1008)

    # iterate through validation corpuses
    for i in range(len(corpus_sets)):
        # iterate through number of topics
        for k in topics_range:
            # iterate through alpha values
            for a in alpha:
                # iterare through beta values
                for b in beta:
                    # get the coherence score for the given parameters
                    cv = compute_coherence_values(corpus=corpus_sets[i], dictionary=id2word,
                                                  k=k, a=a, b=b)
                    if cv > 0.6:
                        # Save the model results
                        model_results['Validation_Set'].append(corpus_title[i])
                        model_results['Topics'].append(k)
                        model_results['Alpha'].append(a)
                        model_results['Beta'].append(b)
                        model_results['Coherence'].append(cv)

                    pbar.update(1)
    pd.DataFrame(model_results).to_csv('Resultados/lda_tuning_results+0.6.csv', index=False)
    pbar.close()

# Para la documentación
'''df_results = pd.DataFrame(model_results)
print(df_results.head())
coherencia = df_results["Coherence"]
num_topics = df_results["Topics"]

import matplotlib.pyplot as plt
plt.plot(num_topics, coherencia, marker = 'o')
plt.savefig("Imagenes/numTopics.png")
plt.show()'''



