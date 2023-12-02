#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 13:29:02 2023

@author: cygnus
"""
import re
import nltk
# to show progress bar
from tqdm import tqdm
# for pretty printing
import pprint
import spacy
import pandas as pd 
from itertools import chain
from collections import Counter
from nltk.corpus import stopwords
# Descargar el conjunto de stopwords en español si no lo tienes
nltk.download('stopwords')
# Obtención de listado de stopwords del inglés
stop_words = list(stopwords.words('spanish'))
nlp = spacy.load("es_core_news_sm")

# for topic modelling
from gensim import corpora
from gensim.models import ldamodel
from gensim.models import CoherenceModel

# for data viz
import pyLDAvis
import pyLDAvis.gensim
import seaborn as sns



##############################DataFrame################################
coments_x = pd.read_csv("/home/milton/Documentos/khorda_data/lomeli/twitter/coments_x.csv", usecols=["texto"])
coments_tiktok = pd.read_csv("/home/milton/Documentos/khorda_data/lomeli/tiktok/coments_tiktok.csv", usecols=["texto"])
coments_insta = pd.read_csv("/home/milton/Documentos/khorda_data/lomeli/insta/coments_insta.csv", usecols=["texto"])
coments_fb = pd.read_csv("/home/milton/Documentos/khorda_data/lomeli/facebook/coments_fb.csv", usecols=["texto"])

# Concatenar todos los dataframes

df_coments = pd.concat([coments_fb, coments_x, coments_tiktok, coments_insta], axis=0)
df_coments

def data_clean(df):
    """
    Parameters
    ----------
    df : DataFrame con columnas
            'texto','comentarisos'repost', 'likes','views'.
    Returns
    -------
    df_posts : DataFrame
        Retorna dos DataFrames limpios:
            El primero con las columnas 'coment','rt','like','views'.
            El segundo con las columnas 'texto', 'tokens'.
    """
    df['texto'] = df['texto'].apply(lambda x: re.sub(r'\d', '', str(x)))
    df['texto'] = df['texto'].apply(lambda x: re.sub(r'[.,;!?]', '', str(x)))
    df['texto'] = df['texto'].apply(lambda x: re.sub(r'jaja(ja)*', '', str(x)))
    df['texto'] = df['texto'].apply(lambda x: re.sub(r"(, '[\W\.]')",r"", str(x)))
    df['texto'] = df['texto'].apply(lambda x: re.sub(r'[^\w\s]', '', x.lower()))
    df = df.fillna('')
    return df

df_coments= data_clean(df_coments)


nlp.disable_pipes('ner')
print(nlp.pipe_names)
dataset = df_coments['texto']
docs = []
for text in tqdm(nlp.pipe(dataset), total=len(dataset)):
    doc = nlp(text) 
    pt = [token.lemma_.lower() for token in doc if
           (len(token.lemma_) > 1 and token.pos_ == "NOUN" and 
          not token.is_stop)]
    docs.append(pt)
counts_word_occurence = Counter(chain(*[x for x in docs]))
low_freq_words = {key:value for (key,value) in counts_word_occurence.items() if value==1}
docs = [[lemma for lemma in text if counts_word_occurence[lemma]>1] for text in docs]  
docs_length=len(docs)
counts_word_percentage = Counter(chain(*[set(x) for x in docs]))
counts_word_percentage = {key:(value/docs_length)*100 for (key,value) in counts_word_percentage.items()}
high_freq_words = {key:value for (key,value) in counts_word_percentage.items() if value>4}
docs =  [[lemma for lemma in text if counts_word_percentage[lemma]<4] for text in docs]


######Nube de Palabras########################
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Limpieza de Stop_words
stop_words.extend(("q", "d", "van", "si", "pa", "así", "ser", "solo", "tan", "va", "as", "aquí", "hacia","dra", "día","hoy","dd","drcarloslomeli","jajaja"
                "le", "con","ella", "qué", "por", "qu", "ers", "das", "ve", "jajaja", "jeje", "La", "nimo", "ms", "da","doccast","vas","jajajaja",
                "drcarloslomeli", "doc", "dr", "jajajajajajaja", "vez"))

##########NUBE DE PALABRAS
def preprocess_text(text):
    words = text # Tokenizar y convertir a minúsculas
    words = [word for word in words if word.isalpha()]  # Eliminar caracteres no alfabéticos
    words = [word for word in words if word not in stop_words]  # Eliminar palabras vacías
    return words

docs = [word for word in docs if word not in stop_words]
# Unimos todas las listas de palabras en una sola lista
all_words = [word for sublist in docs for word in sublist]

# Creamos un DataFrame con las palabras y sus frecuencias
word_counts = pd.Series(all_words).value_counts()

# Tomamos las palabras más comunes (puedes ajustar este valor según tus necesidades)
top_words = word_counts.head(10)

# Unir las palabras en un solo texto (separadas por espacios)
texto = " ".join(all_words)
# Cículo
# Crear una máscara en blanco para establecer un fondo transparente

# Crear el objeto WordCloud
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(texto)
# Mostrar la nube de palabras utilizando matplotlib
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Desactivar ejes
plt.savefig("nube_palabras_coments.png", dpi=300, )
plt.show()
###############################################
pp = pprint.PrettyPrinter(compact=True)
pp.pprint(docs)
lengths =  [len(x) for x in docs]
sns.histplot(lengths)
dictionary = corpora.Dictionary(docs)
corpus = [dictionary.doc2bow(text) for text in docs]

def calculate_coherence(dictionary, corpus, docs, start, stop):
    scores = []
    for topics in range(start, stop):

        # defining the model
        lda_model = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=26, alpha='auto', eval_every=5)

        # U_mass coherence score
        cm_u_mass = CoherenceModel(model=lda_model, corpus=corpus, dictionary=dictionary, coherence='u_mass')
        u_mass_coherence = cm_u_mass.get_coherence()

        # C_v coherence score
        cm_c_v = CoherenceModel(model=lda_model, texts=docs, dictionary=dictionary, coherence='c_v')
        c_v_coherence = cm_c_v.get_coherence()

        values = [topics, u_mass_coherence, c_v_coherence]

        scores.append(values)

    return scores

scores = calculate_coherence(dictionary, corpus, docs, 10, 30)
df = pd.DataFrame(scores, columns = ['number_of_topics','u_mass_coherence','c_v_coherence'])
df = df.melt(id_vars=['number_of_topics'], value_vars=['u_mass_coherence','c_v_coherence'])
# Plotting u_mass_coherence
sns.lineplot(data=df.loc[df['variable'] == 'u_mass_coherence'], x="number_of_topics", y="value").set_title('u_mass coherence')
# Plotting c_v_coherence
sns.lineplot(data=df.loc[df['variable'] == 'c_v_coherence'], x="number_of_topics", y="value").set_title('c_v coherence')

###########Modelo LDA Topics##########################
lda_model = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=17, alpha='auto', eval_every=5)

# print topics
lda_model.print_topics(-1)


# print topics
topics = lda_model.print_topics(-1)

# Mostrar los temas y sus palabras clave
for topic in topics:
    print(topic)

########################################
pyLDAvis.enable_notebook()
viz = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
print(type(viz))