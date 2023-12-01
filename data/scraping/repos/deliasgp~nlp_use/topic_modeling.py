import pandas as pd
import numpy as np
import gensim
import nltk
#nltk.download('punkt')
import sys
from sklearn.model_selection import train_test_split
sys.path.append('D:/repositorios_git/nlp_use/')
import normalizar_texto as nt
#-----------------------------------------------------------------------------*
fec_t = "20230816"
obs_dir = 'E:/Mi unidad/dgavidia_minedu/BD USE/NLP/TABLETAS/Input/observaciones/'+fec_t+'/obs_recepcion_'+fec_t+'.xlsx'
datos = pd.read_excel(obs_dir)
datos = datos[((datos['flg_cat']==97) | 
                           (datos['flg_cat']==1) | 
                           (datos['flg_cat']==2) |
                           (datos['flg_cat']==3))==False]
#-----------------------------------------------------------------------------*
stopword_list = nltk.corpus.stopwords.words('spanish')
stop_words_tablets = nt.stop_words_use(local_file=False,maindir='') + stopword_list + ['estudiante','padre','madre','padres','madres']+ ['segun','san']
#-----------------------------------------------------------------------------*
eliminar_stop_words = ['no','si','solo','se']
for word in eliminar_stop_words:
    if word in stop_words_tablets:
        stop_words_tablets.remove(word)
#-----------------------------------------------------------------------------*
text_corpus = nt.normalizar_texto(datos['OBSERVACION_RECEPCION'],
                                    contraction_expansion=True,
                                    accented_char_removal=True, 
                                    text_lower_case=True, 
                                    text_stemming=False, text_lemmatization=True, 
                                    special_char_removal=True, remove_digits=True,
                                    stopword_removal=True, special_cases = True,
                                    autocorrecion=False,
                                    stopwords = stop_words_tablets)
#-----------------------------------------------------------------------------*
#Eliminando palabras repetidas
from normalizar_texto import palabras_repetidas
texto_limpio = []
for doc in text_corpus:
    word = palabras_repetidas(doc)
    texto_limpio.append(word)
#-----------------------------------------------------------------------------*
datos['obs'] = texto_limpio
datos['obs'] = datos.obs.replace('', 'NA')
datos['target'] = np.where(datos.flg_cat==97,1,0)
#-----------------------------------------------------------------------------*
datos['obs'] = datos['obs'].str.replace("él", "se")
#-----------------------------------------------------------------------------*
print(np.mean(datos['target']))
#%% Topic Label
print("Esta sección busca categorizar respuestas donde los usurios indican que tienen alguna observación")

datos_topic = datos.copy()

print(datos_topic.shape)

datos_topic['flg_cat'].value_counts()


topic_train_corpus, topic_test_corpus, topic_train_label_nums, topic_test_label_nums = train_test_split(
     datos_topic['obs'], #np.array(datos['obs'])
     datos_topic['flg_cat'], #np.array(datos['target'])
     test_size=1/10, random_state=42)

index_train_topic = topic_train_corpus.index
index_test_topic = topic_test_corpus.index

#Tokenizar las oraciones y consruir bigramas
text_tokens = [nltk.word_tokenize(sentence) for sentence in topic_train_corpus]
bigram_model = gensim.models.phrases.Phrases(text_tokens, min_count=1, threshold=0.1)
 #min_count (float, optional) – Ignore all words and bigrams with total collected count lower than this value.
 #threshold (float, optional) – Represent a score threshold for forming the phrases (higher means fewer phrases). 
 #A phrase of words a followed by b is accepted if the score of the phrase is greater than threshold. Heavily depends on concrete scoring-function, 
 #see the scoring parameter.

norm_corpus_bigrams = [bigram_model[doc] for doc in text_tokens]
# Create a dictionary representation of the documents.
dictionary = gensim.corpora.Dictionary(norm_corpus_bigrams)
print('Total Vocabulary Size:', len(dictionary))

bow_corpus = [dictionary.doc2bow(text) for text in norm_corpus_bigrams]
#%%% LSI (Latent Semantic Indexing)-BoW
#words that are used in the same contexts tend to have similar meanings.
from gensim.models.coherencemodel import CoherenceModel

min_topics = 2
max_topics = 10
#Definir una lista para almacenar los resultados
coherence_scores = []
models = []
#Iterar sobre los números de tópicos y calcular la coherencia
from datetime import datetime
start = datetime.now()
np.random.seed(50)
for num_topics in range(min_topics, max_topics + 1):
    lsi_bow = gensim.models.LsiModel(bow_corpus, 
                                     id2word=dictionary, 
                                     num_topics=num_topics,
                                     onepass=True, 
                                     chunksize=1740, 
                                     power_iters=1000)    
    coherence_model = CoherenceModel(model=lsi_bow, texts=norm_corpus_bigrams, corpus=bow_corpus, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    coherence_scores.append((num_topics, coherence_score))
    models.append(lsi_bow)
print(datetime.now() - start)
# Crear un dataframe a partir de los resultados
lsi_df = pd.DataFrame(coherence_scores, columns=['Número de Tópicos', 'Coherencia'])
print(lsi_df)


import matplotlib.pyplot as plt

# Crear un gráfico de línea para los resultados
plt.plot(lsi_df['Número de Tópicos'], lsi_df['Coherencia'], marker='o')
# Añadir etiquetas y título al gráfico
plt.xlabel('Número de Tópicos')
plt.ylabel('Coherencia')
plt.title('Relación entre el número de tópicos y la coherencia')
#Mostrar el gráfico
plt.show()


# """Obteniendo el máximo número de tópicos"""
max_coherence_row = lsi_df.loc[lsi_df['Coherencia'].idxmax()]
num_topics_max_coherence = max_coherence_row['Número de Tópicos']

opt_topic  = num_topics_max_coherence
best_model_idx = lsi_df[lsi_df['Número de Tópicos'] == opt_topic].index[0]
best_lsi_model = models[best_model_idx]
best_lsi_model.num_topics

topn = 20
for topic_id, topic in best_lsi_model.print_topics(num_topics=int(opt_topic), num_words=topn):
   print('Topic #'+str(topic_id+1)+':')
   print(topic)
   print()
#-----------------------------------------------------------------------------*    
best_lsi_model.show_topic(1, topn=20)
#-----------------------------------------------------------------------------*    
for n in range(int(opt_topic)):
    print('Topic #'+str(n+1)+':')
    print('='*50)
    d1 = []
    d2 = []
    for term, wt in lsi_bow.show_topic(n, topn=20):
        if wt >= 0:
            d1.append((term, round(wt, 3))) #Si los pesos son positivos
        else:
            d2.append((term, round(wt, 3))) #Si los pesos son negativos
    print('Direction 1:', d1) #Dirección del tema 1
    print('-'*50)
    print('Direction 2:', d2) #Dirección del tema 2
    print('-'*50)
    print()
#-----------------------------------------------------------------------------* 
# Matrices - Singular Value Decomposition (SVD)   
# M = U*S*V^T
# donde:
# U = term_topic
# S = singular_values
# V = topic_document
term_topic = best_lsi_model.projection.u
singular_values = best_lsi_model.projection.s
topic_document = (gensim.matutils.corpus2dense(best_lsi_model[bow_corpus], len(singular_values)).T / singular_values).T
term_topic.shape, singular_values.shape, topic_document.shape
#-----------------------------------------------------------------------------* 
document_topics = pd.DataFrame(np.round(topic_document.T, 5), 
                               columns=['T'+str(i) for i in range(1, int(opt_topic)+1)])
document_topics.head(15)
#-----------------------------------------------------------------------------* 
document_numbers = range(0,3)


for document_number in document_numbers:
    top_topics = list(document_topics.columns[np.argsort(-np.absolute(document_topics.iloc[document_number].values))][:1])
    #total_score = sum()
    print('Document #'+str(document_number)+':')
    print('Dominant Topics (top 3):', top_topics)
    print('Comentario:')
    print(topic_train_corpus.iloc[document_number])
    print()
    
 

topics = [[(term, round(wt, 3)) 
               for term, wt in best_lsi_model.show_topic(n, topn=topn)] 
                   for n in range(0, best_lsi_model.num_topics)]

for idx, topic in enumerate(topics):
    print('Topic #'+str(idx+1)+':')
    print([term for term, wt in topic])
    print()
#-----------------------------------------------------------------------------*    
topics_df = pd.DataFrame([[term for term, wt in topic] for topic in topics], 
                          columns = ['Term'+str(i) for i in range(1, topn+1)],
                          index=['Topic '+str(t) for t in range(1, best_lsi_model.num_topics+1)]).T
topics_df    
#-----------------------------------------------------------------------------*    
pd.set_option('display.max_colwidth', -1)
topics_df = pd.DataFrame([', '.join([term for term, wt in topic]) for topic in topics],
                          columns = ['Terms per Topic'],
                          index=['Topic'+str(t) for t in range(1, best_lsi_model.num_topics+1)]
                          )
topics_df
#-----------------------------------------------------------------------------*    
 
# Transform the BoW corpus using the LSI model
lsi_corpus = best_lsi_model[bow_corpus]

# Create a function to get the dominant topic
for i in document_topics:
    print(i)
    
    

data_topic_fit = datos_topic.loc[index_train_topic]

document_numbers = range(0,len(document_topics))
dominant_topics = []
for document_number in document_numbers:
    top_topics = list(document_topics.columns[np.argsort(-np.absolute(document_topics.iloc[document_number].values))][:1])
    dominant_topics.append(top_topics)
    
corpus_topics_lsi = pd.DataFrame({
    'Document': range(1, len(dominant_topics) + 1),
    'Dominant_Topic': [topic[0] for topic in dominant_topics],
    'Train_topic': topic_train_corpus,
    'cod_mod': data_topic_fit['CODIGO_MODULAR'],
    'obs': data_topic_fit['OBSERVACION_RECEPCION'],
    'flg_cat': data_topic_fit['flg_cat']
})
#----------------------------------------------------------------------------*
flg_cat_dir = 'E:/Mi unidad/dgavidia_minedu/BD USE/NLP/TABLETAS/d_flg_cat_recep.csv'
d_flg_cat_recep = pd.read_csv(flg_cat_dir,  encoding = 'Latin-1')

corpus_topics_lsi = corpus_topics_lsi.merge(d_flg_cat_recep, 
                                how = 'left', 
                                left_on=(['flg_cat']),
                                right_on=(['idcatrecepcion']),
                                indicator = False)
corpus_topics_lsi['des_cat'] = corpus_topics_lsi['flg_cat'].astype(str) + ': ' + corpus_topics_lsi['descatrecepcion']

pd.crosstab(corpus_topics_lsi.des_cat,corpus_topics_lsi.Dominant_Topic)
#----------------------------------------------------------------------------*

corpus_topics_lsi.Dominant_Topic.value_counts(normalize=True)
corpus_topics_lsi.flg_cat.value_counts(normalize=True)
maindir_2 = 'E:/Mi unidad/dgavidia_minedu/BD USE/NLP/'
output_topic_dir =  maindir_2 + 'topic_mdeling.csv'   
data_topic_fit.to_csv(output_topic_dir,  encoding = 'UTF-8')

data_topic_fit.head(10)

pd.crosstab(corpus_topics_lsi.des_cat,corpus_topics_lsi.Dominant_Topic)
#Tópico 1: Problemas relacionados con la entrega y la fecha de recepción.
#Tópico 2: Preocupaciones sobre la falta de tabletas y los problemas con los cargadores solares, así como la incertidumbre en torno a la entrega y asignación de dispositivos.
#Tópico 3: Dificultades para registrar las tabletas y asignarlas correctamente debido a problemas con la UGEL y la falta de información clara.
#Tópico 4: Los directores discuten la devolución de tabletas a la UGEL y los desafíos asociados, como problemas con los chips y la logística de recolección.
#Tópico 5: Los directores abordan problemas con la energía eléctrica en las escuelas y la coordinación necesaria para asegurar un funcionamiento adecuado de las tabletas.
#Tópico 6: Los directores mencionan la necesidad de reconocer los chips de las tabletas y destacan aspectos técnicos, como la instalación de aplicativos y la ubicación de las tabletas en las aulas.
#new_df = df[df['obs'].str.contains("cargador|bateria")]
temp_rev = corpus_topics_lsi[corpus_topics_lsi['flg_cat']==4]
temp_rev = corpus_topics_lsi[(corpus_topics_lsi['obs'].str.contains("internet|datos|chip"))==True] #t
emp_rev = datos_topic[(datos_topic['obs'].str.contains("dev")) & (datos_topic['flg_cat']!=16)]#.flg_cat.value_counts()
#%%% LDA (Latent Dirichlet Allocation)-Bow
from gensim.test.utils import common_corpus, common_dictionary
from gensim.models.ldamodel import LdaModel


min_topics = 2
max_topics = 10
#Definir una lista para almacenar los resultados
coherence_scores = []
models = []
perplexity = []
#Iterar sobre los números de tópicos y calcular la coherencia
from datetime import datetime
start = datetime.now()
for num_topics in range(min_topics, max_topics + 1):
    lda_model = gensim.models.LdaModel(corpus=bow_corpus, 
                                       id2word=dictionary,
                                       chunksize=1740, 
                                       alpha='auto',
                                       eta='auto', 
                                       random_state=42,
                                       iterations=500, 
                                       num_topics=num_topics,
                                       passes=20, 
                                       eval_every=None)    
    coherence_model = CoherenceModel(model=lda_model, texts=norm_corpus_bigrams, corpus=bow_corpus, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    coherence_scores.append((num_topics, coherence_score))
    perplexity_res = lda_model.log_perplexity(bow_corpus)
    perplexity.append((num_topics, perplexity_res))
    models.append(lda_model)
print(datetime.now() - start)
# Crear un dataframe a partir de los resultados
df = pd.DataFrame(coherence_scores, columns=['Número de Tópicos', 'Coherencia'])

# Imprimir el dataframe
print(df)

# Crear un gráfico de línea para los resultados
plt.plot(df['Número de Tópicos'], df['Coherencia'], marker='o')
# Añadir etiquetas y título al gráfico
plt.xlabel('Número de Tópicos')
plt.ylabel('Coherencia')
plt.title('Relación entre el número de tópicos y la coherencia')
#Mostrar el gráfico
plt.show()
# """Obteniendo el máximo número de tópicos"""
max_coherence_row = df.loc[df['Coherencia'].idxmax()]
num_topics_max_coherence = max_coherence_row['Número de Tópicos']

topn = 20
opt_topic  = num_topics_max_coherence
best_model_idx = df[df['Número de Tópicos'] == opt_topic].index[0]
best_lda_model = models[best_model_idx]
best_lda_model.num_topics



for topic_id, topic in best_lda_model.print_topics(num_topics=opt_topic, num_words=topn):
   print('Topic #'+str(topic_id+1)+':')
   print(topic)
   print()

topics = [[(term, round(wt, 3)) 
               for term, wt in best_lda_model.show_topic(n, topn=topn)] 
                   for n in range(0, best_lda_model.num_topics)]

for idx, topic in enumerate(topics):
    print('Topic #'+str(idx+1)+':')
    print([term for term, wt in topic])
    print()

topics_df = pd.DataFrame([[term for term, wt in topic] for topic in topics], 
                          columns = ['Term'+str(i) for i in range(1, topn+1)],
                          index=['Topic '+str(t) for t in range(1, best_lda_model.num_topics+1)]).T
topics_df

pd.set_option('display.max_colwidth', -1)
topics_df = pd.DataFrame([', '.join([term for term, wt in topic])  
                              for topic in topics],
                          columns = ['Terms per Topic'],
                          index=['Topic'+str(t) for t in range(1, best_lda_model.num_topics+1)]
                          )
topics_df

# """Se genera el siguiente prompt en GPT y se redactáron las siguientes categorías

# En un proyecto de distribución a tabletas se registráron los comentarios de directores de escuelas sobre la recepción de las tabletas.
# Redacta cateogrías breves que resuman cada tópico.
# Los tópicos generados son:


tm_results = best_lda_model[bow_corpus]
corpus_topics_lda = [sorted(topics, key=lambda record: -record[1])#[0] 
                      for topics in tm_results]

corpus_topic_df = pd.DataFrame()
corpus_topic_df['Document'] = range(0, len(norm_corpus_bigrams))
corpus_topic_df['Dominant Topic'] = [item[0]+1 for item in corpus_topics_lda]
corpus_topic_df['Contribution %'] = [round(item[1]*100, 2) for item in corpus_topics_lda]
corpus_topic_df['Topic Desc'] = [topics_df.iloc[t[0]]['Terms per Topic'] for t in corpus_topics_lda]
corpus_topic_df['Paper'] = norm_corpus_bigrams


print(corpus_topic_df['Dominant Topic'].value_counts())
print(pd.isnull(corpus_topic_df['Dominant Topic']).sum())
print(len(corpus_topic_df['Dominant Topic']))
print(len(index_train_topic))

#"""## Fit Topic"""

#topic_train_corpus, topic_test_corpus, topic_train_label_nums, topic_test_label_nums 

data_topic_fit = datos_topic.loc[index_train_topic]
print(data_topic_fit.shape)
print(corpus_topic_df.shape)
data_topic_fit.reset_index(drop=True, inplace=True)
corpus_topic_df.reset_index(drop=True, inplace=True)

data_topic_fit['Topic'] = corpus_topic_df['Dominant Topic']
cuadro  = pd.crosstab(data_topic_fit['Topic'],data_topic_fit['flg_cat'])

maindir_2 = 'E:/Mi unidad/dgavidia_minedu/BD USE/NLP/'
output_topic_dir =  maindir_2 + 'topic_mdeling.csv'   
data_topic_fit.to_csv(output_topic_dir,  encoding = 'UTF-8')

data_topic_fit.head(10)

# #datos_comp['Topic'] = np.where((datos_comp.OBSERVACION_RECEPCION.str.contains('(?i)(fal|solo|mal estado|malog)')) & (datos_comp['Topic']==10),5,datos_comp['Topic'])
# #datos_comp['Topic'] = np.where((datos_comp.OBSERVACION_RECEPCION.str.contains('(?i)(ning)(.+)(obs|noved)(.+)')),10,datos_comp['Topic'])
# #datos_comp['Topic'] = np.where((datos_comp.OBSERVACION_RECEPCION.str.contains('(?i)(no hay observ)(.+)')),10,datos_comp['Topic'])
# #datos_comp['Topic'] = np.where((datos_comp.OBSERVACION_RECEPCION.str.contains('(?i)(buen)(.+)(estado)(.+)')) &
# #                               (datos_comp.OBSERVACION_RECEPCION.str.contains('(?i)(no|devol|sin|falt|falla|pero|malograda)')==False),
# #                               10,datos_comp['Topic'])

# temp_analisis = data_topic_fit[((data_topic_fit['Topic']==1)) & (data_topic_fit['Target_2']==3)]
# temp_analisis.iloc[:30]



# temp

# temp_export_dir = '/content/drive/MyDrive/dgavidia_minedu/BD USE/NLP/data_topic_fit.xlsx'
# data_topic_fit.to_excel(temp_export_dir)

# datos.shape