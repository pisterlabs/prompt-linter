def limpiar (frase):
    frase= frase.lower()
    doc = nlp(frase)
    lista_limpia = [token.txt for token in doc if not token.is_space and not token.is_punct and not token.is_whitespace]
    return lista_limpia

import gensim
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation, preprocess_string
from gensim import corpora
from gensim.models import LsiModel
from gensim.models.coherencemodel import CoherenceModel
from pprint import pprint
from matplotlib import pyplot as plt
import pyLDAvis
import pyLDAvis.gensim

mensajes=["Me encanta la tortilla de patata con cebolla", "No puedo creer que haya gente que tome la tortilla con cebolla", "sincebollista hasta la muerte, abajo la cebolla", "La cebolla no se toca","ojalá desaparezcan del mundo todos los cebollistas", "yo por mi cebolla mato","cada vez que veo una tortilla de patata con cebolla, me dan una puñalada en el corazon","la tortilla con chorizo es la mejor","cada vez que le echan cebolla a una tortilla, muere un gatito","no me gusta la tortilla","la tortilla mejor con cebolla"]

def ejecutarLSA(mensajes, min_topics, max_topics):
  mensajes_preparados=[limpiar(mensaje) for mensaje in mensajes]
  #les doy corpus y asigno números
  dic=corpora.Dictionary(mensajes_preparados)
  #vemos cuantas veces aparece palabra y lo guardo en variable llamada corpus
  corpus=[dic.doc2bow(text) for text in mensajes_preparados]
  models=[]
  coherences=[]
  for num_topics in range (min_topics,max_topics-1):
    #generamos modelo con diccionario de palabras
    lsa=LsiModel(corpus,num_topics=num_topics,id2word=dic)
    #modelo de coherencia mira cuan parecidas son las frases entre si convirtiendolas en vectores
    coherence_model_lsa=CoherenceModel(model=lsa, texts=mensajes_preparados,dictionary=dic,coherence='c-v')
    coherence_lsa=coherence_model_lsa.get_coherence()
    models.append(lsa)
    coherences.append(coherence_lsa)
    return(dic,coherences,models)

  #obtenemos un nivel de coherencia de mimimimimi

def plot_graph(min_topics,max_topics,coherences,path):
  x=range(min_topics,max_topics-1)
  plt.plot(x,coherences)
  plt.xlabel("Numero de temas")
  plt.ylavel("Coherencia")
  plt.legend("valores de coherencia", loc='best')
  plt.savefig(path) #path la ruta donde guardamos la imagen que se genere mimimi

  


(dic_lsa, coherencias_lsa, modelos_lsa)=ejecutarLSA(mensajes,2,10)
plot_graph=(2,10,coherencias_lsa,"lucia.png")