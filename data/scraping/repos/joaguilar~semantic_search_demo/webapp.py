import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import requests
import json
from elasticsearch import Elasticsearch
from nltk.tokenize import sent_tokenize 
from nltk.tokenize import word_tokenize 
from sentence_transformers import SentenceTransformer
import os
import openai

elastic_host  = os.getenv("ELASTIC_HOST")
elastic_port  = os.getenv("ELASTIC_PORT")
elastic_ssl   = os.getenv("ELASTIC_SSL_ASSERT")
elastic_user  = os.getenv("ELASTIC_USER")
elastic_pwd   = os.getenv("ELASTIC_PASSWORD")
elastic_index = os.getenv("ELASTIC_INDEX")

def cut_last_incomplete_sentence(text:str):
    print("cortando:" + "||"+text.rsplit('.',1)[1])
    return text.rsplit('.',1)[0] + '.'

def get_vector(q):
    sentences =[]
    sentences.append(q)
    # print (sentences)

    
    model = SentenceTransformer("gtr-t5-base")
    vector = model.encode(sentences)

    return vector[0]

def query_elastic(index, query, vector):
    es = Elasticsearch(hosts=[elastic_host+":"+elastic_port],
                        ssl_assert_fingerprint=elastic_ssl,
                        basic_auth=(elastic_user,elastic_pwd))

    qs = {
            "query": {
                "match": {
                "sentence":{
                        "query": query,
                        "boost": 0.5
                        }
                }
            }
    }  

    knns = {
                        "field": "sentence-vector",
                        "k": 10,
                        "num_candidates": 100,
                        "query_vector": vector,
                        "boost":2.0
    }

    # print("AAAA:"+str(vector.tolist()))
    r = es.search(
        index=index,
        # query=qs,
        knn=knns,
        size=20
    )

    return r

def query_elastic_articles(index, query, vector):
    es = Elasticsearch(hosts=[elastic_host+":"+elastic_port],
                        ssl_assert_fingerprint=elastic_ssl,
                        basic_auth=(elastic_user,elastic_pwd))
    # print("Searching vector:"+str(vector.tolist()))
    qs = {

        "has_child": {
            "type": "sentence",
            "score_mode": "max", 
            "query": {
                "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": "cosineSimilarity(params.queryVector, 'sentence-vector') + 1.0",
                    "params": {
                    "query_vector": vector.tolist()
                    }
                }
                }
            }, "inner_hits": {
        "_source": ["sentence"], 
        "size": 2, 
        "highlight": {
          "fields": {
            "sentence":{}
          },
          "boundary_scanner": "sentence",
          "pre_tags": [""],
          "post_tags": [""]
        }
      }
        }        
    }  
    # qs = {

    #     "has_child": {
    #         "type": "sentence",
    #         "score_mode": "max", 
    #             "query": {
    #                 "match_all": {}
                    
    #                 }
    #             ,
    #                 "knns": {
    #                     "field": "sentence-vector",
    #                     "k": 5,
    #                     "num_candidates": 100,
    #                     "query_vector": vector,
    #                     "boost":2.0
    #                 }
    # }
    #         , "inner_hits": {
    #     "_source": ["sentence"], 
    #     "size": 2, 
    #     "highlight": {
    #       "fields": {
    #         "sentence":{}
    #       },
    #       "boundary_scanner": "sentence",
    #       "pre_tags": [""],
    #       "post_tags": [""]
    #     }
    #   }
    # }
                
    # print("Searching on index:"+index)
    r = es.search(
        index=index,
        query=qs,
    )

    return r

def vector_as_text(vector, size):
    vector_text = "["
    for i,v in enumerate(vector):
        if (i % 3 == 0):
            vector_text = vector_text +'\n'
        vector_text = vector_text + str(v) + ", "
        if (i > size):
            vector_text = vector_text + "..., "
            break

    vector_text += "]"
    return vector_text


openai.organization = os.getenv("OPENAI_API_ORG")
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.Model.list()

st.title('Demostración de búsqueda semántica')
st.markdown("""
    Esta es una demostración de búsqueda semántica sobre el corpus de la ley de contratación administrativa.
    El corpus fué del siguiente repositorio: https://www.cgr.go.cr/03-documentos/normativa/ley-reg.html
    """)
with st.form('query', clear_on_submit=False):
    query = st.text_input('Consulta', 'Cual es el número del reglamento que rige sobre la ley de contratación administrativa?')
    st.markdown("""
        Ejemplos de Consultas:
        * *Cual es el número del reglamento que rige sobre la ley de contratación administrativa?*
        * *Pueden los alcaldes municipales ocupar cargos en juntas directivas?*,
        * *Cuales son las garantias de los ciudadanos sobre los servicios de la contraloria?*
        * *Es posible hacer un refrendo por medio de medios electrónicos?*
        * *Puede un individuo participar en compras del estado?*
        * *Si un proyecto se atrasa, se retorna la garantía de cumplimiento?*
        * *Cuales son las garantias de los ciudadanos sobre los servicios de la contraloria?*
        * *Que es la ley de contratación administrativa?*
    """)
    index = st.selectbox("Indice",('vector-leycontraloria','vector-full'))
    individual_sentences = st.checkbox("Retornar las oraciones más relevantes?", value=True)
    generate_answer = st.checkbox("Generar una respuesta?", value=False)
    submitted = st.form_submit_button(label="Query")
if not submitted:
    st.stop()

## Temp:
test_sentences = sent_tokenize(query,language="english")
print("#\t","Sentence")
for i, s in enumerate(test_sentences):
    print(str(i),"\t",s)

vector = get_vector(query)


vector_text = vector_as_text(vector,27)


r = {}
if individual_sentences:
    r = query_elastic(index, query, vector)
else:
    r = query_elastic_articles(index, query, vector)

st.text(r)
results = []
context = ""
c_size = 0
MAX_TOKENS = 1800
if individual_sentences:
    for i,hit in enumerate(r['hits']['hits']):
        result = {}
        result['Id'] = hit['_source']['my_id']
        result['Score'] = hit['_score']
        result['Title'] = hit['_source']['title']
        the_size = len(word_tokenize(result["Title"]))
        if (c_size + the_size) < MAX_TOKENS:
            context = context+" "+result['Title']
            c_size = c_size + the_size
        result['sentence'] = hit['_source']['sentence']
        the_size = len(word_tokenize(result["sentence"]))
        if (c_size + the_size) < MAX_TOKENS:
            context = context+" "+result['sentence']
            c_size = c_size + the_size
        av = hit['_source']['sentence-vector']
        # print(hit['_source'])
        result['sentence-vector'] = vector_as_text(av,10)
        results.append(result)
else:
    for i,hit in enumerate(r['hits']['hits']):
        result = {}
        result['Id'] = hit['_source']['my_id']
        result['Title'] = hit['_source']['title']
        result['Score'] = hit['_score']
        content = hit['_source']['content']

        # Inner hits:
        inner_hit = hit["inner_hits"]["sentence"]["hits"]["hits"][0]
        sentence = inner_hit["_source"]["sentence"]
        # print("Sentence: "+sentence)
        content = content.replace(sentence,"<b> *"+sentence+"* </b>")
        # result['Content'] = hit['_source']['content']
        result['Content'] = content
        context = context+" "+result['Content']
        results.append(result)
if (generate_answer):
        #Preprocesar el contexto:
    print(f"El contexto usado consta de {len(word_tokenize(context))} tokens")
    the_prompt = "Contesta la siguiente pregunta como si fueras un abogado "+query+" basado en "+context+". Si no tiene la información diga que no tiene la información."
    print("PROMPT:||"+the_prompt+"||")
    # response = openai.Completion.create(
    #             model="text-davinci-003",
    #             prompt=the_prompt,
    #             temperature=0.2,
    #             max_tokens=80,
    #             n=1,
    #             stop=["."],
                
    #         )
    response = openai.ChatCompletion.create(
        messages=[
            {"role":"user","content":the_prompt},
            {"role":"system","content":"Actúa como si fueras un abogado"}
        ],
        model="gpt-3.5-turbo",
        temperature=0.4,
        max_tokens=250,
        n=1,
        stop=None #["."]
    )
    print(str(response))
    st.markdown("""---""")
    st.markdown("# Generated Response")
    st.markdown(cut_last_incomplete_sentence(response.choices[0].message.content))
    st.markdown("""---""")
# print(results)
# print(json.dumps(results))

st.markdown("## Vector generado:")
st.text(str(vector_text))

st.markdown("## Oraciones seleccionadas:")
if (len(results) > 0):
    df = pd.DataFrame.from_dict(results)
    st.markdown("## Results")
    st.table(df)
    st.markdown("### Results (Raw)")
    st.text(r)
else:
    st.markdown("## No Results")

