import json, operator, requests, time, pycm, wikipedia, pandas as pd
from langchain import HuggingFaceHub, OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from conceptual_engineering_toolkit import Concept, Entity
from datetime import datetime
from string import Template
from pathlib import Path

DIRECTORY = "wd_experiments_v3"

KNOWLEDGE_GRAPH = "https://query.wikidata.org/sparql"

LARGE_LANGUAGE_MODEL = "gpt-4"

TEMPERATURE = 0.1

QUERY_HEADERS = {
    'User-Agent': 'ConceptualEngineeringAgent/0.2 (https://github.com/bradleypallen/conceptual-engineering-using-llms; b.p.allen@uva.nl)',
}

QUERY_LIMIT = 20

ENTITY_TRIPLES_QUERY = Template("""SELECT DISTINCT ?s ?p ?o WHERE {
{ 
  VALUES ?s { <$id> }
  ?s ?p ?o . 
  FILTER(LANG(?o) = "en") .
}
UNION
{ 
  VALUES ?o { <$id> }
  ?s ?p ?o . 
  FILTER(LANG(?o) = "en") .
}
}
LIMIT $limit
""")
                                
WIKIPEDIA_ARTICLE_QUERY = Template("""SELECT ?articleTitle WHERE {
    ?article schema:about <$id> ;
            schema:inLanguage "en" ;
            schema:isPartOf <https://en.wikipedia.org/> ;
            schema:name ?articleTitle .
}
""")
                                  
P_DEFINITION = Template("""Using the following set of RDF statements, 
define the concept "$label". Work set by step and check your facts. State your definition in the manner 
of a dictionary.
                                                           
$serialization'
""")

P_DESCRIPTION = Template("""Summarize the following set of RDF statements 
describing the entity "$label". Work set by step and check your facts. State your summarization 
in the manner of the first paragraph of an encylopedia article on the topic.
                                                   
$serialization'
""")
                      
P_RATIONALE = Template("""
""")

P_CLASSIFICATION = Template("""
""")
                      
def llm(model_name, temperature):
    if model_name in [ "gpt-4", "gpt-3.5-turbo" ]:
        return ChatOpenAI(model_name=model_name, temperature=temperature)
    elif model_name in [ "text-curie-001" ]:
        return OpenAI(model_name=model_name, temperature=temperature)
    elif model_name in [ "meta-llama/Llama-2-70b-chat-hf", "google/flan-t5-xxl" ]:
        return HuggingFaceHub(repo_id=model_name, model_kwargs={ "temperature": temperature })
    else:
        raise Exception(f'Model {model_name} not supported')

def serialize(e, graph=KNOWLEDGE_GRAPH):
    headers = QUERY_HEADERS
    headers['Accept'] = 'text/tab-separated-values'
    query = ENTITY_TRIPLES_QUERY.substitute({"id": e, "limit": QUERY_LIMIT})
    response = requests.get(graph, params={'query' : query}, headers=headers)
    response.raise_for_status()
    return response.text.removeprefix('?s	?p	?o\n')

def summarize(e, label):
    query = WIKIPEDIA_ARTICLE_QUERY.substitute({"id": e})
    response = requests.get(KNOWLEDGE_GRAPH, params={'query' : query, 'format' : 'json'}, headers=QUERY_HEADERS)
    response.raise_for_status()
    json = response.json()
    if len(json["results"]["bindings"]) > 0:
        title = json["results"]["bindings"][0]["articleTitle"]["value"]
        return label + ": " + wikipedia.summary(title, auto_suggest=False).replace('"', r'\"')
    else:
        return label

def define(llm, c, label, serialization):
    return llm.predict(P_DEFINITION.substitute({"label": label, "serialization": serialization}))

def describe(llm, e, label, serialization):
    return llm.predict(P_DESCRIPTION.substitute({"label": label, "serialization": serialization}))

def rationalize(llm, definition, description):
    pass

def classify(llm, definition, description):
    pass

# def evaluate(cls):
#     positives = positive_examples(cls)
#     negatives = negative_examples(cls)
#     concept = Concept(cls["id"], cls["label"], class_definition(cls["id"], cls["label"]), "gpt-4", 0.1)
#     df_positives = pd.DataFrame.from_records(positives)
#     df_positives["actual"] = "positive"
#     df_negatives = pd.DataFrame.from_records(negatives)
#     df_negatives["actual"] = "negative"
#     df_data = pd.concat([df_positives, df_negatives], ignore_index=True, axis=0)
#     df_data["description"] = df_data.apply(lambda ex: instance_description(ex["id"], ex["label"]), axis=1)
#     predictions = [ concept.classify(Entity(ex["id"], ex["label"], ex["description"])) for ex in df_data.to_dict("records") ]
#     df_predictions = pd.DataFrame(predictions, columns = [ 'predicted', 'rationale' ])
#     df_predictions["predicted"] = df_predictions["predicted"].str.lower()
#     df_results = pd.concat([df_data, df_predictions], axis=1)
#     cm = pycm.ConfusionMatrix(df_results["actual"].tolist(), df_results["predicted"].tolist(), digit=2, classes=[ 'positive', 'negative' ])
#     evaluation = { "created": datetime.now().isoformat(), "concept": concept.to_json(), "data": df_results.to_dict('records'), "confusion_matrix": cm.matrix, }
#     experiment_filename = f'{DIRECTORY}/{cls["label"].replace(" ","_")}/{evaluation["concept"]["model_name"]}_{evaluation["concept"]["label"].replace(" ","_")}_{evaluation["created"]}.json'
#     experiment_path = Path(experiment_filename)
#     experiment_path.parent.mkdir(parents=True, exist_ok=True)
#     json.dump(evaluation, open(experiment_filename, 'w+'))