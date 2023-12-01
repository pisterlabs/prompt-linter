from nltk.tokenize import word_tokenize
import nltk
from google.cloud import bigquery
import os, sys
from sklearn.metrics.pairwise import cosine_similarity
import embeddings
import numpy as np
import anthropic
import flask, json
import logging
from flask_cors import CORS

logging.basicConfig(stream=sys.stdout, format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
#nltk.download('all-nltk')
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"/home/fromamine/.config/gcloud/application_default_credentials.json"
api_key = os.environ.get('CLAUDE_API_KEY')
anthropic_client = anthropic.Client(api_key)
app = flask.Flask(__name__)
CORS(app)
client = bigquery.Client()

def get_query_preprocessed(query):
    logging.info(f"START - QUERY PREPROCESSING {query}")
    return word_tokenize(query.lower())

def get_similar_docs(query_preprocessed : list) -> list:
    logging.info("START - SIMILAR DOCS TO QUERY USING TFIDF")
    query_preproccesed_str = '"'+'","'.join(query_preprocessed)+'"'

    query = f"""
    SELECT id, sum(score) as total_score 
    FROM `cobalt-deck-389420.search_wiki_10000.tfidf_wiki_10000`
    WHERE word IN ({query_preproccesed_str})
    GROUP BY id
    HAVING total_score > 0
    ORDER BY total_score desc
    """
        
    query_job = client.query(query)
    logging.info("END - SIMILAR DOCS TO QUERY USING TFIDF")
    return query_job

def get_relevant_docs_embeddings(docs):
    logging.info("START - RELEVANT DOCS EMBEDDINGS")
    docs_ids = []

    for doc in docs:
        docs_ids.append(doc["id"])


    docs_ids_str = ", ".join(str(doc_id) for doc_id in docs_ids)

    query = f"""
    SELECT id, embedding
    FROM `cobalt-deck-389420.search_wiki_10000.embeddings_wiki_10000`
    WHERE id in ({docs_ids_str})
    """
        
    query_job = client.query(query)
    logging.info("END - RELEVANT DOCS EMBEDDINGS")
    return query_job

def get_filtered_relevant_docs(docs, query):
    logging.info("START - FILTERED RELEVANT DOCS")
    model = embeddings.get_model("multi-qa-mpnet-base-dot-v1")

    txt_join = " ".join(query)
    embeddings_query = model.encode(txt_join)

    cosine_similarity_scores = []

    for doc in docs:
        doc_embedding = doc["embedding"].strip('[]')
        doc_embedding = [float(token) for token in doc_embedding.split()]

        doc_embedding = np.array(doc_embedding)
        
        cosine_similarity_score = cosine_similarity(embeddings_query.reshape(1, -1), doc_embedding.reshape(1, -1))
        if cosine_similarity_score > 0.4:
            cosine_similarity_scores.append((cosine_similarity_score[0][0], doc["id"]))
    
    cosine_similarity_scores.sort(reverse=True)
    logging.info("END - FILTERED RELEVANT DOCS")
    return cosine_similarity_scores[:3]

def get_docs_data(docs_scored):
    logging.info("START - DOCS DATA")
    docs_ids_str = ", ".join(str(doc_id) for _, doc_id in docs_scored)

    query = f"""
        SELECT id, url, title, text
        FROM `cobalt-deck-389420.search_wiki_10000.wiki_10000`
        WHERE id in ({docs_ids_str})
    """

    query_job = client.query(query)

    docs_data_dict_list = []
    i = 1
    for row in query_job:
        doc_data_dict = {"id": i, "url": row["url"], "text": row["text"]}
        docs_data_dict_list.append(doc_data_dict)
        i += 1
    logging.info("END - DOCS DATA")
    return docs_data_dict_list

def get_llm_answer(docs_scored, query):
    logging.info(f"START - LLM ANSWER - Number of docs considered: {len(docs_scored)}")

    prompt = f"""
    Your only knowledge: {docs_scored},
    Question: {{ {query} }}

    Answer the question only if the answer is in the knowledge available; otherwise tell me you don't know it.
    Cite the id and the url.
    """

    payload = prompt
    response = anthropic_client.completion(
    prompt=f"{anthropic.HUMAN_PROMPT}{payload}?{anthropic.AI_PROMPT}",
    stop_sequences = [anthropic.HUMAN_PROMPT],
    model="claude-v1",
    max_tokens_to_sample=100,
    )

    logging.info("END - LLM ANSWER")
    return response["completion"]

@app.route("/query", methods = ["POST"])
def get_results():
    logging.info("REQUEST RECEIVED")
    content = flask.request.get_json(silent=True)
    query = content["query"]
    query_preprocessed = get_query_preprocessed(query)
    pre_filtered_docs = get_similar_docs(query_preprocessed)
    relevant_docs_embeddings = get_relevant_docs_embeddings(pre_filtered_docs)
    relevant_docs_scored = get_filtered_relevant_docs(relevant_docs_embeddings, query_preprocessed)
    docs_data_dict_list = get_docs_data(relevant_docs_scored)
    answer = get_llm_answer(docs_data_dict_list, query)
    
    load = dict()
    load["answer"] = answer
    json_data = json.dumps(load)

    logging.info("REQUEST COMPLETED")
    return json_data

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)