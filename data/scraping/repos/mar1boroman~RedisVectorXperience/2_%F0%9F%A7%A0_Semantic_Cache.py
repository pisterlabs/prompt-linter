import redis
import os
import streamlit as st
import openai
import numpy as np
from redis.commands.search.query import Query
from dotenv import load_dotenv
from streamlit_extras.colored_header import colored_header
from streamlit_extras.tags import tagger_component
from rich import print
from typing import List
import uuid
from transformers import BartTokenizer, BartForConditionalGeneration
import time
import json


load_dotenv()
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"
INDEX_NAME = "idx:blogs"
SEMANTIC_CACHE_PREFIX = "streamlit:semantic_cache"
SEMANTIC_INDEX_NAME = "idx:semantic"
EXPLANATION = []
# Common Functions


def get_embedding(doc):
    EXPLANATION.append(
        f"The app uses the Open AI *{OPENAI_EMBEDDING_MODEL}* API to generate an embedding for the text '{doc}'"
    )
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.Embedding.create(
        input=doc, model=OPENAI_EMBEDDING_MODEL, encoding_format="float"
    )
    embedding = response["data"][0]["embedding"]
    return embedding


def get_redis_conn() -> redis.Redis:
    redis_host, redis_port, redis_user, redis_pass = (
        os.getenv("redis_host"),
        os.getenv("redis_port"),
        os.getenv("redis_user"),
        os.getenv("redis_pass"),
    )
    if not redis_user:
        r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
    else:
        r = redis.Redis(
            host=redis_host,
            port=redis_port,
            username=redis_user,
            password=redis_pass,
            decode_responses=True,
        )
    return r


def find_docs(r: redis.Redis, query_vector):
    """
    Finds 3 similar docs in redis based on user prompt using vector similarity search
    """
    responses = []
    query = (
        Query(f"(*)=>[KNN 3 @vector $query_vector AS vector_score]")
        .sort_by("vector_score")
        .return_fields("vector_score", "id", "url", "author", "date", "title", "text")
        .dialect(2)
    )

    query_vector_bytes = np.array(query_vector, dtype=np.float32).tobytes()
    result_docs = (
        r.ft(INDEX_NAME).search(query, {"query_vector": query_vector_bytes}).docs
    )

    EXPLANATION.append(
        f"A vector similarity operation occurs between the prompt embedding and the ~700 blog embeddings stored in Redis"
    )
    EXPLANATION.append(
        f"A KNN 3 search is performed and only the documents with a vector similarity score > 0.85 are returned back to the app"
    )

    docs_str = f"**Similar Documents**\n"
    for doc in result_docs:
        vector_score = round(1 - float(doc.vector_score), 2)
        if vector_score > 0.85:
            responses.append(
                {
                    "title": doc.title,
                    "url": doc.url,
                    "author": doc.author,
                    "date": doc.date,
                    "text": doc.text,
                    "vector_score": vector_score,
                }
            )
            docs_str += f"- Doc Accepted : {doc.title} , Score :{vector_score}\n"
        else:
            docs_str += f"- Doc Rejected : {doc.title} , Score :{vector_score}\n"

    if result_docs:
        EXPLANATION.append(docs_str)
    else:
        EXPLANATION.append(
            "No docs found after vector similarity search based on defined similarity threshold"
        )

    return responses


def get_summary(text: List[str]) -> List[str]:
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    input_ids = tokenizer.batch_encode_plus(
        text, truncation=True, padding=True, return_tensors="pt", max_length=1024
    )["input_ids"]
    summary_ids = model.generate(input_ids, max_length=500)
    summaries = [
        tokenizer.decode(
            s, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for s in summary_ids
    ]
    return summaries


def summarize_results(results: List):
    response = []
    for doc in results:
        EXPLANATION.append(
            f"Use the [facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn) model to generate a summary for the blog *{doc['title']}*"
        )
        doc["summary"] = get_summary([doc["text"]])[0]
        response.append(doc)
    return response


def render_results(results):
    for doc in results:
        st.markdown(
            """
            <style>
            a {
                text-decoration: none;
            }
            a:hover {
                text-decoration: none;
                color: black;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.subheader(body=f"[{doc['title']}]({doc['url']})", divider="rainbow")

        # st.markdown(f"Visit the [link]({doc['url']})")
        st.text(f"{doc['author']} | {doc['date']}")
        st.markdown(doc["summary"])
        tagger_component(
            "*Vector Similarity Score*",
            [doc["vector_score"]],
            color_name=["blue"],
        )
        st.markdown("<br>", unsafe_allow_html=True)


def cache_results(r: redis.Redis, prompt, prompt_embedding, results):
    uid = uuid.uuid4().hex[:6].upper()
    keyname = SEMANTIC_CACHE_PREFIX + ":" + uid
    cache_obj = {
        "prompt": prompt,
        "results": json.dumps(results),
        "prompt_embedding": prompt_embedding,
    }
    r.json().set(name=keyname, path="$", obj=cache_obj)
    EXPLANATION.append(f"Prompt : {prompt} cached with results for further use")


def search_semantic_cache(r: redis.Redis, query_vector):
    """
    Finds 3 similar docs in redis based on user prompt using vector similarity search
    """
    query_vector_bytes = np.array(query_vector, dtype=np.float32).tobytes()
    responses = []
    query = (
        Query(f"(*)=>[KNN 1 @vector $query_vector AS vector_score]")
        .sort_by("vector_score")
        .return_fields("vector_score", "prompt", "results")
        .dialect(2)
    )
    result_docs = (
        r.ft(SEMANTIC_INDEX_NAME)
        .search(query, {"query_vector": query_vector_bytes})
        .docs
    )

    EXPLANATION.append(
        f"A vector similarity operation occurs between the prompt embedding and the cached prompt embeddings stored in Redis"
    )
    EXPLANATION.append(
        f"A KNN 1 search is performed and only the prompts with a vector similarity score > 0.85 are returned back to the app"
    )

    docs_str = f'<span style="background-color: #c7d9b7"> **Similar Prompts** </span>\n'
    for doc in result_docs:
        vector_score = round(1 - float(doc.vector_score), 2)
        if vector_score > 0.85:
            responses.append(
                {
                    "prompt": doc.prompt,
                    "results": json.loads(doc.results),
                    "vector_score": vector_score,
                }
            )
            docs_str += f"- Prompt Accepted : {doc.prompt} , Score :{vector_score}\n"
        else:
            docs_str += f"- Prompt Rejected : {doc.prompt} , Score :{vector_score}\n"

    if result_docs:
        EXPLANATION.append(docs_str)
    else:
        EXPLANATION.append(
            "No cached responses found after vector similarity search based on defined similarity threshold"
        )

    return responses


def clear_semantic_cache(r:redis.Redis):
    print('Clearing semantic cache')
    keys = [key for key in r.scan_iter(match=f"{SEMANTIC_CACHE_PREFIX}*")]
    if keys:
        r.delete(*keys)
    EXPLANATION.append('Semantic Cache deleted')
        

def get_explanation():
    expl_doc = ""
    for i, txt in enumerate(EXPLANATION):
        expl_doc += f"{i+1} : {txt}<br>\n\n"
    return expl_doc



def main():
    r = get_redis_conn()
    st.set_page_config()
    colored_header(
        label="Scale Your App using Semantic Caching",
        description="Semantic Caching allows your search to respond to similar questions using cached responses.",
        color_name="violet-60",
    )
    form = st.form(key="search_form")
    prompt = form.text_input(label="Enter some text")
    submit = form.form_submit_button(label="Submit")
    clear_cache = form.form_submit_button(label="Clear Semantic Cache")

    if submit:
        prompt_embedding = get_embedding(prompt)

        response_time_start = time.time()

        cached_response = search_semantic_cache(r=r, query_vector=prompt_embedding)

        if cached_response:
            result = cached_response[0]
            summarized_results = result["results"]
        else:
            results = find_docs(r, prompt_embedding)
            summarized_results = summarize_results(results=results)
            cache_results(r=r, prompt=prompt, prompt_embedding=prompt_embedding, results=summarized_results)

        render_results(summarized_results)

        EXPLANATION.insert(
            0,
            f'Total Response time : <span style="background-color: #FFE66D">{round(time.time() - response_time_start, 2)} seconds</span>',
        )

    with form.expander(label="Execution Log"):
        st.markdown(get_explanation(), unsafe_allow_html=True)

    if clear_cache:
        clear_semantic_cache(r=r)

if __name__ == "__main__":
    main()
