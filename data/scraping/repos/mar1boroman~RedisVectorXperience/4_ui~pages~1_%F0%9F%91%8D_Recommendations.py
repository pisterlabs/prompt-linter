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

load_dotenv()
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"
INDEX_NAME = "idx:blogs"
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


def find_docs(r: redis.Redis, query_vector, selected_author="*"):
    """
    Finds 3 similar docs in redis based on user prompt using vector similarity search
    """

    author_pre_filter = selected_author
    if selected_author != "*":
        author_pre_filter = "@author:{" + selected_author + "}"
        EXPLANATION.append(
            f"Before performing a vector similarity search, the documents are filtered based on author selected.\nIn this case, the author selected is **{selected_author}**"
        )
    else:
        EXPLANATION.append(
            f"The results are not filtered before performing vector similarity search since no specific author is selected"
        )

    responses = []
    query = (
        Query(f"({author_pre_filter})=>[KNN 3 @vector $query_vector AS vector_score]")
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
        EXPLANATION.append("No docs found after vector similarity search based on defined similarity threshold")

    return responses


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
        tagger_component(
            "*Vector Similarity Score*",
            [doc["vector_score"]],
            color_name=["blue"],
        )


def get_explanation():
    expl_doc = ""
    for i, txt in enumerate(EXPLANATION):
        expl_doc += f"{i+1} : {txt}<br><br>"
    return expl_doc


def get_all_authors(r: redis.Redis):
    options = r.ft(INDEX_NAME).tagvals("author")
    options.insert(0, "*")
    return options


def main():
    r = get_redis_conn()
    st.set_page_config()
    colored_header(
        label="Hyper Personalize your search with Hybrid Searches",
        description="Personalize the experience for the user based on filters.",
        color_name="violet-60",
    )
    form = st.form(key="search_form")
    prompt = form.text_input(label="Enter some text")
    submit = form.form_submit_button(label="Submit")
    author_options = get_all_authors(r=r)
    selected_author = form.selectbox(label="Preferred Author", options=author_options)

    if submit:
        prompt_embedding = get_embedding(prompt)
        results = find_docs(r, prompt_embedding, selected_author=selected_author)
        render_results(results)
        print(results)

    with form.expander(label="Execution Log"):
        st.markdown(get_explanation(), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
