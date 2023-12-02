import redis
import os
import streamlit as st
import openai
import numpy as np
from redis.commands.search.query import Query
from dotenv import load_dotenv
from streamlit_extras.colored_header import colored_header
from rich import print

load_dotenv()
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"
OPENAI_TEXT_MODEL = "gpt-3.5-turbo"
INDEX_NAME = "idx:blogs"
CHAT_HISTORY = "streamlit:chat:history"
openai.api_key = os.getenv("OPENAI_API_KEY")
EXPLANATION = []
# Common Functions


def get_explanation():
    expl_doc = ""
    for i, txt in enumerate(EXPLANATION):
        expl_doc += f"{i+1} : {txt}<br><br>"
    return expl_doc


def sticky():
    # make header sticky.
    st.markdown(
        """
            <div class='fixed-header'/>
            <style>
                div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
                    position: sticky;
                    top: 2.875rem;
                    background-color: white;
                    z-index: 999;
                }
                .fixed-header {
                    border-bottom: 1px solid #A0A2E4;
                }
            </style>
        """,
        unsafe_allow_html=True,
    )


def fixed_bottom():
    # make footer fixed.
    st.markdown(
        """
            <div class='fixed-header'/>
            <style>
                div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
                    position: sticky;
                    top: 2.875rem;
                    background-color: white;
                    z-index: 999;
                }
                .fixed-header {
                    border-bottom: 1px solid black;
                }
            </style>
        """,
        unsafe_allow_html=True,
    )


def center_columns():
    st.markdown(
        """
            <style>
                div[data-testid="column"]:nth-of-type(2)
                {
                    text-align: center;
                } 
            </style>
            """,
        unsafe_allow_html=True,
    )


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


def render_chat_history(r: redis.Redis, stream):
    if r.exists(stream):
        chat_history_msgs = r.xrange(stream)
        for ts, message in chat_history_msgs:
            with st.chat_message(message["role"]):
                st.markdown(message["content"], unsafe_allow_html=True)


def clear_chat_history(r: redis.Redis):
    print("Clearing Chat History")
    keys = [key for key in r.scan_iter(match=f"{CHAT_HISTORY}*")]
    if keys:
        r.delete(*keys)


def get_context(r: redis.Redis, prompt):
    """
    Finds 3 similar docs in redis based on user prompt using vector similarity search
    """

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

    query_vector = get_embedding(prompt)
    query_vector_bytes = np.array(query_vector, dtype=np.float32).tobytes()

    context = ""
    query = (
        Query(f"(*)=>[KNN 3 @vector $query_vector AS vector_score]")
        .sort_by("vector_score")
        .return_fields("vector_score", "id", "url", "author", "date", "title", "text")
        .dialect(2)
    )

    result_docs = (
        r.ft(INDEX_NAME).search(query, {"query_vector": query_vector_bytes}).docs
    )

    EXPLANATION.append(
        f"""A vector similarity operation occurs between the prompt embedding and the ~700 blog 
        embeddings stored in Redis.A KNN 3 search is performed and only the documents with a vector 
        similarity score > 0.85 are added to context"""
    )

    docs_str = f"**Similar Documents**\n"
    for doc in result_docs:
        vector_score = round(1 - float(doc.vector_score), 2)
        if vector_score > 0.85:
            context += doc.text
            docs_str += f"- Doc Accepted : {doc.title} , Score :{vector_score}\n"
        else:
            docs_str += f"- Doc Rejected : {doc.title} , Score :{vector_score}\n"

    if result_docs:
        EXPLANATION.append(docs_str + "<br>\n")
        return context
    else:
        EXPLANATION.append(
            "No docs found after vector similarity search based on defined similarity threshold"
        )
        return False

def build_prompt_with_context(prompt, context):
    if context:
        new_prompt = f"Use the following pieces of context only to answer the question at the end.\n"
        new_prompt += "If you don't know the answer, say that you don't know, don't try to make up an answer.\n"
        new_prompt += "\nContext:\n"
        new_prompt += context
        new_prompt += "\nQuestion:\n"
        new_prompt += prompt
        new_prompt += "\nBe succinct in response and answer in bullet points if possible\n"
    else:
        new_prompt = prompt
    
    EXPLANATION.append(f"The following prompt is sent to OPEN AI {OPENAI_TEXT_MODEL} for chat completion")
    EXPLANATION.append(f"\n```{new_prompt}```\n")
    return new_prompt


def main():
    r = get_redis_conn()
    st.set_page_config()

    with st.container():
        colored_header(
            label="LLM Chatbot using RAG",
            description="'Up to date' information and 'accurate information'",
            color_name="violet-60",
        )
        # st.header(body="LLM Chatbot without RAG", divider="violet")
        col1, col2, col3 = st.columns(3)
        center_columns()
        col2.button(
            "Clear Chat History",
            type="primary",
            on_click=clear_chat_history,
            kwargs={"r": r},
        )
        expl = st.expander(label="Execution Log")
        sticky()

    with st.container():
        render_chat_history(r, stream=CHAT_HISTORY)

        if prompt := st.chat_input("Ask me anything!"):
            with st.chat_message("user"):
                st.markdown(prompt)
                r.xadd(CHAT_HISTORY, {"role": "user", "content": prompt})
                EXPLANATION.append(f"Prompt entered by the user : '{prompt}' ")

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

                context = get_context(r, prompt) # Get context using vector similarity search
                enhanced_prompt = build_prompt_with_context(prompt, context) # Build new prompt = old prompt + context
                

                for response in openai.ChatCompletion.create(
                    model=OPENAI_TEXT_MODEL,
                    messages=[{"role": "user", "content": enhanced_prompt}],
                    stream=True,
                ):
                    full_response += response.choices[0].delta.get("content", "")
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                EXPLANATION.append(
                    f"Open AI *{OPENAI_TEXT_MODEL}* responds with a generated response"
                )

                r.xadd(CHAT_HISTORY, {"role": "assistant", "content": full_response})

            expl.markdown(get_explanation(), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
