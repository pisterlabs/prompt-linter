import os
from datetime import datetime
from tempfile import NamedTemporaryFile
from typing import Tuple, List, Dict, Any, Union

import anthropic
import langsmith.utils
import openai
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.tracers.langchain import LangChainTracer, wait_for_all_tracers
from langchain.callbacks.tracers.run_collector import RunCollectorCallbackHandler
from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI, ChatAnyscale, ChatAnthropic
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory, StreamlitChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.schema.document import Document
from langchain.schema.retriever import BaseRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langsmith.client import Client
from streamlit_feedback import streamlit_feedback

from qagen import get_rag_qa_gen_chain
from summarize import get_rag_summarization_chain

__version__ = "0.0.11"

# --- Initialization ---
st.set_page_config(
    page_title=f"langchain-streamlit-demo v{__version__}",
    page_icon="🦜",
)


def st_init_null(*variable_names) -> None:
    for variable_name in variable_names:
        if variable_name not in st.session_state:
            st.session_state[variable_name] = None


st_init_null(
    "chain",
    "client",
    "doc_chain",
    "document_chat_chain_type",
    "llm",
    "ls_tracer",
    "provider",
    "retriever",
    "run",
    "run_id",
    "trace_link",
)

# --- Memory ---
STMEMORY = StreamlitChatMessageHistory(key="langchain_messages")
MEMORY = ConversationBufferMemory(
    chat_memory=STMEMORY,
    return_messages=True,
    memory_key="chat_history",
)


# --- Callbacks ---
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


RUN_COLLECTOR = RunCollectorCallbackHandler()


# --- Model Selection Helpers ---
MODEL_DICT = {
    "gpt-3.5-turbo": "OpenAI",
    "gpt-4": "OpenAI",
    "claude-instant-v1": "Anthropic",
    "claude-2": "Anthropic",
    "meta-llama/Llama-2-7b-chat-hf": "Anyscale Endpoints",
    "meta-llama/Llama-2-13b-chat-hf": "Anyscale Endpoints",
    "meta-llama/Llama-2-70b-chat-hf": "Anyscale Endpoints",
    "codellama/CodeLlama-34b-Instruct-hf": "Anyscale Endpoints",
}
SUPPORTED_MODELS = list(MODEL_DICT.keys())


# --- Constants from Environment Variables ---
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "gpt-3.5-turbo")
DEFAULT_SYSTEM_PROMPT = os.environ.get(
    "DEFAULT_SYSTEM_PROMPT",
    "You are a helpful chatbot.",
)
MIN_TEMP = float(os.environ.get("MIN_TEMPERATURE", 0.0))
MAX_TEMP = float(os.environ.get("MAX_TEMPERATURE", 1.0))
DEFAULT_TEMP = float(os.environ.get("DEFAULT_TEMPERATURE", 0.7))
MIN_MAX_TOKENS = int(os.environ.get("MIN_MAX_TOKENS", 1))
MAX_MAX_TOKENS = int(os.environ.get("MAX_MAX_TOKENS", 100000))
DEFAULT_MAX_TOKENS = int(os.environ.get("DEFAULT_MAX_TOKENS", 1000))
DEFAULT_LANGSMITH_PROJECT = os.environ.get("LANGCHAIN_PROJECT")
PROVIDER_KEY_DICT = {
    "OpenAI": os.environ.get("OPENAI_API_KEY", ""),
    "Anthropic": os.environ.get("ANTHROPIC_API_KEY", ""),
    "Anyscale Endpoints": os.environ.get("ANYSCALE_API_KEY", ""),
    "LANGSMITH": os.environ.get("LANGCHAIN_API_KEY", ""),
}
OPENAI_API_KEY = PROVIDER_KEY_DICT["OpenAI"]

MIN_CHUNK_SIZE = 1
MAX_CHUNK_SIZE = 10000
DEFAULT_CHUNK_SIZE = 1000

MIN_CHUNK_OVERLAP = 0
MAX_CHUNK_OVERLAP = 10000
DEFAULT_CHUNK_OVERLAP = 0

DEFAULT_RETRIEVER_K = 4


@st.cache_data
def get_texts_and_retriever(
    uploaded_file_bytes: bytes,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    k: int = DEFAULT_RETRIEVER_K,
) -> Tuple[List[Document], BaseRetriever]:
    with NamedTemporaryFile() as temp_file:
        temp_file.write(uploaded_file_bytes)
        temp_file.seek(0)

        loader = PyPDFLoader(temp_file.name)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        texts = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

        bm25_retriever = BM25Retriever.from_documents(texts)
        bm25_retriever.k = k

        faiss_vectorstore = FAISS.from_documents(texts, embeddings)
        faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": k})

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5],
        )

        return texts, ensemble_retriever


# --- Sidebar ---
sidebar = st.sidebar
with sidebar:
    st.markdown("# Menu")

    model = st.selectbox(
        label="Chat Model",
        options=SUPPORTED_MODELS,
        index=SUPPORTED_MODELS.index(DEFAULT_MODEL),
    )

    st.session_state.provider = MODEL_DICT[model]

    provider_api_key = PROVIDER_KEY_DICT.get(
        st.session_state.provider,
    ) or st.text_input(
        f"{st.session_state.provider} API key",
        type="password",
    )

    if st.button("Clear message history"):
        STMEMORY.clear()
        st.session_state.trace_link = None
        st.session_state.run_id = None

    # --- Document Chat Options ---
    with st.expander("Document Chat", expanded=False):
        uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

        openai_api_key = (
            provider_api_key
            if st.session_state.provider == "OpenAI"
            else OPENAI_API_KEY
            or st.sidebar.text_input("OpenAI API Key: ", type="password")
        )

        document_chat = st.checkbox(
            "Document Chat",
            value=False,
            help="Uploaded document will provide context for the chat.",
        )

        k = st.slider(
            label="Number of Chunks",
            help="How many document chunks will be used for context?",
            value=DEFAULT_RETRIEVER_K,
            min_value=1,
            max_value=10,
        )

        chunk_size = st.slider(
            label="Number of Tokens per Chunk",
            help="Size of each chunk of text",
            min_value=MIN_CHUNK_SIZE,
            max_value=MAX_CHUNK_SIZE,
            value=DEFAULT_CHUNK_SIZE,
        )
        chunk_overlap = st.slider(
            label="Chunk Overlap",
            help="Number of characters to overlap between chunks",
            min_value=MIN_CHUNK_OVERLAP,
            max_value=MAX_CHUNK_OVERLAP,
            value=DEFAULT_CHUNK_OVERLAP,
        )

        chain_type_help_root = (
            "https://python.langchain.com/docs/modules/chains/document/"
        )
        chain_type_help = "\n".join(
            f"- [{chain_type_name}]({chain_type_help_root}/{chain_type_name})"
            for chain_type_name in (
                "stuff",
                "refine",
                "map_reduce",
                "map_rerank",
            )
        )
        document_chat_chain_type = st.selectbox(
            label="Document Chat Chain Type",
            options=[
                "stuff",
                "refine",
                "map_reduce",
                "map_rerank",
                "Q&A Generation",
                "Summarization",
            ],
            index=0,
            help=chain_type_help,
            disabled=not document_chat,
        )

        if uploaded_file:
            if openai_api_key:
                (
                    st.session_state.texts,
                    st.session_state.retriever,
                ) = get_texts_and_retriever(
                    uploaded_file_bytes=uploaded_file.getvalue(),
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    k=k,
                )
            else:
                st.error("Please enter a valid OpenAI API key.", icon="❌")

    # --- Advanced Options ---
    with st.expander("Advanced Options", expanded=False):
        st.markdown("## Feedback Scale")
        use_faces = st.toggle(label="`Thumbs` ⇄ `Faces`", value=False)
        feedback_option = "faces" if use_faces else "thumbs"

        system_prompt = (
            st.text_area(
                "Custom Instructions",
                DEFAULT_SYSTEM_PROMPT,
                help="Custom instructions to provide the language model to determine style, personality, etc.",
            )
            .strip()
            .replace("{", "{{")
            .replace("}", "}}")
        )
        temperature = st.slider(
            "Temperature",
            min_value=MIN_TEMP,
            max_value=MAX_TEMP,
            value=DEFAULT_TEMP,
            help="Higher values give more random results.",
        )

        max_tokens = st.slider(
            "Max Tokens",
            min_value=MIN_MAX_TOKENS,
            max_value=MAX_MAX_TOKENS,
            value=DEFAULT_MAX_TOKENS,
            help="Higher values give longer results.",
        )

        # --- API Keys ---
        LANGSMITH_API_KEY = PROVIDER_KEY_DICT.get("LANGSMITH") or st.text_input(
            "LangSmith API Key (optional)",
            type="password",
        )
        LANGSMITH_PROJECT = DEFAULT_LANGSMITH_PROJECT or st.text_input(
            "LangSmith Project Name",
            value="langchain-streamlit-demo",
        )
        if st.session_state.client is None and LANGSMITH_API_KEY:
            st.session_state.client = Client(
                api_url="https://api.smith.langchain.com",
                api_key=LANGSMITH_API_KEY,
            )
            st.session_state.ls_tracer = LangChainTracer(
                project_name=LANGSMITH_PROJECT,
                client=st.session_state.client,
            )


# --- LLM Instantiation ---
if provider_api_key:
    if st.session_state.provider == "OpenAI":
        st.session_state.llm = ChatOpenAI(
            model_name=model,
            openai_api_key=provider_api_key,
            temperature=temperature,
            streaming=True,
            max_tokens=max_tokens,
        )
    elif st.session_state.provider == "Anthropic":
        st.session_state.llm = ChatAnthropic(
            model=model,
            anthropic_api_key=provider_api_key,
            temperature=temperature,
            streaming=True,
            max_tokens_to_sample=max_tokens,
        )
    elif st.session_state.provider == "Anyscale Endpoints":
        st.session_state.llm = ChatAnyscale(
            model_name=model,
            anyscale_api_key=provider_api_key,
            temperature=temperature,
            streaming=True,
            max_tokens=max_tokens,
        )


# --- Chat History ---
if len(STMEMORY.messages) == 0:
    STMEMORY.add_ai_message("Hello! I'm a helpful AI chatbot. Ask me a question!")

for msg in STMEMORY.messages:
    st.chat_message(
        msg.type,
        avatar="🦜" if msg.type in ("ai", "assistant") else None,
    ).write(msg.content)


# --- Current Chat ---
if st.session_state.llm:
    # --- Regular Chat ---
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt + "\nIt's currently {time}.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{query}"),
        ],
    ).partial(time=lambda: str(datetime.now()))

    # --- Chat Input ---
    prompt = st.chat_input(placeholder="Ask me a question!")
    if prompt:
        st.chat_message("user").write(prompt)
        feedback_update = None
        feedback = None

        # --- Chat Output ---
        with st.chat_message("assistant", avatar="🦜"):
            callbacks = [RUN_COLLECTOR]

            if st.session_state.ls_tracer:
                callbacks.append(st.session_state.ls_tracer)

            config: Dict[str, Any] = dict(
                callbacks=callbacks,
                tags=["Streamlit Chat"],
            )
            if st.session_state.provider == "Anthropic":
                config["max_concurrency"] = 5

            use_document_chat = all(
                [
                    document_chat,
                    st.session_state.retriever,
                ],
            )

            full_response: Union[str, None] = None

            message_placeholder = st.empty()
            stream_handler = StreamHandler(message_placeholder)
            callbacks.append(stream_handler)

            def get_rag_runnable():
                if document_chat_chain_type == "Q&A Generation":
                    return get_rag_qa_gen_chain(
                        st.session_state.retriever,
                        st.session_state.llm,
                    )
                elif document_chat_chain_type == "Summarization":
                    return get_rag_summarization_chain(
                        prompt,
                        st.session_state.retriever,
                        st.session_state.llm,
                    )
                else:
                    return RetrievalQA.from_chain_type(
                        llm=st.session_state.llm,
                        chain_type=document_chat_chain_type,
                        retriever=st.session_state.retriever,
                        memory=MEMORY,
                        output_key="output_text",
                    ) | (lambda output: output["output_text"])

            st.session_state.chain = (
                get_rag_runnable()
                if use_document_chat
                else LLMChain(
                    prompt=chat_prompt,
                    llm=st.session_state.llm,
                    memory=MEMORY,
                )
                | (lambda output: output["text"])
            )

            try:
                full_response = st.session_state.chain.invoke(prompt, config)

            except (openai.error.AuthenticationError, anthropic.AuthenticationError):
                st.error(
                    f"Please enter a valid {st.session_state.provider} API key.",
                    icon="❌",
                )

            if full_response is not None:
                message_placeholder.markdown(full_response)

                # --- Tracing ---
                if st.session_state.client:
                    st.session_state.run = RUN_COLLECTOR.traced_runs[0]
                    st.session_state.run_id = st.session_state.run.id
                    RUN_COLLECTOR.traced_runs = []
                    wait_for_all_tracers()
                    try:
                        st.session_state.trace_link = st.session_state.client.read_run(
                            st.session_state.run_id,
                        ).url
                    except langsmith.utils.LangSmithError:
                        st.session_state.trace_link = None
    if st.session_state.trace_link:
        with sidebar:
            st.markdown(
                f'<a href="{st.session_state.trace_link}" target="_blank"><button>Latest Trace: 🛠️</button></a>',
                unsafe_allow_html=True,
            )

    # --- Feedback ---
    if st.session_state.client and st.session_state.run_id:
        feedback = streamlit_feedback(
            feedback_type=feedback_option,
            optional_text_label="[Optional] Please provide an explanation",
            key=f"feedback_{st.session_state.run_id}",
        )

        # Define score mappings for both "thumbs" and "faces" feedback systems
        score_mappings: dict[str, dict[str, Union[int, float]]] = {
            "thumbs": {"👍": 1, "👎": 0},
            "faces": {"😀": 1, "🙂": 0.75, "😐": 0.5, "🙁": 0.25, "😞": 0},
        }

        # Get the score mapping based on the selected feedback option
        scores = score_mappings[feedback_option]

        if feedback:
            # Get the score from the selected feedback option's score mapping
            score = scores.get(
                feedback["score"],
            )

            if score is not None:
                # Formulate feedback type string incorporating the feedback option
                # and score value
                feedback_type_str = f"{feedback_option} {feedback['score']}"

                # Record the feedback with the formulated feedback type string
                # and optional comment
                feedback_record = st.session_state.client.create_feedback(
                    st.session_state.run_id,
                    feedback_type_str,
                    score=score,
                    comment=feedback.get("text"),
                )
                # feedback = {
                #     "feedback_id": str(feedback_record.id),
                #     "score": score,
                # }
                st.toast("Feedback recorded!", icon="📝")
            else:
                st.warning("Invalid feedback score.")

else:
    st.error(f"Please enter a valid {st.session_state.provider} API key.", icon="❌")
