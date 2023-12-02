from datetime import datetime
from typing import Tuple, List, Dict, Any, Union, Optional

import anthropic
import langsmith.utils
import openai
import streamlit as st
from langchain.callbacks.tracers.langchain import LangChainTracer, wait_for_all_tracers
from langchain.callbacks.tracers.run_collector import RunCollectorCallbackHandler
from langchain.memory import ConversationBufferMemory, StreamlitChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.document import Document
from langchain.schema.retriever import BaseRetriever
from langsmith.client import Client
from streamlit_feedback import streamlit_feedback

from defaults import default_values

from llm_resources import get_runnable, get_llm, get_texts_and_retriever, StreamHandler

__version__ = "0.1.2"

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
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
    "AZURE_OPENAI_BASE_URL",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_DEPLOYMENT_NAME",
    "AZURE_OPENAI_EMB_DEPLOYMENT_NAME",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_MODEL_VERSION",
    "AZURE_AVAILABLE",
)

# --- LLM globals ---
STMEMORY = StreamlitChatMessageHistory(key="langchain_messages")
MEMORY = ConversationBufferMemory(
    chat_memory=STMEMORY,
    return_messages=True,
    memory_key="chat_history",
)

RUN_COLLECTOR = RunCollectorCallbackHandler()

st.session_state.LANGSMITH_API_KEY = (
    st.session_state.LANGSMITH_API_KEY
    or default_values.PROVIDER_KEY_DICT.get("LANGSMITH")
)

st.session_state.LANGSMITH_PROJECT = st.session_state.LANGSMITH_PROJECT or (
    default_values.DEFAULT_LANGSMITH_PROJECT or "langchain-streamlit-demo"
)


def azure_state_or_default(*args):
    st.session_state.update(
        {
            arg: st.session_state.get(arg) or default_values.AZURE_DICT.get(arg)
            for arg in args
        },
    )


azure_state_or_default(
    "AZURE_OPENAI_BASE_URL",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_DEPLOYMENT_NAME",
    "AZURE_OPENAI_EMB_DEPLOYMENT_NAME",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_MODEL_VERSION",
)

st.session_state.AZURE_AVAILABLE = all(
    [
        st.session_state.AZURE_OPENAI_BASE_URL,
        st.session_state.AZURE_OPENAI_API_VERSION,
        st.session_state.AZURE_OPENAI_DEPLOYMENT_NAME,
        st.session_state.AZURE_OPENAI_API_KEY,
        st.session_state.AZURE_OPENAI_MODEL_VERSION,
    ],
)

st.session_state.AZURE_EMB_AVAILABLE = (
    st.session_state.AZURE_AVAILABLE
    and st.session_state.AZURE_OPENAI_EMB_DEPLOYMENT_NAME
)

AZURE_KWARGS = (
    None
    if not st.session_state.AZURE_EMB_AVAILABLE
    else {
        "openai_api_base": st.session_state.AZURE_OPENAI_BASE_URL,
        "openai_api_version": st.session_state.AZURE_OPENAI_API_VERSION,
        "deployment": st.session_state.AZURE_OPENAI_EMB_DEPLOYMENT_NAME,
        "openai_api_key": st.session_state.AZURE_OPENAI_API_KEY,
        "openai_api_type": "azure",
    }
)


@st.cache_data
def get_texts_and_retriever_cacheable_wrapper(
    uploaded_file_bytes: bytes,
    openai_api_key: str,
    chunk_size: int = default_values.DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = default_values.DEFAULT_CHUNK_OVERLAP,
    k: int = default_values.DEFAULT_RETRIEVER_K,
    azure_kwargs: Optional[Dict[str, str]] = None,
    use_azure: bool = False,
) -> Tuple[List[Document], BaseRetriever]:
    return get_texts_and_retriever(
        uploaded_file_bytes=uploaded_file_bytes,
        openai_api_key=openai_api_key,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        k=k,
        azure_kwargs=azure_kwargs,
        use_azure=use_azure,
    )


# --- Sidebar ---
sidebar = st.sidebar
with sidebar:
    st.markdown("# Menu")

    model = st.selectbox(
        label="Chat Model",
        options=default_values.SUPPORTED_MODELS,
        index=default_values.SUPPORTED_MODELS.index(default_values.DEFAULT_MODEL),
    )

    st.session_state.provider = default_values.MODEL_DICT[model]

    provider_api_key = (
        default_values.PROVIDER_KEY_DICT.get(
            st.session_state.provider,
        )
        or st.text_input(
            f"{st.session_state.provider} API key",
            type="password",
        )
        if st.session_state.provider != "Azure OpenAI"
        else ""
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
            else default_values.OPENAI_API_KEY
            or st.sidebar.text_input("OpenAI API Key: ", type="password")
        )

        document_chat = st.checkbox(
            "Document Chat",
            value=True if uploaded_file else False,
            help="Uploaded document will provide context for the chat.",
        )

        k = st.slider(
            label="Number of Chunks",
            help="How many document chunks will be used for context?",
            value=default_values.DEFAULT_RETRIEVER_K,
            min_value=1,
            max_value=10,
        )

        chunk_size = st.slider(
            label="Number of Tokens per Chunk",
            help="Size of each chunk of text",
            min_value=default_values.MIN_CHUNK_SIZE,
            max_value=default_values.MAX_CHUNK_SIZE,
            value=default_values.DEFAULT_CHUNK_SIZE,
        )

        chunk_overlap = st.slider(
            label="Chunk Overlap",
            help="Number of characters to overlap between chunks",
            min_value=default_values.MIN_CHUNK_OVERLAP,
            max_value=default_values.MAX_CHUNK_OVERLAP,
            value=default_values.DEFAULT_CHUNK_OVERLAP,
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
        )
        use_azure = st.toggle(
            label="Use Azure OpenAI",
            value=st.session_state.AZURE_EMB_AVAILABLE,
            help="Use Azure for embeddings instead of using OpenAI directly.",
        )

        if uploaded_file:
            if st.session_state.AZURE_EMB_AVAILABLE or openai_api_key:
                (
                    st.session_state.texts,
                    st.session_state.retriever,
                ) = get_texts_and_retriever_cacheable_wrapper(
                    uploaded_file_bytes=uploaded_file.getvalue(),
                    openai_api_key=openai_api_key,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    k=k,
                    azure_kwargs=AZURE_KWARGS,
                    use_azure=use_azure,
                )
            else:
                st.error("Please enter a valid OpenAI API key.", icon="❌")

    # --- Advanced Settings ---
    with st.expander("Advanced Settings", expanded=False):
        st.markdown("## Feedback Scale")
        use_faces = st.toggle(label="`Thumbs` ⇄ `Faces`", value=False)
        feedback_option = "faces" if use_faces else "thumbs"

        system_prompt = (
            st.text_area(
                "Custom Instructions",
                default_values.DEFAULT_SYSTEM_PROMPT,
                help="Custom instructions to provide the language model to determine style, personality, etc.",
            )
            .strip()
            .replace("{", "{{")
            .replace("}", "}}")
        )

        temperature = st.slider(
            "Temperature",
            min_value=default_values.MIN_TEMP,
            max_value=default_values.MAX_TEMP,
            value=default_values.DEFAULT_TEMP,
            help="Higher values give more random results.",
        )

        max_tokens = st.slider(
            "Max Tokens",
            min_value=default_values.MIN_MAX_TOKENS,
            max_value=default_values.MAX_MAX_TOKENS,
            value=default_values.DEFAULT_MAX_TOKENS,
            help="Higher values give longer results.",
        )

    # --- LangSmith Options ---
    if default_values.SHOW_LANGSMITH_OPTIONS:
        with st.expander("LangSmith Options", expanded=False):
            st.session_state.LANGSMITH_API_KEY = st.text_input(
                "LangSmith API Key (optional)",
                value=st.session_state.LANGSMITH_API_KEY,
                type="password",
            )

            st.session_state.LANGSMITH_PROJECT = st.text_input(
                "LangSmith Project Name",
                value=st.session_state.LANGSMITH_PROJECT,
            )

    if st.session_state.client is None and st.session_state.LANGSMITH_API_KEY:
        st.session_state.client = Client(
            api_url="https://api.smith.langchain.com",
            api_key=st.session_state.LANGSMITH_API_KEY,
        )
        st.session_state.ls_tracer = LangChainTracer(
            project_name=st.session_state.LANGSMITH_PROJECT,
            client=st.session_state.client,
        )

    # --- Azure Options ---
    if default_values.SHOW_AZURE_OPTIONS:
        with st.expander("Azure Options", expanded=False):
            st.session_state.AZURE_OPENAI_BASE_URL = st.text_input(
                "AZURE_OPENAI_BASE_URL",
                value=st.session_state.AZURE_OPENAI_BASE_URL,
            )

            st.session_state.AZURE_OPENAI_API_VERSION = st.text_input(
                "AZURE_OPENAI_API_VERSION",
                value=st.session_state.AZURE_OPENAI_API_VERSION,
            )

            st.session_state.AZURE_OPENAI_DEPLOYMENT_NAME = st.text_input(
                "AZURE_OPENAI_DEPLOYMENT_NAME",
                value=st.session_state.AZURE_OPENAI_DEPLOYMENT_NAME,
            )

            st.session_state.AZURE_OPENAI_EMB_DEPLOYMENT_NAME = st.text_input(
                "AZURE_OPENAI_EMB_DEPLOYMENT_NAME",
                value=st.session_state.AZURE_OPENAI_EMB_DEPLOYMENT_NAME,
            )

            st.session_state.AZURE_OPENAI_API_KEY = st.text_input(
                "AZURE_OPENAI_API_KEY",
                value=st.session_state.AZURE_OPENAI_API_KEY,
                type="password",
            )

            st.session_state.AZURE_OPENAI_MODEL_VERSION = st.text_input(
                "AZURE_OPENAI_MODEL_VERSION",
                value=st.session_state.AZURE_OPENAI_MODEL_VERSION,
            )


# --- LLM Instantiation ---
st.session_state.llm = get_llm(
    provider=st.session_state.provider,
    model=model,
    provider_api_key=provider_api_key,
    temperature=temperature,
    max_tokens=max_tokens,
    azure_available=st.session_state.AZURE_AVAILABLE,
    azure_dict={
        "AZURE_OPENAI_BASE_URL": st.session_state.AZURE_OPENAI_BASE_URL,
        "AZURE_OPENAI_API_VERSION": st.session_state.AZURE_OPENAI_API_VERSION,
        "AZURE_OPENAI_DEPLOYMENT_NAME": st.session_state.AZURE_OPENAI_DEPLOYMENT_NAME,
        "AZURE_OPENAI_API_KEY": st.session_state.AZURE_OPENAI_API_KEY,
        "AZURE_OPENAI_MODEL_VERSION": st.session_state.AZURE_OPENAI_MODEL_VERSION,
    },
)

# --- Chat History ---
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

            st.session_state.chain = get_runnable(
                use_document_chat,
                document_chat_chain_type,
                st.session_state.llm,
                st.session_state.retriever,
                MEMORY,
                chat_prompt,
                prompt,
            )

            # --- LLM call ---
            try:
                full_response = st.session_state.chain.invoke(prompt, config)

            except (openai.error.AuthenticationError, anthropic.AuthenticationError):
                st.error(
                    f"Please enter a valid {st.session_state.provider} API key.",
                    icon="❌",
                )

            # --- Display output ---
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
                    except (
                        langsmith.utils.LangSmithError,
                        langsmith.utils.LangSmithNotFoundError,
                    ):
                        st.session_state.trace_link = None

    # --- LangSmith Trace Link ---
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
                st.toast("Feedback recorded!", icon="📝")
            else:
                st.warning("Invalid feedback score.")

else:
    st.error(f"Please enter a valid {st.session_state.provider} API key.", icon="❌")
