import requests
from bs4 import BeautifulSoup
from langchain.callbacks import get_openai_callback
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceHubEmbeddings
from langchain.llms import OpenAIChat, HuggingFaceHub
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from streamlit.logger import get_logger

from utils.constants import (
    KNOWLEDGEBASE_DIR,
    AssistantType,
    BS_HTML_PARSER,
    TEXT_TAG,
    SOURCE_TAG,
    ANSWER_TAG,
    QUESTION_TAG,
    HF_TEXT_GENERATION_REPO_ID,
    EmbeddingType,
    TOTAL_TOKENS_TAG,
    PROMPT_TOKENS_TAG,
    COMPLETION_TOKENS_TAG,
    TOTAL_COST_TAG,
    OPENAI_CHAT_COMPLETIONS_MODEL,
)

logger = get_logger(__name__)


def extract_text_from(url_: str):
    html = requests.get(url_).text
    soup = BeautifulSoup(html, features=BS_HTML_PARSER)
    text = soup.get_text()

    lines = (line.strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def create_knowledgebase(
    urls: list,
    assistant_type: AssistantType,
    embedding_type: EmbeddingType,
    embedding_api_key: str,
    knowledgebase_name: str,
):
    pages: list[dict] = []
    for url in urls:
        pages.append({TEXT_TAG: extract_text_from(url_=url), SOURCE_TAG: url})

    chunk_size = 500
    chunk_overlap = 30
    if assistant_type == AssistantType.OPENAI:
        # # override the default chunk configs
        # chunk_size = 1500
        # chunk_overlap = 200
        if embedding_type == EmbeddingType.HUGGINGFACE:
            embeddings = HuggingFaceHubEmbeddings(
                huggingfacehub_api_token=embedding_api_key
            )
            logger.info(f"Using `hf` embeddings")
        else:
            embeddings = OpenAIEmbeddings(openai_api_key=embedding_api_key)
            logger.info(f"Using `openai` embeddings")
    else:
        embeddings = HuggingFaceHubEmbeddings(
            huggingfacehub_api_token=embedding_api_key
        )
        logger.info(
            f"Since the assistant type is set to `hf`, `hf` embeddings are used by default."
        )

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator="\n"
    )

    docs, metadata = [], []
    for page in pages:
        splits = text_splitter.split_text(page[TEXT_TAG])
        docs.extend(splits)
        metadata.extend([{SOURCE_TAG: page[SOURCE_TAG]}] * len(splits))
        print(f"Split {page[SOURCE_TAG]} into {len(splits)} chunks")

    vectorstore = FAISS.from_texts(texts=docs, embedding=embeddings, metadatas=metadata)
    vectorstore.save_local(folder_path=KNOWLEDGEBASE_DIR, index_name=knowledgebase_name)


def load_vectorstore(
    embedding_type: EmbeddingType,
    embedding_api_key: str,
    knowledgebase_name: str,
):
    if embedding_type == EmbeddingType.OPENAI:
        embeddings = OpenAIEmbeddings(openai_api_key=embedding_api_key)
    else:
        embeddings = HuggingFaceHubEmbeddings(
            huggingfacehub_api_token=embedding_api_key
        )
        logger.info(
            f"Since the assistant type is set to `hf`, `hf` embeddings are used by default."
        )

    store = FAISS.load_local(
        folder_path=KNOWLEDGEBASE_DIR,
        embeddings=embeddings,
        index_name=knowledgebase_name,
    )
    return store


def construct_query_response(result: dict) -> dict:
    return {ANSWER_TAG: result}


class Knowledgebase:
    def __init__(
        self,
        assistant_type: AssistantType,
        embedding_type: EmbeddingType,
        assistant_api_key: str,
        embedding_api_key: str,
        knowledgebase_name: str,
    ):
        self.assistant_type = assistant_type
        self.embedding_type = embedding_type
        self.assistant_api_key = assistant_api_key
        self.embedding_api_key = embedding_api_key
        self.knowledgebase = load_vectorstore(
            embedding_type=embedding_type,
            embedding_api_key=embedding_api_key,
            knowledgebase_name=knowledgebase_name,
        )

    def query_knowledgebase(self, query: str) -> tuple[dict, dict]:
        try:
            logger.info(
                f"The assistant API key for the current session: ***{self.assistant_api_key[-4:]}"
            )
            logger.info(
                f"The embedding API key for the current session: ***{self.embedding_api_key[-4:]}"
            )

            query = query.strip()
            if not query:
                return {
                    ANSWER_TAG: "Oh snap! did you hit send accidentally, because I can't see any questions 🤔",
                }, {}

            if self.assistant_type == AssistantType.OPENAI:
                llm = OpenAIChat(
                    model_name=OPENAI_CHAT_COMPLETIONS_MODEL,
                    temperature=0,
                    verbose=True,
                    openai_api_key=self.assistant_api_key,
                )
                # # this is deprecated
                # chain = VectorDBQAWithSourcesChain.from_llm(
                #     llm=llm,
                #     vectorstore=self.knowledgebase,
                #     max_tokens_limit=2048,
                #     k=2,
                #     reduce_k_below_max_tokens=True,
                # )
                chain = RetrievalQAWithSourcesChain.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=self.knowledgebase.as_retriever(),
                    reduce_k_below_max_tokens=True,
                    chain_type_kwargs={"verbose": True},
                )
            else:
                llm = HuggingFaceHub(
                    repo_id=HF_TEXT_GENERATION_REPO_ID,
                    model_kwargs={"temperature": 0.5, "max_length": 64},
                    huggingfacehub_api_token=self.assistant_api_key,
                    verbose=True,
                )
                chain = RetrievalQAWithSourcesChain.from_chain_type(
                    llm=llm,
                    chain_type="refine",
                    retriever=self.knowledgebase.as_retriever(),
                    max_tokens_limit=1024,
                    reduce_k_below_max_tokens=True,
                    chain_type_kwargs={"verbose": True},
                )

            with get_openai_callback() as cb:
                result = chain({QUESTION_TAG: query})
                print(f"Total Tokens: {cb.total_tokens}")
                print(f"Prompt Tokens: {cb.prompt_tokens}")
                print(f"Completion Tokens: {cb.completion_tokens}")
                print(f"Total Cost (USD): ${cb.total_cost}")

                metadata = {
                    TOTAL_TOKENS_TAG: cb.total_tokens,
                    PROMPT_TOKENS_TAG: cb.prompt_tokens,
                    COMPLETION_TOKENS_TAG: cb.completion_tokens,
                    TOTAL_COST_TAG: cb.total_cost,
                }
            return result, metadata
        except Exception as e:
            logger.error(f"{e.__class__.__name__}: {e}")
            return {ANSWER_TAG: f"{e.__class__.__name__}: {e}"}, {}
