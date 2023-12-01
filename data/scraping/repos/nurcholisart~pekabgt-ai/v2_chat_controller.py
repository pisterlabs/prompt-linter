import os
import requests
import traceback

from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import RedisChatMessageHistory, ConversationBufferMemory
from langchain.callbacks import get_openai_callback
from langchain.vectorstores.base import VectorStoreRetriever

from fastapi.responses import JSONResponse

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
# OPENAI_API_BASE = "https://peka-bgt.free.beeceptor.com/v1"
OPENAI_API_BASE = "https://api.openai.com/v1"


class V2ChatController:
    def __init__(
        self,
        api_key: str,
        session_id: str,
        question: str,
        system_prompt: str,
        faiss_url: str,
        pkl_url: str,
    ) -> None:
        self.api_key = api_key
        self.session_id = session_id
        self.question = question
        self.system_prompt = system_prompt
        self.faiss_url = faiss_url
        self.pkl_url = pkl_url

        self.condense_prompt_template = self._set_condense_prompt_template()
        self.user_prompt_template = self._set_user_prompt_template()
        self.memory = self._set_memory()
        self.chat_llm = self._set_chat_llm()
        self.embedding_wrapper = self._set_embedding_wrapper()
        self.vectorstore = self._get_vectorstore()
        self.retriever = self._set_retriever()

    def call(self) -> JSONResponse:
        with get_openai_callback() as cb:
            try:
                qa = ConversationalRetrievalChain.from_llm(
                    llm=self.chat_llm,
                    verbose=True,
                    retriever=self.retriever,
                    condense_question_prompt=self.condense_prompt_template,
                    condense_question_llm=self.chat_llm,
                    combine_docs_chain_kwargs={"prompt": self.user_prompt_template},
                    return_source_documents=True,
                    memory=self.memory,
                )

                result = qa({"question": self.question})

                source_documents: list[Document] = result["source_documents"]

                source_documents_dict = []
                for sd in source_documents:
                    source_documents_dict.append(sd.dict())

                dict_response = {
                    "success": True,
                    "question": result["question"],
                    "answer": result["answer"],
                    "chat_history": [],
                    "source_documents": source_documents_dict,
                    "meta": {
                        "prompt_tokens": cb.prompt_tokens,
                        "completion_tokens": cb.completion_tokens,
                        "total_tokens": cb.total_tokens,
                        "total_cost": cb.total_cost,
                        "success_requests": cb.successful_requests,
                    },
                }

                return JSONResponse(dict_response)
            except Exception as e:
                traceback.print_exc(limit=5)

                return JSONResponse(
                    {
                        "success": False,
                        "error": {
                            "humanized_message": "Execution failed",
                            "message": str(e),
                            "type": e.__class__.__name__,
                        },
                        "meta": {
                            "prompt_tokens": cb.prompt_tokens,
                            "completion_tokens": cb.completion_tokens,
                            "total_tokens": cb.total_tokens,
                            "total_cost": cb.total_cost,
                            "success_requests": cb.successful_requests,
                        },
                    },
                    status_code=500,
                )

    def _set_retriever(self) -> VectorStoreRetriever:
        retriever = self.vectorstore.as_retriever()

        retriever.search_type = "similarity"
        retriever.search_kwargs = {"k": 3}

        # retriever.search_type = "similarity_score_threshold"
        # retriever.search_kwargs = {"k": 3, "score_threshold": 0.3}

        return retriever

    def _set_memory(self) -> ConversationBufferMemory:
        return ConversationBufferMemory(
            chat_memory=self.__get_message_history_store(),
            return_messages=True,
            memory_key="chat_history",
            input_key="question",
            output_key="answer",
        )

    def __get_message_history_store(self) -> RedisChatMessageHistory:
        FIFTEEN_MINUTES = 900

        return RedisChatMessageHistory(
            session_id=f"chat_history:{self.session_id}",
            ttl=FIFTEEN_MINUTES,
            url=REDIS_URL,
        )

    def _set_user_prompt_template(self) -> str:
        _template = """CONTEXT:
        {context}

        QUESTION: {question}
        HELPFUL ANSWER:"""

        return PromptTemplate(
            template=("\n" + self.system_prompt + "\n\n" + _template),
            input_variables=["question", "context"],
        )

    def _set_condense_prompt_template(self) -> str:
        _template = """Given the following conversation and a follow up input, rephrase the follow up input to be a standalone input, in its original language.

        Chat History:
        {chat_history}

        Follow Up Input:
        {question}

        Standalone input:
        """

        return PromptTemplate.from_template(_template)

    def _set_chat_llm(self) -> ChatOpenAI:
        # return OpenAI(
        #     openai_api_key=self.api_key,
        #     openai_api_base=OPENAI_API_BASE,
        # )

        return ChatOpenAI(
            openai_api_key=self.api_key,
            model="gpt-3.5-turbo-16k",
            openai_api_base=OPENAI_API_BASE,
        )

    def _set_embedding_wrapper(self) -> OpenAIEmbeddings:
        return OpenAIEmbeddings(
            openai_api_key=self.api_key, openai_api_base=OPENAI_API_BASE
        )

    def _get_vectorstore(self) -> FAISS:
        faiss_file_name = self.faiss_url.split("/")[-1]
        directory_name = faiss_file_name.split(".faiss")[0]
        tmp_directory_name = f"./tmp/{directory_name}"

        if not os.path.exists(tmp_directory_name):
            self.__download_faiss_and_pkl_url()

        vectorstore = FAISS.load_local(
            tmp_directory_name, embeddings=self.embedding_wrapper
        )

        return vectorstore

    def __download_faiss_and_pkl_url(self) -> None:
        get_faiss_resp = requests.get(self.faiss_url, allow_redirects=True)

        faiss_directory_name = self.faiss_url.split("/")[-1].split(".faiss")[0]
        faiss_tmp_directory_name = f"./tmp/{faiss_directory_name}"
        faiss_file_name = f"{faiss_tmp_directory_name}/index.faiss"

        os.makedirs(faiss_tmp_directory_name, exist_ok=True)
        with open(faiss_file_name, "xb") as file:
            file.write(get_faiss_resp.content)

        get_pkl_resp = requests.get(self.pkl_url, allow_redirects=True)

        pkl_directory_name = self.pkl_url.split("/")[-1].split(".pkl")[0]
        pkl_tmp_directory_name = f"./tmp/{pkl_directory_name}"
        pkl_file_name = f"{pkl_tmp_directory_name}/index.pkl"

        os.makedirs(pkl_tmp_directory_name, exist_ok=True)
        with open(pkl_file_name, "xb") as file:
            file.write(get_pkl_resp.content)
