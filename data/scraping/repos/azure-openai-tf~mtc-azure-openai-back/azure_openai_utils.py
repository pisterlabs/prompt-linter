"""
@created_by ayaan
@created_at 2023.05.08
"""
import os
import requests
import time
from custom_exception import APIException
from utils.common_utils import CommonUtils
from collections import OrderedDict
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import QueryType
from azure.search.documents import SearchClient
from azure.search.documents.indexes.aio import SearchIndexerClient
from azure.search.documents.indexes.aio import SearchIndexClient

# LangCahin & OpenAI 패키지
import openai
from langchain.chat_models import AzureChatOpenAI

# from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings import OpenAIEmbeddings

from dotenv import load_dotenv

# Log DB Insert
from database import MysqlEngine
from domains.models import ChatRequestHistory

load_dotenv()


class AzureOpenAIUtils:
    """Azure OpenAI Utilities"""

    azure_search_endpoint = os.environ.get("SEARCH_ENDPOINT")
    azure_search_api_version = "2021-04-30-preview"
    content_field = "pages"
    sourcepage_field = "metadata_storage_name"
    chatgpt_deployment = "gpt-35-turbo"
    cognitive_search_credential = AzureKeyCredential(os.environ.get("SEARCH_KEY"))
    headers = {"Content-Type": "application/json", "api-key": os.environ.get("SEARCH_KEY")}
    params = {"api-version": "2021-04-30-preview"}

    search_top = 2
    azure_openai_key = os.environ.get("OPEN_AI_KEY")
    azure_openai_endpoint = os.environ.get("OPEN_AI_ENDPOINT")
    azure_openai_api_version = "2023-03-15-preview"

    openai.api_type = "azure"
    openai.api_version = "2023-03-15-preview"  # 최신 preview 버전 2023-03-15-preview
    openai.api_base = os.environ.get("OPEN_AI_ENDPOINT")
    openai.api_key = os.environ.get("OPEN_AI_KEY")

    prompt_prefix = """<|im_start|>system
Find the contents of the query in Source. If you found it, and if you couldn't find it, say you '자료를 찾지 못하였습니다.
All answers are in Korean.
Query:
{query}
Sources:
{sources}
<|im_end|>
"""
    follow_up_questions_prompt_content = """Generate three very brief follow-up questions that the user would likely ask next about their healthcare plan and employee handbook. 
    Use double angle brackets to reference the questions, e.g. <<Are there exclusions for prescriptions?>>.
    Try not to repeat questions that have already been asked.
    Only generate questions and do not generate any text before or after the questions, such as 'Next Questions'"""

    def nonewlines(self, text: str) -> str:
        return text.replace("\n", " ").replace("\r", " ")

    async def get_index(self, index_name):
        """cognitive Search Get Index"""
        search_index_client = SearchIndexClient(self.azure_search_endpoint, self.cognitive_search_credential)
        index = await search_index_client.get_index(index_name)
        await search_index_client.close()
        return index

    async def get_indexer(self, indexer_name):
        """cognitive Search Get Indexer"""
        search_indexer_client = SearchIndexerClient(self.azure_search_endpoint, self.cognitive_search_credential)
        indexer = await search_indexer_client.get_index(indexer_name)
        await search_indexer_client.close()
        return indexer

    async def get_index_list(self):
        """cognitive Search Get Index List"""
        search_index_client = SearchIndexClient(self.azure_search_endpoint, self.cognitive_search_credential)
        index_list = []
        async for index in search_index_client.list_indexes():
            index_list.append({"index_name": index.name})

        await search_index_client.close()

        return index_list

    async def get_indexer_list(self):
        """cognitive Search Get Indexer List"""
        search_indexer_client = SearchIndexerClient(self.azure_search_endpoint, self.cognitive_search_credential)
        indexer_list = []
        for indexer in await search_indexer_client.get_indexers():
            indexer_list.append({"indexer_name": indexer.name, "target_index_name": indexer.target_index_name})

        await search_indexer_client.close()

        return indexer_list

    async def cognitive_search_run_indexer(self, index_name):
        """cognitive_search_run_indexer"""
        search_indexer_client = SearchIndexerClient(self.azure_search_endpoint, self.cognitive_search_credential)
        # indexer = await search_indexer_client.get_indexer(indexer_name)
        await search_indexer_client.run_indexer(index_name)

        await search_indexer_client.close()

    async def cognitive_search_get_indexer_status(self, indexer_name):
        """cognitive_search_get_indexer_status"""
        search_indexer_client = SearchIndexerClient(self.azure_search_endpoint, self.cognitive_search_credential)
        # indexer = await search_indexer_client.get_indexer(indexer_name)
        result = await search_indexer_client.get_indexer_status(indexer_name)

        await search_indexer_client.close()
        return result

    async def query_openai(self, query, messages):
        """Query Open AI

        Args:
            query (str): 질의
            messages (list): Messages

        Returns:
            dict: messages & answer
        """
        if len(messages) == 0:
            messages = [
                {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
                {"role": "user", "content": query},
            ]
        else:
            messages.append({"role": "user", "content": query})

        response = openai.ChatCompletion.create(engine=self.chatgpt_deployment, messages=messages)
        messages.append(response["choices"][0]["message"])

        result = {"messages": messages, "answer": response["choices"][0]["message"]["content"]}

        return result

    async def execute_openai(self, session, question, index_name, vector_store_name):
        """Excute OpenAI"""
        return_dict = {"question": question, "answer": "자료를 찾지 못하였습니다.", "reference_file": []}
        # 로그 저장
        start = time.time()
        chat_request_history = ChatRequestHistory(selected_index=index_name, query=question, user_id=22, status=ChatRequestHistory.Statues.running)
        session.add(chat_request_history)
        session.commit()

        # Call Cognitive API
        url = self.azure_search_endpoint + "/indexes/" + index_name + "/docs"
        params = {
            "api-version": self.azure_search_api_version,
            "search": question,
            "queryLanguage": "en-US",
            "queryType": "semantic",
            "semanticConfiguration": "semantic-config",
            "select": "*",
            "$count": "true",
            "speller": "lexicon",
            "$top": self.search_top,
            "answers": "extractive|count-3",
            "captions": "extractive|highlight-false",
        }

        resp = requests.get(url, params=params, headers=self.headers)
        search_results = resp.json()  # 결과값

        if resp.status_code != 200:
            chat_request_history.answer = return_dict["answer"]
            chat_request_history.status = ChatRequestHistory.Statues.fail
            chat_request_history.response_at = CommonUtils.get_kst_now()
            chat_request_history.running_time = CommonUtils.get_running_time(start, time.time())
            session.commit()
            raise APIException(resp.status_code, "Cognitive Search API 실패", error=resp.json())

        if search_results["@odata.count"] == 0:
            chat_request_history.answer = return_dict["answer"]
            chat_request_history.status = ChatRequestHistory.Statues.success
            chat_request_history.response_at = CommonUtils.get_kst_now()
            chat_request_history.running_time = CommonUtils.get_running_time(start, time.time())
            session.commit()
            return return_dict
        else:
            file_content = OrderedDict()
            for result in search_results["value"]:
                if result["@search.rerankerScore"] > 0.03:  # Semantic Search 최대 점수 4점
                    file_content[result["metadata_storage_path"]] = {
                        "chunks": result["pages"][:1],
                        "caption": result["@search.captions"][0]["text"],
                        "score": result["@search.rerankerScore"],
                        "file_name": result["metadata_storage_name"],
                    }

            # AzureOpenAI Service 연결
            docs = []
            for key, value in file_content.items():
                for page in value["chunks"]:
                    docs.append(Document(page_content=page, metadata={"source": value["file_name"]}))

            if len(docs) == 0:
                chat_request_history.answer = return_dict["answer"]
                chat_request_history.status = ChatRequestHistory.Statues.success
                chat_request_history.response_at = CommonUtils.get_kst_now()
                chat_request_history.running_time = CommonUtils.get_running_time(start, time.time())
                session.commit()
                return return_dict

            # Embedding 모델 생성
            # 아래소스에서 chunk_size=1 이 아닌 다른 값을 넣으면 다음 소스에서 에러가 난다.
            embeddings = OpenAIEmbeddings(
                model="text-embedding-ada-002", chunk_size=1, openai_api_key=self.azure_openai_key
            )  # Azure OpenAI embedding 사용시 주의

            # Vector Store 생성
            vector_store = FAISS.from_documents(docs, embeddings)
            # if vector_store_name == 'Chroma':
            # persist_directory = "db"
            # 	vector_store = Chroma.from_documents(docs, embeddings)
            # 	vector_store = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory)

            # LangChain🦜 & Azure GPT🤖 연결
            llm = AzureChatOpenAI(
                deployment_name=self.chatgpt_deployment,
                openai_api_key=self.azure_openai_key,
                openai_api_base=self.azure_openai_endpoint,
                openai_api_version=self.azure_openai_api_version,
                temperature=0.0,
                max_tokens=1000,
            )

            # https://python.langchain.com/en/latest/modules/chains/index_examples/qa_with_sources.html
            qa = RetrievalQAWithSourcesChain.from_chain_type(
                llm=llm,
                # chain_type='stuff',
                chain_type="map_reduce",
                retriever=vector_store.as_retriever(),
                return_source_documents=True,
            )

            prompt_prefix = f"""
            Find the contents of the query in Source. If you found it, and if you couldn't find it, say you '자료를 찾지 못하였습니다.
            Please answer all answers in Korean with honorific like '~입니다'
            question: {question}
            """
            result = qa({"question": prompt_prefix})

            print("질문 :", question)
            print("답변 :", result["answer"])
            print("📄 참고 자료 :", result["sources"].replace(",", "\n"))
            return_dict["answer"] = result["answer"].replace("\n", "")
            return_dict["reference_file"] = result["sources"].split(",")
            # return_dict['reference_file_link'] = result["sources"]

            chat_request_history.answer = result["answer"]
            chat_request_history.status = ChatRequestHistory.Statues.success
            chat_request_history.response_at = CommonUtils.get_kst_now()
            chat_request_history.running_time = CommonUtils.get_running_time(start, time.time())
            chat_request_history.reference_file = result["sources"].replace(",", "\n")
            session.commit()

            return return_dict

    async def execute_openai_v2(self, session, question, index_name, vector_store_name):
        """Excute OpenAI"""
        return_dict = {"question": question, "answer": "자료를 찾지 못하였습니다.", "reference_file": []}
        # 로그 저장
        start = time.time()
        chat_request_history = ChatRequestHistory(selected_index=index_name, query=question, user_id=22, status=ChatRequestHistory.Statues.running)
        session.add(chat_request_history)
        session.commit()

        search_client = SearchClient(endpoint=self.azure_search_endpoint, index_name=index_name, credential=self.cognitive_search_credential)

        # Step1. Cognitive Search 질문
        response = search_client.search(
            question,
            # filter=filter,
            query_type=QueryType.SEMANTIC,
            query_language="en-us",
            query_speller="lexicon",
            semantic_configuration_name="semantic-config",
            top=self.search_top,
            query_caption="extractive|highlight-false",
        )

        # Step2. Parsing
        reference_files = [doc[self.sourcepage_field] for doc in response]
        results = [doc[self.sourcepage_field] + ": " + self.nonewlines(doc[self.content_field][0]) for doc in response]
        content = "\n".join(results)

        # Step3. Prompt Engineering
        # follow_up_questions_prompt = self.follow_up_questions_prompt_content
        prompt = self.prompt_prefix.format(sources=content, query=question)

        completion = openai.Completion.create(
            engine=self.chatgpt_deployment,
            prompt=prompt,
            temperature=0.7,
            max_tokens=1024,
            n=1,
            stop=["<|im_end|>", "<|im_start|>"],
        )

        return_dict["answer"] = completion.choices[0].text
        return_dict["reference_file"] = reference_files

        return return_dict

        # return {
        #     "data_points": results,
        #     "answer": completion.choices[0].text,
        #     "thoughts": f"Searched for:<br>{question}<br><br>Prompt:<br>" + prompt.replace("\n", "<br>"),
        # }
