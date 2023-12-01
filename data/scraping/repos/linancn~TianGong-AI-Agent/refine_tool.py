import asyncio
import logging
import os
import json
from datetime import datetime
from typing import Optional, Type

import pinecone
import streamlit as st
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from langchain.schema.document import Document
from langchain.tools import BaseTool
from langchain.vectorstores import Pinecone
from pydantic import BaseModel
from xata.client import XataClient
import psycopg2
from langchain.llms import OpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chains.openai_functions import create_structured_output_chain

llm_model = st.secrets["llm_model"]
langchain_verbose = str(st.secrets["langchain_verbose"])
os.environ["PINECONE_API_KEY"] = st.secrets["pinecone_api_key"]
os.environ["PINECONE_ENVIRONMENT"] = st.secrets["pinecone_environment"]
os.environ["PINECONE_INDEX"] = st.secrets["pinecone_index"]
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
embeddings = OpenAIEmbeddings()
pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT"],
)
vectorstore = Pinecone.from_existing_index(
    index_name=os.environ["PINECONE_INDEX"],
    embedding=embeddings,
)

class ReviewToolWithoutOutlines(BaseTool):
    name = "review_tool_without_detailed_outlines"
    description = "The ReviewToolWithoutOutlines is designed to streamline the review process, especially when users do not provide a detailed outline or only supply a few broad headings. In such cases, this tool takes the initiative to generate a comprehensive outline that captures the essence of the content to be reviewed. It then utilizes this outline as a guide to systematically search the knowledge base and other relevant sources, gathering pertinent information for each section of the outline. Finally, it integrates the gathered information to form a well-structured and insightful review. This approach ensures that the review is thorough, relevant, and tailored to the content, providing valuable insights for the user."

    class InputSchema(BaseModel):
        query: str

    args_schema: Type[BaseModel] = InputSchema

    def summary_chain(self):
        langchain_verbose = st.secrets["langchain_verbose"]
        openrouter_api_key = st.secrets["openrouter_api_key"]
        openrouter_api_base = st.secrets["openrouter_api_base"]

        selected_model = "anthropic/claude-2"
        # selected_model = "openai/gpt-3.5-turbo-16k"
        # selected_model = "openai/gpt-4-32k"
        # selected_model = "meta-llama/llama-2-70b-chat"

        llm_chat = ChatOpenAI(
            model_name=selected_model,
            temperature=0.9,
            streaming=True,
            verbose=langchain_verbose,
            openai_api_key=openrouter_api_key,
            openai_api_base=openrouter_api_base,
            headers={"HTTP-Referer": "http://localhost"},
            # callbacks=[],
        )

        # chain = load_summarize_chain(llm_chat, chain_type="stuff")

        # Define prompt
        prompt_template = """You must:
        based on the following provided information (if any) and your own knowledge, provide a logical, clear, well-organized, and critically analyzed summary to "{query}";
        delve deep into the topic and provide an exhaustive answer;
        ensure summary as detailed as possible;
        ensure summary with detailed case studies and examples;
        give in-text citations where relevant in Author-Date mode, NOT in Numeric mode.

        UPLOADED INFO:
        "{uploaded_docs}".

        KNOWLEDGE BASE:
        "{pinecone_docs}".

        You must not:
        include any duplicate or redundant information."""

        prompt = PromptTemplate(
            input_variables=["query", "uploaded_docs", "pinecone_docs"],
            template=prompt_template,
        )

        chain = LLMChain(
            llm=llm_chat,
            prompt=prompt,
            verbose=langchain_verbose,
        )

        return chain

    def review_chain(self):
        llm_model = st.secrets["llm_model"]
        langchain_verbose = st.secrets["langchain_verbose"]

        llm_chat = ChatOpenAI(
            model=llm_model,
            temperature=0.7,
            streaming=True,
            verbose=langchain_verbose,
            callbacks=[],
        )

        # chain = load_summarize_chain(llm_chat, chain_type="stuff")

        # Define prompt
        prompt_template = """You a worldclass literature review writter. You must:
        based on the following provided information and your own knowledge, provide a logical, clear, well-organized, and critically analyzed review to "{query}";
        ensure multiple sections or paragraphs;
        ensure each section and paragraph are fully discussed with detailed case studies and examples;
        ensure review length longer than {length} words if user request;
        give in-text citations where relevant in Author-Date mode, NOT in Numeric mode.
        You must not cut off at the end.

        SUMMARIZED INFO FOR EACH SECTION OR PARAGRAPH:
        {summary}.

        COMPLETE REVIEW:"""

        prompt = PromptTemplate(
            input_variables=["query", "summary", "length"],
            template=prompt_template,
        )

        chain = LLMChain(
            llm=llm_chat,
            prompt=prompt,
            verbose=langchain_verbose,
        )

        return chain

    def refine_chain(self):
        llm_model = st.secrets["llm_model"]
        langchain_verbose = st.secrets["langchain_verbose"]

        llm_chat = ChatOpenAI(
            model=llm_model,
            temperature=0.7,
            streaming=True,
            verbose=langchain_verbose,
            callbacks=[],
        )

        # Define prompt
        prompt_template = """ You must take the following actions for your response:

        1. Retain the content from your previous answer.
        2. Incorporate, update, or remove text based on "{current_query}".
        3. Use the specific information provided, supplemented with your own knowledge, to enhance the response.
        4. Deliver a polished and thorough review.
        5. Ensure that the response is logical, clear, well-structured, and offers critical analysis.
        6. Aim for maximum detail and depth.

        PREVIOUSE RESPONSE:
        "{history}"

        SEPCIFIC INFO:
        "{sepcific_info}"

        You must not:
        include any duplicate or redundant information."""

        prompt = PromptTemplate(
            input_variables=["current_query", "sepcific_info", "history"],
            template=prompt_template,
        )

        chain = LLMChain(
            llm=llm_chat,
            prompt=prompt,
            verbose=langchain_verbose,
        )

        return chain

    def fetch_uploaded_docs_vector(
        self, query: str, k: int = 16
    ) -> list[Document]:
        """Fetch uploaded docs in similarity search."""
        username = st.session_state["username"]
        session_id = st.session_state["selected_chat_id"]
        embeddings = OpenAIEmbeddings()
        query_vector = embeddings.embed_query(query)
        results = (
            XataClient()
            .data()
            .vector_search(
                "tiangong_chunks",
                {
                    "queryVector": query_vector,  # array of floats
                    "column": "embedding",  # column name,
                    "similarityFunction": "cosineSimilarity",  # space function
                    "size": k,  # number of results to return
                    "filter": {
                        "username": username,
                        "sessionId": session_id,
                    },  # filter expression
                },
            )
        )

        # results = (
        #     XataClient()
        #     .data()
        #     .query(
        #         "tiangong_chunks",
        #         {
        #             "columns": ["content", "source"],
        #             "filter": {"username": username, "sessionId": session_id},
        #         },
        #     )
        # )
        docs = []
        for record in results["records"]:
            page_content = record["content"]
            # metadata = {
            #     "source": record["source"],
            # }
            # doc = Document(
            #     page_content=page_content,
            #     metadata=metadata,
            # )
            docs.append(page_content)

        return docs

    def outline_func_calling_chain(self):
        func_calling_json_schema = {
            "title": "get_querys_and_filters_to_search_database",
            "description": "Extract the queries and filters for database searching",
            "type": "object",
            "properties": {
                "query": {
                    "title": "Query",
                    "description": "Multiple queries extracted for a vector database semantic search from a chat history, separate queries with a semicolon",
                    "type": "string",
                },
                "created_at": {
                    "title": "Date Filter",
                    "description": 'Date extracted for a vector database semantic search, in MongoDB\'s query and projection operators, in format like {"$gte": 1609459200.0, "$lte": 1640908800.0}',
                    "type": "string",
                },
            },
            "required": ["query"],
        }

        prompt_func_calling_msgs = [
            SystemMessage(
                content="You are a world class algorithm for extracting the all queries and filters from a chat history, for searching vector database. Give the initial overview of the topic, extract and list all the key queries that need to be addressed for a full review. Each query should be speccific, independent and structured to facilitate separate searches in a vector database. Make ensure to provide multiple queries to fully cover the user's request. Make sure to answer in the correct structured format."
            ),
            HumanMessage(content="The initial overview:"),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]

        prompt_func_calling = ChatPromptTemplate(messages=prompt_func_calling_msgs)

        llm_func_calling = ChatOpenAI(
            model_name=llm_model, temperature=0, streaming=False
        )

        func_calling_chain = create_structured_output_chain(
            output_schema=func_calling_json_schema,
            llm=llm_func_calling,
            prompt=prompt_func_calling,
            verbose=langchain_verbose,
        )

        return func_calling_chain

    def nextquery_func_calling_chain(self):
        func_calling_json_schema = {
            "title": "get_querys_and_filters_to_search_database",
            "description": "Extract the queries and filters for database searching",
            "type": "object",
            "properties": {
                "query": {
                    "title": "Query",
                    "description": "Next query extracted for a vector database semantic search from a chat history",
                    "type": "string",
                },
                "created_at": {
                    "title": "Date Filter",
                    "description": 'Date extracted for a vector database semantic search, in MongoDB\'s query and projection operators, in format like {"$gte": 1609459200.0, "$lte": 1640908800.0}',
                    "type": "string",
                },
            },
            "required": ["query"],
        }

        prompt_func_calling_msgs = [
            SystemMessage(
                content="You are a world class algorithm for extracting a query and filters from a history, for searching vector database. According to the context text and the specific outline point, provide next query of a complete sentence for searching vector database. Make sure to answer in the correct structured format."
            ),
            HumanMessage(content="The context:"),
            HumanMessagePromptTemplate.from_template("{contex}"),
            HumanMessage(content="The next outline point:"),
            HumanMessagePromptTemplate.from_template("{next_outline_point}"),
        ]

        prompt_func_calling = ChatPromptTemplate(messages=prompt_func_calling_msgs)

        llm_func_calling = ChatOpenAI(
            model_name=llm_model, temperature=0, streaming=False
        )

        func_calling_chain = create_structured_output_chain(
            output_schema=func_calling_json_schema,
            llm=llm_func_calling,
            prompt=prompt_func_calling,
            verbose=langchain_verbose,
        )

        return func_calling_chain

    def search_pinecone(self, query: str, filters: dict = {}, top_k: int = 16):
        """Search Pinecone index for documents similar to query."""
        if top_k == 0:
            return []

        if filters:
            docs = vectorstore.similarity_search(query, k=top_k, filter=filters)
        else:
            docs = vectorstore.similarity_search(query, k=top_k)

        docs_list = []
        for doc in docs:
            # date = datetime.fromtimestamp(doc.metadata["created_at"])
            # formatted_date = date.strftime("%Y-%m")  # Format date as 'YYYY-MM'
            # source_entry = "[{}. {}. {}. {}.]({})".format(
            #     doc.metadata["source_id"],
            #     doc.metadata["source"],
            #     doc.metadata["author"],
            #     formatted_date,
            #     doc.metadata["url"],
            # )
            # docs_list.append({"content": doc.page_content, "source": source_entry})
            docs_list.append(doc.page_content)

        return docs_list

    def search_postgres(self):
        # 连接到 PostgreSQL 数据库
        conn_pg = psycopg2.connect(
            database="chat",
            user="postgres",
            password=st.secrets("postgres_password"),
            host=st.secrets("postgres_host"),
            port=st.secrets("postgres_port"),
        )
        query = f"SELECT uuid FROM journals WHERE title LIKE '%dynamic material flow%'LIMIT 5"
        cursor = conn_pg.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        cursor.close()
        conn_pg.close()
        uuid_for_filter = [item[0] for item in results]
        return uuid_for_filter

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool synchronously."""

        ###单链
        ##search strategy 1: direct search pinecone
        # query = "Dynamic material flow analysis"
        # filter = {"source": "JOURNAL OF INDUSTRIAL ECOLOGY"}
        nextquery_func_calling_chain = self.nextquery_func_calling_chain()
        summary_chain = self.summary_chain()
        review_chain = self.review_chain()

        uploaded_docs = self.fetch_uploaded_docs_vector(query=query, k=2)
        pinecone_docs = self.search_pinecone(
                        query=query, top_k=2
                    )
        overview_response = summary_chain.run({"query": query,"uploaded_docs": uploaded_docs,"pinecone_docs": pinecone_docs})
        func_calling_outline = self.outline_func_calling_chain().run(overview_response)
        query = func_calling_outline.get("query")
        queries = query.split("; ")

        user_original_latest_query = st.session_state["xata_history"].messages[-1].content



        try:
            created_at = json.loads(func_calling_outline.get("created_at", None))
        except TypeError:
            created_at = None


        filters = {}
        if created_at:
            filters["created_at"] = created_at

        try:
            history = st.session_state["xata_history"].messages[-2].content
        except IndexError:
            history = []

        summary_response = []
        result = ""
        if history == []:
            for query in queries:
                if summary_response == []:
                    uploaded_docs = self.fetch_uploaded_docs_vector(query=query, k=2)
                    pinecone_docs = self.search_pinecone(
                        query=query, filters=filters, top_k=2
                    )
                    summary_response = summary_chain.run(
                        {
                            "query": query,
                            "uploaded_docs": uploaded_docs,
                            "pinecone_docs": pinecone_docs,
                        },
                    )
                    result += summary_response
                else:
                    next_query = nextquery_func_calling_chain.run(
                        {"contex": summary_response, "next_outline_point": query}
                    ).get("query")
                    new_uploaded_docs = self.fetch_uploaded_docs_vector(
                        query=next_query, k=2
                    )
                    new_pinecone_docs = self.search_pinecone(
                        query=next_query, filters=filters, top_k=2
                    )
                    summary_response = summary_chain.run(
                        {
                            "query": next_query,
                            "uploaded_docs": new_uploaded_docs,
                            "pinecone_docs": new_pinecone_docs,
                        },
                    )
                    uploaded_docs.extend(new_uploaded_docs)
                    pinecone_docs.extend(new_pinecone_docs)
                    result += summary_response

            response = review_chain.run(
                {
                    "query": latest_query,
                    "summary": result,
                    "length": length,
                },
            )
            return response

        else:
            uploaded_docs = self.fetch_uploaded_docs_vector(query=query, k=2)
            pinecone_docs = self.search_pinecone(query=query, top_k=2)
            summary_chain = self.summary_chain()
            summary_response = summary_chain.run(
                {
                    "query": query,
                    "uploaded_docs": uploaded_docs,
                    "pinecone_docs": pinecone_docs,
                },
            )
            refine_chain = self.refine_chain()
            refine_response = refine_chain(
                {
                    # "current_query": current_query,
                    "sepcific_info": summary_response,
                    "history": history,
                },
            )
            return refine_response

        ###search strategy 2: search title in postgres, then search related docs in pinecone
        # query = "Dynamic material flow analysis"
        # uuid_for_filter = self.search_postgres()
        # filter = {"document_id": {"$in":uuid_for_filter}}
        # pinecone_docs = self.search_pinecone(query=query, filters=filter, top_k=50)
        # main_chain = self.main_chain()
        # response = main_chain.run(pinecone_docs)
        # return response

        ###多链
        ###Overview+Details
        # query = "Dynamic material flow analysis"
        # filter = {"source": "JOURNAL OF INDUSTRIAL ECOLOGY"}
        # pinecone_docs = self.search_pinecone(query=query, filters=filter, top_k=50)
        # main_chain = self.main_chain()
        # overview_response = main_chain.run(pinecone_docs)
        # query_detail = "Dynamic material flow analysis in built environment, metal, housing appliances, etc."
        # pinecone_docs_detail = self.search_pinecone(query=query_detail, filters=filter, top_k=20)
        # detail_chain = self.detail_chain()
        # detail_response = detail_chain.run(pinecone_docs_detail)
        # response = overview_response + detail_response
        # return response

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
      