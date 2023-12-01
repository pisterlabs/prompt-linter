import asyncio
import json
import os
from typing import Optional, Type

import cohere
import streamlit as st
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.chains.llm import LLMChain
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage
from langchain.schema.document import Document
from langchain.tools import BaseTool
from pydantic import BaseModel
from xata.client import XataClient

from src.modules.tools.common.search_pinecone import SearchPinecone

llm_model = st.secrets["llm_model"]
langchain_verbose = str(st.secrets["langchain_verbose"])
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
os.environ["COHERE_API_KEY"] = st.secrets["cohere_api_key"]
co = cohere.Client(os.environ["COHERE_API_KEY"])

embeddings = OpenAIEmbeddings()
search_pinecone = SearchPinecone()


class ReviewToolWithDetailedOutlines(BaseTool):
    name = "review_tool_with_detailed_outlines"
    description = "The ReviewToolWithDetailedOutlines is specifically designed to facilitate the review process by leveraging user-provided detailed outlines. When a user supplies a comprehensive outline, this tool systematically searches the knowledge base, retrieving relevant information for each section of the outline. It then integrates the retrieved information to form a cohesive and informative review. This approach ensures that the review is tailored to the user's specific requirements and provides a thorough and insightful evaluation of the content in question."

    class InputSchema(BaseModel):
        query: str

    args_schema: Type[BaseModel] = InputSchema

    def summary_chain(self):
        # langchain_verbose = st.secrets["langchain_verbose"]
        # openrouter_api_key = st.secrets["openrouter_api_key"]
        # openrouter_api_base = st.secrets["openrouter_api_base"]

        # # selected_model = "anthropic/claude-2"
        # # selected_model = "openai/gpt-3.5-turbo-16k"
        # selected_model = "openai/gpt-4-32k"
        # # selected_model = "meta-llama/llama-2-70b-chat"

        # llm_chat = ChatOpenAI(
        #     model_name=selected_model,
        #     temperature=0,
        #     streaming=True,
        #     verbose=langchain_verbose,
        #     openai_api_key=openrouter_api_key,
        #     openai_api_base=openrouter_api_base,
        #     headers={"HTTP-Referer": "http://localhost"},
        #     # callbacks=[],
        # )

        llm_chat = Bedrock(
            credentials_profile_name="default",
            model_id="anthropic.claude-v2",
            streaming=False,
            model_kwargs={"max_tokens_to_sample": 2048, "temperature": 0},
        )

        # chain = load_summarize_chain(llm_chat, chain_type="stuff")

        # Define prompt
        prompt_template = """You a worldclass literature review writter. You must:
        based on the following provided information and your own knowledge, provide a logical, clear, well-organized, and critically analyzed review to respond the query:
        "{query}". 
        You must:
        delve deep into the topic and provide an exhaustive answer;
        ensure review as detailed as possible;
        ensure each section and paragraph are fully discussed with detailed case studies and examples;
        use multiple case studies or examples to support and enrich your arguments;
        ensure review length longer than {length} words as user request;
        give in-text citations where relevant in Author-Date mode, NOT in Numeric mode.

        UPLOADED INFO:
        "{uploaded_docs}".

        KNOWLEDGE BASE Search Results:
        "{pinecone_docs}".

        """

        prompt = PromptTemplate(
            input_variables=["query", "length", "uploaded_docs", "pinecone_docs"],
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
        ensure review length longer than {length} words as user request;
        give in-text citations where relevant in Author-Date mode, NOT in Numeric mode.
        You must not cut off at the end.

        SUMMARIZED INFO:
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
                "length": {
                    "title": "Request Review Length",
                    "description": "The length of the review requested by the user, in words",
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
                content="You are a world class algorithm for extracting the all queries and filters from a chat history, for searching vector database. Give the user's story line, extract and list all the key queries that need to be addressed for a review. Each query should be speccific, independent and structured to facilitate separate searches in a vector database. Make ensure to provide multiple queries to fully cover the user's request. Make sure to answer in the correct structured format."
            ),
            HumanMessage(content="The chat history:"),
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

    # def search_postgres(self):
    #     conn_pg = psycopg2.connect(
    #         database="chat",
    #         user="postgres",
    #         password=st.secrets("postgres_password"),
    #         host=st.secrets("postgres_host"),
    #         port=st.secrets("postgres_port"),
    #     )
    #     query = f"SELECT uuid FROM journals WHERE title LIKE '%dynamic material flow%'LIMIT 5"
    #     cursor = conn_pg.cursor()
    #     cursor.execute(query)
    #     results = cursor.fetchall()
    #     cursor.close()
    #     conn_pg.close()
    #     uuid_for_filter = [item[0] for item in results]
    #     return uuid_for_filter

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool synchronously."""

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        user_original_latest_query = (
            st.session_state["xata_history"].messages[-1].content
        )
        func_calling_outline = self.outline_func_calling_chain().run(
            user_original_latest_query
        )
        outline_response = func_calling_outline.get("query")
        queries = outline_response.split("; ")
        summary_chain = self.summary_chain()

        try:
            created_at = json.loads(func_calling_outline.get("created_at", None))
        except TypeError:
            created_at = None

        length = func_calling_outline.get("length", None)

        filters = {}
        if created_at:
            filters["created_at"] = created_at

        try:
            history = st.session_state["xata_history"].messages[-2].content
        except IndexError:
            history = []

        k = 60
        rerank_response = []
        summary_response = []
        if history == []:
            pinecone_docs = await asyncio.gather(
                *[
                    search_pinecone.async_similarity(query=query, top_k=k)
                    for query in queries
                ]
            )
            # uploaded_docs = await asyncio.gather(
            #     *[
            #         self.search_uploaded_docs(query=query, top_k=k)
            #         for query in queries
            #     ]
            # )
            pinecone_contents = search_pinecone.get_contentslist(pinecone_docs)

            for index, pinecone_content in enumerate(pinecone_contents):
                response = co.rerank(
                    model="rerank-english-v2.0",
                    query=queries[index],
                    documents=pinecone_content,
                    top_n=30,
                )
                result = [result.document["text"] for result in response.results]
                rerank_response.extend(result)

            summary_response = summary_chain.run(
                {
                    "query": user_original_latest_query,
                    "length": length,
                    "uploaded_docs": "",
                    "pinecone_docs": rerank_response,
                },
            )
            # summary_response = await asyncio.gather(
            #     *[
            #         summary_chain.arun(
            #             {
            #                 "query": query,
            #                 "uploaded_docs": "",
            #                 "pinecone_docs": pinecone_doc,
            #             }
            #         )
            #         for query, pinecone_doc in zip(queries, pinecone_docs)
            #     ]
            # )
            # response = review_chain.run(
            #     {
            #         "query": user_original_latest_query,
            #         "summary": summary_response,
            #         "length": length,
            #     },
            # )
            return summary_response
        else:
            return "Go for RefineTool."
