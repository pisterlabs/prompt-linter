import time
import json
import pandas as pd
from typing import List

from qdrant_client import QdrantClient
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryMemory

from backend.config import QDRANT_URL, QDRANT_API_KEY, OPENAI_API_KEY


class NeuralSearcher:

    def __init__(self, collection_name: str):
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )
        self.vector_store = Qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=OpenAIEmbeddings(),
        )
        self.llm = ChatOpenAI(
            model='gpt-3.5-turbo',
            api_key=OPENAI_API_KEY,
        )
        self.memory = ConversationSummaryMemory(
            llm=self.llm,
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
        prompt_template = '''
        About: You are a Product Recommendation Agent who gets his context from the retrieved descriptions of the products that matches best with the User's query. User is a human who, as a customer, wants to buy a product from this application.

        Given below is the summary of conversation between you (AI) and the user (Human):
        Context: {chat_history}

        Now use this summary of previous conversations and the retrieved descriptions of products to answer the following question asked by the user:
        Question: {question}

        Note: While answering the question, give only one short sentence description along with rating and price (in INR ₹) for each retrived product. Do not give any unnecessary information. Also, do not repeat the information that is already present in the context. The answer should be crisp so that it can fit the token limit. The tone of the answer should be like a polite and friendly AI Assistant.
        '''
        self.PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["chat_history", "question"]
        )

    def search(self, question: str, num_results: int, filter_: dict = None) -> dict:
        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={'k':num_results}),
            memory=self.memory,
            return_source_documents=True,
        )
        gen_prompt = self.PROMPT.format(question=question, chat_history=self.memory.load_memory_variables({})['chat_history'][0].content)
        start_time = time.time()
        res = chain(gen_prompt)
        print(f"Search took {time.time() - start_time} seconds")

        ret = {}
        ret['answer'] = res['answer']

        srcs = [json.loads(row.page_content) for row in res['source_documents']]

        df = pd.DataFrame(srcs)
        df = df.fillna('null')
        # df.set_index('product', inplace=True)

        df1 = df[['product','brand', 'sale_price', 'rating', 'description']]

        # Remove duplicates
        df1 = df1.drop_duplicates()

        ret['products'] = df1.to_dict(orient='records')
        return ret