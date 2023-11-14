""""Agent model."""

import qdrant_client
from dotenv import load_dotenv
from langchain import OpenAI, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant

load_dotenv()


class Agent:
    """Agent model."""

    qa: RetrievalQA

    def __init__(self):
        llm = OpenAI(temperature=0.4, max_tokens=300)
        embeddings = OpenAIEmbeddings(client=llm)
        client = qdrant_client.QdrantClient(url="http://localhost/qdrant")
        qdrant = Qdrant(
            client=client,
            collection_name="langchain-service",
            embedding_function=embeddings.embed_query,
        )
        retriever = qdrant.as_retriever()

        template = """You are a knowledgeable and helpful support agent, dedicated to providing accurate and professional answers. Based on the context provided, please answer the user's question. If you do not have enough information to answer the question, kindly respond that you do not know the answer.

        Context: {context}

        User Question: {question}

        Agent's Answer:"""

        prompt_template = PromptTemplate(
            input_variables=["question", "context"], template=template
        )

        self.qa = RetrievalQA.from_llm(
            llm=llm, prompt=prompt_template, retriever=retriever
        )

    def run(self, query: str) -> str:
        """Run the agent on a query."""
        return self.qa.run(query).strip()
