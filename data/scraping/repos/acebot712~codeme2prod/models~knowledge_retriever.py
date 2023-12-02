import logging
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models.openai import ChatOpenAI
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.retrievers.web_research import WebResearchRetriever
from langchain.chains import RetrievalQAWithSourcesChain
from dotenv import load_dotenv
import os

load_dotenv()

env = os.environ.get('ENVIRONMENT', 'development')

if env == 'production':
    logging.basicConfig(level=logging.CRITICAL)
else:
    logging.basicConfig(level=logging.DEBUG)
    
logging.getLogger("langchain.retrievers.web_research")

class KnowledgeRetriever:
    """
    A class to retrieve knowledge using OpenAI LLM with Google search.

    Attributes:
    - vectorstore: Chroma vector store with OpenAI embeddings.
    - llm: Chat OpenAI instance.
    - search: Google Search API wrapper instance.
    - web_research_retriever: Web Research Retriever instance.
    """

    def __init__(self, persist_directory="./chroma_db_oai"):
        self.vectorstore = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory=persist_directory)
        self.llm = ChatOpenAI(temperature=0)
        self.search = GoogleSearchAPIWrapper()
        self.web_research_retriever = WebResearchRetriever.from_llm(
            vectorstore=self.vectorstore,
            llm=self.llm,
            search=self.search
        )

    def retrieve_knowledge(self, question: str) -> str:
        """
        Retrieves answer for a given question using the chain.

        Parameters:
        - question: The question for which the answer needs to be retrieved.

        Returns:
        - str: The answer to the question.
        """
        qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm=self.llm, retriever=self.web_research_retriever)
        result = qa_chain({"question": question})
        return result["answer"]
    
    def enhance_prompt_with_context(self, user_input: str) -> str:
        """
        Enhances the user input with context retrieved from the knowledge base.

        Parameters:
        - user_input: The original user input prompt.

        Returns:
        - str: Enhanced input prompt using retrieved context.
        """
        context = self.retrieve_knowledge(user_input)
        enhanced_prompt = f"Context: {context}\nUse all the above info to answer the query: {user_input}"
        return enhanced_prompt


if __name__ == "__main__":
    retriever = KnowledgeRetriever()
    user_input = "Tell me more about Napoleon"
    # answer = retriever.retrieve_knowledge(user_input)
    enhanced_prompt = retriever.enhance_prompt_with_context(user_input)
    print(enhanced_prompt)
    # print(answer)
