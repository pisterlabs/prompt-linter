import logging, os
import cohere
import tomli

from dotenv import load_dotenv
from langchain.chat_models import ChatCohere
from langchain.document_loaders import ArxivLoader
from langchain.llms import Cohere
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.retrievers import CohereRagRetriever
from langchain.schema.document import Document
from pydantic import BaseModel, Field, field_validator
from tenacity import retry, stop_after_attempt, wait_random_exponential


class Tweet(BaseModel):
    """
    Pydantic Model to generate an structured Tweet with Validation
    """
    text: str = Field(..., description="Tweet text")

    @field_validator('text')
    def validate_text(cls, v: str) -> str:
        if "https://" not in v and "http://" not in v:
            logging.error("Tweet does not include a link to the paper!")
            raise ValueError("Tweet must include a link to the paper!")
        return v


class Email(BaseModel):
    """
    Pydantic Model to generate an structured Email
    """
    subject: str = Field(..., description="Email subject")
    body: str = Field(..., description="Email body")


class CohereEngine:
    def __init__(self) -> None:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s [%(levelname)s] %(message)s")
        self.vars = self.__load_environment_vars()
        self.cohere = self.__cohere_client(self.vars["COHERE_API_KEY"])
        self.templates = self.__load_prompt_templates()

        logging.info("Initialized CohereEngine")


    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
    def query_article(self, article: str, query: str):
        """
        Query Article.

        Parameters:
        - article (str): Article to query
        - query (str): Query to search for

        Returns:
        - str: Relevant passages from the article
        """
        logging.info("query_llm (started)")
        
        rag = CohereRagRetriever(llm=ChatCohere())
        docs = rag.get_relevant_documents(query, 
                                          source_documents=[Document(page_content=article)])

        #ranked_docs = self.cohere.rerank(query=query, documents=docs, top_n=4, model="rerank-english-v2.0")

        logging.info("query_llm (OK)")
        return docs


    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
    def generate_tweet(self, summary: str, link: str) -> Tweet:
        """
        Generate an structured Tweet object about a research paper.
        Under the hood it uses Cohere's LLM, a custom Pydantic Tweet Model, and Langchain Expression Language with Templates.

        Parameters:
        - summary (str): Summary of the research paper
        - link (str): Link to the research paper

        Returns:
        - Tweet: Tweet object
        """
        logging.info(f"generate_tweet ({link}) (started)")

        model = Cohere(model='command', temperature=0.3, max_tokens=250)
        prompt = PromptTemplate.from_template(self.templates['tweet']['prompt'])
        parser = PydanticOutputParser(pydantic_object=Tweet)

        tweet_chain = prompt | model | parser
        tweet = tweet_chain.invoke({"summary": summary, "link": link})

        logging.info("generate_tweet (OK)")
        return tweet
    

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
    def generate_email(self, sender: str, institution: str, receivers: list, title: str, topic: str) -> Email:     
        """
        Generate an structured Email object to the authors of a research paper.
        Under the hood it uses Cohere's LLM, a custom Pydantic Email Model, and Langchain Expression Language with Templates.

        Parameters:
        - sender (str): Name of the sender
        - institution (str): Institution of the sender
        - receivers (list): Names of the receivers
        - title (str): Title of the research paper
        - topic (str): Topic of the research paper
        """
        logging.info("generate_email (started)")

        model = Cohere(model='command', temperature=0.1, max_tokens=500)
        prompt = PromptTemplate.from_template(self.templates['email']['prompt'])
        parser = PydanticOutputParser(pydantic_object=Email)

        email_chain = prompt | model | parser
        email = email_chain.invoke({"sender": sender, 
                                    "institution": institution, 
                                    "receivers": receivers, 
                                    "title": title, 
                                    "topic": topic})
        
        logging.info("generate_email (OK)")
        return email
    

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
    def enrich_abstract(self, text: str) -> str:
        """
        Identifies technical Named Entities, and enrich them with Wikipedia Links.

        Parameters:
        - text (str): Text to be enriched

        Returns:
        - str: Text enriched with Wikipedia links
        """
        logging.info("enrich_abstract (started)")

        model = Cohere(model='command', temperature=0.3, max_tokens=4096, truncate=None)
        prompt = PromptTemplate.from_template(self.templates['abstract']['prompt'])

        abstract_chain = prompt | model
        abstract = abstract_chain.invoke({"text": text})

        logging.info("enrich_abstract (OK)")
        return abstract
    

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
    def extract_keywords(self, text: str) -> str:
        """
        Extract keywords from a research paper. For each keyword, it provides a brief explanation of its significance in the context of this research.

        Parameters:
        - text (str): Text to extract keywords from

        Returns:
        - str: Keywords extracted from the text
        """
        logging.info("extract_keywords (started)")

        model = Cohere(model='command', temperature=0.1, max_tokens=4096)
        prompt = PromptTemplate.from_template(self.templates['keywords']['prompt'])

        keywords_chain = prompt | model
        keywords = keywords_chain.invoke({"text": text})

        logging.info("extract_keywords (OK)")
        return keywords


    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
    def summarize(self, text: str) -> str:
        logging.info("summarize (started)")

        response = self.cohere.summarize(
            text = text,
            length='auto',
            format='bullets',
            model='command',
            additional_command='',
            temperature=0.8,
        )

        logging.info("summarize (OK)")
        return response.summary
    

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
    def embed(self, texts: dict) -> dict:
        return self.cohere.embed(
            model='embed-english-v3.0',
            texts=texts,
            input_type='search_document',
        ).embeddings
    

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
    def load_arxiv_paper(self, paper_id: str) -> (dict, str):
        logging.info("load_arxiv_paper (started)")

        docs = ArxivLoader(query=paper_id, load_max_docs=2, load_all_available_meta=True).load()
        metadata = docs[0].metadata
        content = docs[0].page_content

        logging.info("load_arxiv_paper (OK)")
        return metadata, content
    

    def __load_environment_vars(self):
        """
        Load environment variables from .env file
        """
        logging.info("load_environment_vars (started)")

        load_dotenv()
        required_vars = ["COHERE_API_KEY"]
        env_vars = {var: os.getenv(var) for var in required_vars}
        for var, value in env_vars.items():
            if not value:
                raise EnvironmentError(f"{var} environment variable not set.")
        
        logging.info("load_environment_vars (OK)")
        return env_vars
    

    def __load_prompt_templates(self):
        """
        Load prompt templates from prompts/athena.toml
        """
        logging.info("load_prompt_templates (started)")

        try:
            with open("prompts/athena.toml", "rb") as f:
                prompts = tomli.load(f)
        except FileNotFoundError as e:
            logging.error(e)
            raise OSError("Prompt templates file not found.")
        
        logging.info("load_prompt_templates (OK)")
        return prompts
    
    
    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def __cohere_client(self, cohere_api_key):
        """
        Initialize Cohere client

        Parameters:
        - cohere_api_key (str): Cohere API key

        Returns:
        - cohere.Client: Cohere client
        """
        return cohere.Client(cohere_api_key)