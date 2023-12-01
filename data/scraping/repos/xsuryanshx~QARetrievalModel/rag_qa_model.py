from typing import Tuple, List
import openai
import pandas as pd
import json
import time
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.llms import Cohere
from langchain.vectorstores import Qdrant
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from tqdm import tqdm
from rouge_score import rouge_scorer
import re


class RAG_QA_Model:
    """QA Model Class"""

    def __init__(self) -> None:
        """initializer"""
        _ = load_dotenv(find_dotenv())
        self.__db = None
        self.__documents = []

    def load_document(
        self,
        selected_document: str,
        model_type: str,
    ):
        """load documents

        Args:
            selected_document (str): name of selected document
            model_type (str): model type that defines the vector storage type
            openai_key (str, optional): Defaults to os.getenv("OPENAI_API_KEY")
        """
        data_path = Path("./files").resolve()
        with open(Path("./document_config.json").resolve(), "r") as f:
            document_config = json.load(f)
        if selected_document in document_config:
            file_prefix = document_config[selected_document]["document"]
            path_to_file = Path(data_path, file_prefix).as_posix()

        # Load the text document
        loader = TextLoader(path_to_file, encoding="utf8")
        self.__documents = loader.load()

        # split the documents into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n", chunk_size=1000, chunk_overlap=100
        )
        self.__documents = text_splitter.split_documents(self.__documents)

        # Use the Open AI Embeddings

        if model_type == "Normal":
            embeddings = OpenAIEmbeddings()
            self.__db = Chroma.from_documents(self.__documents, embeddings)
        elif model_type == "Multilingual":
            embeddings = CohereEmbeddings(
                model="multilingual-22-12",
                cohere_api_key=os.getenv("COHERE_API_KEY"),
            )
            self.__db = Qdrant.from_documents(
                self.__documents,
                embeddings,
                location=":memory:",
                collection_name="my_documents",
                distance_func="Dot",
            )
        else:
            raise ValueError("Invalid vector storage type.")

    def is_valid_api_key(self, api_key: str) -> bool:
        """
        Determine whether the input api key is valid.

        Parameters
        ----------
        api_key: str
            An API key

        Returns
        -------
        api_key_is_valid: bool
            Whether the API key is valid or not
        """
        try:
            test = OpenAI(openai_api_key=api_key, max_tokens=2)
            test("test")
        except openai.error.AuthenticationError:
            return False
        else:
            return True

    def set_api_key(self, api_key: str) -> None:
        """
        Set the api key.

        Parameters
        ----------
        api_key: str
            An API key
        """
        openai.api_key = api_key

    def answer_questions(
        self,
        question: str,
        number_of_documents_to_review: int,
        temperature: float,
    ) -> Tuple[pd.DataFrame, float]:
        """model used to answer question based on input question and parameters

        Args:
            question (str): question in string
            number_of_documents_to_review (int): number of most chunks of text used to answer
            temperature (float): temperature

        Returns:
            Tuple[pd.DataFrame, float]: dataframe of answer
        """
        retriever = self.__db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": number_of_documents_to_review},
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=OpenAI(temperature=temperature),
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": self.prompt_template},
        )
        start_time = time.time()

        similar_documents = retriever.get_relevant_documents(question)

        with get_openai_callback() as cb:
            result = qa_chain({"input_documents": similar_documents, "query": question})

        end_time = time.time()
        total_request_time = round(end_time - start_time)

        scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
        scores = scorer.score(result["result"], similar_documents[0].page_content)

        resulting_df = pd.DataFrame(
            {
                "Question": [question],
                "Answer": [result["result"]],
                "Score": [scores["rouge1"].fmeasure],
                "Request Time (s)": [total_request_time],
                "Total Cost ($)": [cb.total_cost],
                "Total Tokens": [cb.total_tokens],
            }
        )

        return resulting_df

    @property
    def total_chunks(self):
        """number of chunks of text made after splitting

        Returns:
            int: number of chunks
        """
        return len(self.__documents)

    @property
    def prompt_template(self):
        """Prompt for generating answer."""
        template_format = """
        You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
        Give answer intelligently like a professional, in bullet points.
        If you don't know the answer, just say that you don't know. Don't try to make up an answer.

        Examples of some expected answers - 

        Examples #1
        Summary: Apples are red in colour.

        Question: who is the prime minister of india?
        Answer: Sorry, I don't know how to answer this question.

        Examples #2
        Summary: The witness took the stand as directed. It was night and the witness forgot his glasses. \
        he was not sure if it was a sports car or an suv. The rest of the report shows everything was okay.

        Question: what type was the car?
        Answer: He was not sure if it was a sports car or an suv.

        Examples #3
        Summary: Pears are either red or orange

        Question: what color are apples?
        Answer: Sorry, I don't know how to answer this question.

        Now your turn, Begin!

        Summary: {context}
        Question :{question}
        Answer:
        """
        prompt = PromptTemplate(
            template=template_format,
            input_variables=["context", "question"],
        )
        return prompt
