import argparse
import json
import os
from dataclasses import dataclass
from getpass import getpass
from typing import List, Any, Optional

from langchain import FAISS, OpenAI, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.schema import Document, OutputParserException
from langchain.text_splitter import RecursiveCharacterTextSplitter


@dataclass
class Config:
    file: str = None
    chunk_size: int = None
    chunk_overlap: int = None
    query: str = None

    @staticmethod
    def check_envvars() -> None:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            os.environ["OPENAI_API_KEY"] = getpass("Please enter the OpenAI API key: ")

    def parse_from_command_line(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--chunk-size', dest='chunk_size', type=int, default=2000,
                            help='Chunk size (default: 2000)')
        parser.add_argument('--chunk-overlap', dest='chunk_overlap', type=int, default=10,
                            help='Chunk overlap (default: 10)')
        parser.add_argument('-f', '--file', dest='file', required=True,
                            help='Path to the markdown file')
        parser.add_argument('-q', '--query', dest='query', type=str, default='',
                            help='Q&A based on the markdown file')
        args = parser.parse_args()

        self.file = args.file
        self.query = args.query
        self.chunk_size = args.chunk_size
        self.chunk_overlap = args.chunk_overlap

        self.check_envvars()


class MarkdownPost:

    def __init__(self, config: Config):
        self.config = config
        self.path: Optional[str] = config.file
        self.filename: Optional[str] = os.path.splitext(os.path.basename(self.path))[0]
        self.document: List[Document] = UnstructuredMarkdownLoader(self.path).load()
        self.splits: List[Document] = self.split()
        self.faiss_file: str = os.path.join(os.path.dirname(self.path), f"{self.filename}.faiss")
        self.db: FAISS | None = None

    def split(self) -> List[Document]:
        """
        Split the unstructed markdown document into sumaller chunks
        :return: List of Document
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        return text_splitter.split_documents(self.document)

    def build_embedding(self, force_rebuild: bool = False) -> None:
        """
        Build embeddings using FAISS index
        :param force_rebuild: Rebuild the embeddings even if a previous index exists
        """
        embeddings = OpenAIEmbeddings()
        if force_rebuild or not os.path.exists(self.faiss_file):
            db = FAISS.from_documents(self.splits, embeddings)
            db.save_local(self.faiss_file)
            self.db = db
        else:
            self.db = FAISS.load_local(self.faiss_file, embeddings)

    def search(self, query: str, **kargs: Any) -> List[Document]:
        """
        Search by embedding similarity
        :param query: query term
        :param kargs: search args
        :return: matched Documents
        """
        return self.db.similarity_search(query, **kargs)

    def question_answer(self, query: str, **kargs: Any) -> str:
        """
        Question answering over the post index
        :param query: query term
        :return: answer
        """
        retriever = self.db.as_retriever(**kargs)
        qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)
        return qa.run(query)

    def summarize(self) -> Any:
        """
        Summarize the markdown post content
        :return: JSON output based on the prebuilt prompt template
        """
        map_prompt = PromptTemplate(
            template="write a concise summary of the following:\n\n\"{text}\"\n\n"
                     "CONCISE SUMMARY WITH THE AUTHOR'S TONE IN THE ORIGINAL LANGUAGE:",
            input_variables=["text"],
        )
        # Create an instance of the output parser and configure the response schema
        response_schemas = [
            ResponseSchema(name="summary",
                           description="PROVIDE A CONCISE SUMMARY IN THE ORIGINAL LANGUAGE "
                                       "WITH NO MORE THAN 3 SENTENCES AND USE THE AUTHOR'S TONE"),
            ResponseSchema(name="keywords",
                           description="NO MORE THAN 5 KEYWORDS RELATED TO THE TEXT"),
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

        # Define the PromptTemplate and set the output parser
        reduce_prompt = PromptTemplate(
            template="Write a concise summary of the following:\n\n\"{text}\"\n\n{instructions}",
            input_variables=["text"],
            partial_variables={"instructions": output_parser.get_format_instructions()},
            output_parser=output_parser,
        )

        chain = load_summarize_chain(
            llm=OpenAI(temperature=0),
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=reduce_prompt
        )
        output = chain.run({"input_documents": self.splits})

        try:
            return output_parser.parse(output)
        except OutputParserException:
            # In case LLM gives invalid JSON output
            # Most failed cases are missing a comma between the "summary" and "keywords" fields
            try:
                # Make a workaround and try once again
                return output_parser.parse(output.replace('"\n\t"keywords"', '",\n\t"keywords"'))
            except OutputParserException as ex:
                # If all attempts to parse the output fail, just return the raw content
                print(f'[ERROR] {ex}')
                return output


if __name__ == '__main__':
    c = Config()
    c.parse_from_command_line()
    post = MarkdownPost(c)
    post.build_embedding()
    if c.query is None or c.query == "":
        print(json.dumps(post.summarize()))
    else:
        print(post.question_answer(c.query).strip())
