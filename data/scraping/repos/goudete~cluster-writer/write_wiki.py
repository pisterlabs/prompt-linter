
# Dependencies
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import (Language, RecursiveCharacterTextSplitter)
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser

# Custom Dependencies
from writers.wiki_writer import WikiWriter
from transformers.transformer import Transformer
# from retrievers.retriever import Retriever

# Config
# from config.wiki_writer_config import AppConfig


def main():
    root_dir = './cluster'

    writer = WikiWriter(
        loader=GenericLoader.from_filesystem(
            path=root_dir,
            glob="**/*",
            suffixes=[".py"],
            parser=LanguageParser(language=Language.PYTHON, parser_threshold=500)
        ),
        transformer=Transformer(
            splitter=RecursiveCharacterTextSplitter
                .from_language(
                    language=Language.PYTHON,
                    chunk_size=2000, 
                    chunk_overlap=200
                )
        ),
        # retriever=Retriever(
        #     vectorstore=DeepLakeProvider.instance(
        #         dataset_path=dataset_path,
        #         embeddings=OpenAIEmbeddings(
        #             openai_api_key=AppConfig.OPENAI_API_KEY,
        #         ),
        #         read_only=True
        #     ),
        #     model=ChatOpenAI(
        #         model='gpt-3.5-turbo',
        #         openai_api_key=AppConfig.OPENAI_API_KEY
        #     ),
        #     chain=ConversationalRetrievalChain()
        # ),
    )

    writer.write()

if __name__ == '__main__':
    main()