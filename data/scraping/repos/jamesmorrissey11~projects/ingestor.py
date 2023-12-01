import argparse
import os

import pandas as pd
from langchain import OpenAI
from langchain.chains import LLMChain
from langchain.docstore.document import Document
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import CohereEmbeddings, OpenAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from nltk import sent_tokenize


class Ingestor:
    def __init__(self, args):
        self.args = args
        self.embeddings = OpenAIEmbeddings()
        self.llm = OpenAI(model_name="gpt-3.5-turbo")
        loader = DirectoryLoader(
            self.args.data_dir, glob="**/*.txt", loader_cls=TextLoader
        )
        self.chats = loader.load()

        self.sentences = [sent_tokenize(chat.page_content) for chat in self.chats]
        self.first_five_sentences = [
            Document(page_content=" ".join(sent[0:5])) for sent in self.sentences
        ]

    def generate_vectorstore(self):
        text_splitter = CharacterTextSplitter(
            chunk_size=1000, chunk_overlap=0, separator=" "
        )
        docs = text_splitter.split_documents(self.chats)
        self.vectorstore = Chroma.from_documents(
            docs,
            self.embeddings,
            persist_directory=self.args.save_dir,
            collection_name="chats",
        )
        print(f"persisting chroma vectorstore at {self.args.save_dir}")

    def get_chat_outcome(self):
        template = """
        Is the following issue fully resolved? Answer with "yes", "no", or "unsure".

        Chat: {chat}
        Answer:
        """

        prompt_template = PromptTemplate(input_variables=["chat"], template=template)

        self.chat_outcomes = []
        self.chats_text = []

        for i in range(len(self.chats)):
            output = self.llm(prompt_template.format(chat=self.chats[i].page_content))
            output = str(output).lower().replace(".", "")
            self.chat_outcomes.append(output)
            self.chats_text.append(self.chats[i].page_content)
            if i % 20 == 0:
                print(f"processed {i} chats")

        self.outcome_df = pd.DataFrame(
            {"chats": ingestor.chats_text, "outcome": ingestor.chat_outcomes}
        )

    def summarize(self):
        template = """
        Summarize the customer issue in one sentence.

        Chat: {chat}

        Summary: """

        prompt_template = PromptTemplate(input_variables=["chat"], template=template)
        summarize_chain = LLMChain(prompt=prompt_template, llm=self.llm)
        summaries = []
        for i, abbreviated_chat in enumerate(self.first_five_sentences):
            output = summarize_chain.run([abbreviated_chat])
            summaries.append(output)
            if i % 20 == 0:
                print(f"processed {i} chats")

        chat_strings = [chat.page_content for chat in self.chats]
        abbreviated_chat_strings = [
            chat.page_content for chat in self.first_five_sentences
        ]
        summary_df = pd.DataFrame(
            {
                "chats": chat_strings,
                "summaries": summaries,
                "abbreviated_chats": abbreviated_chat_strings,
            }
        )
        summary_df.to_csv("./hackday/datasets/csv/summaries.csv", index=False)

    def write_outcomes_to_csv(self):
        self.outcome_df.to_csv(
            self.args.save_dir + "/csv/resolved_chats.csv", index=False
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--data_dir", type=str, help="Location of source data")
    parser.add_argument(
        "--embedding_model",
        type=str,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    ingestor = Ingestor(args)
    ingestor.get_chat_outcome()
    ingestor.write_outcomes_to_csv()
    ingestor.summarize()
    ingestor.get_chat_outcome()
    ingestor.write_outcomes_to_csv()
    ingestor.generate_vectorstore()
    print("done")
