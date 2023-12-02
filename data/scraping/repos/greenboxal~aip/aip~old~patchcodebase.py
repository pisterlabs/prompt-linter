import os
import sys
import pathlib
import re
import shutil
import pinecone
import ast
from typing import List

from pinecone_text.sparse import BM25Encoder
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

from langchain.chains import ConversationChain
from langchain.memory import (
    ConversationSummaryBufferMemory,
    ConversationEntityMemory,
    ConversationBufferWindowMemory,
    CombinedMemory,
    VectorStoreRetrieverMemory,
)
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate


class CrazyTextLoader(BaseLoader):
    """Load text files."""

    def __init__(self, file_path: str, encoding=None):
        """Initialize with file path."""
        self.file_path = file_path
        self.encoding = encoding

    def load(self) -> List[Document]:
        """Load from file path."""
        with open(self.file_path, encoding=self.encoding) as f:
            text = f.read()
        metadata = {"source": self.file_path}
        return [Document(page_content="####### File Name: " + self.file_path + "\n\n" + text, metadata=metadata)]


BEGIN_FILE_MARKER = re.compile(r'OUTPUT FILE +([a-zA-Z0-9_./ -]+):')
END_FILE_MARKER = re.compile(r'LLM EOF')
EOM_MARKER = re.compile(r'END OF MESSAGE')

_objective = """
You are a helpful AI that describes code functions in one sentence.
"""

_chat_prompt = """
You are an assistant to a human, powered by a large language model trained by OpenAI.

You are designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, you are able to generate human-like text based on the input you receive, allowing you to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

You are constantly learning and improving, and your capabilities are constantly evolving. You are able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. You have access to some personalized information provided by the human in the Context section below. Additionally, you are able to generate your own text based on the input you receive, allowing you to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, you are a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether the human needs help with a specific question or just wants to have a conversation about a particular topic, you are here to assist.

Context:
{code_memory}

Chat History:
{chat_memory}
Human: {input}
AI Assistant: """

chat_prompt = PromptTemplate(
    input_variables=["input", "code_memory", "chat_memory"],
    template=_chat_prompt,
)

initial_prompt = _objective + """

What is the content of each file? Show me the entire content of each file upon request.
"""

continue_prompt = """Continue."""


class Agent:
    def __init__(self, retriever):
        self.parser = FileSplitter("./output", clean=True, immediate=True)

        self.llm_feeling = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.30)
        self.llm_memory = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.70)
        self.llm_codex = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.70)
        self.llm_reason = ChatOpenAI(model_name="gpt-4-32k", temperature=0.70)

        # self.short_term_memory = ConversationSummaryBufferMemory(llm=self.llm_memory, max_token_limit=200, memory_key="short_term_memory", input_key="input", ai_prefix = "AI Assistant")
        # self.long_term_memory = ConversationSummaryBufferMemory(llm=self.llm_feeling, max_token_limit=500, memory_key="long_term_memory", input_key="input", ai_prefix = "AI Assistant")
        self.chat_memory = ConversationBufferWindowMemory(memory_key="chat_memory", input_key="input",
                                                          ai_prefix="AI Assistant", k=1000)
        # self.entity_memory = ConversationEntityMemory(llm=self.llm_memory, ai_prefix="AI Assistant", chat_history_key="chat_memory", input_key="input")

        self.code_memory = VectorStoreRetrieverMemory(memory_key="code_memory", input_key="input", retriever=retriever,
                                                      return_docs=True)

        self.memory = CombinedMemory(memories=[
            # self.short_term_memory,
            # self.long_term_memory,
            self.chat_memory,
            # self.entity_memory,
            self.code_memory,
        ])

        self.codex_chain = ConversationChain(
            llm=self.llm_codex,
            verbose=True,
            memory=self.memory,
            prompt=chat_prompt,
        )

    def get_function_description(self, code):
        input = "Describe this code:\n" + code
        result = self.codex_chain.predict(input=input)
        return result


def main():
    input_folder = "./input"
    code_extensions = [".py", ".kt", ".js", ".ts", ".java", ".json", ".go"]
    docs = []

    for root, _, files in os.walk(input_folder):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1]

            if file_ext in code_extensions:
                loader = CrazyTextLoader(file_path, encoding='utf-8')

                docs.extend(loader.load_and_split())

    embeddings = OpenAIEmbeddings()
    index_name = "langchain-patchcodebase-3"

    try:
        pinecone.create_index(
            name=index_name,
            dimension=1536,  # dimensionality of dense model
            metric="dotproduct",  # sparse values supported only for dotproduct
            pod_type="s1",
            metadata_config={"indexed": []}  # see explaination above
        )
    except:
        pass

    text_splitter = CharacterTextSplitter(chunk_size=2048, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)
    Pinecone.from_documents(name="code", documents=texts, embedding=embeddings, index_name=index_name)

    retriever = Pinecone.from_existing_index(embedding=embeddings,
                                             index_name=index_name).as_retriever()
    agent = Agent(retriever)

    #    with open(output_file, "w") as output:
    #        for func in functions:
    #            name = func.splitlines()[0]
    #            description = agent.get_function_description(name)
    #            output.write(f"{name}: {description}\n")
    #            output.flush()

    for line in sys.stdin:
        if 'Exit' == line.rstrip():
            break

        result = agent.codex_chain.predict(input=line)
        print(result)
    print("Done")


class FileSplitter():
    def __init__(self, output_path, immediate=False, clean=False):
        self.immediate = immediate
        self.current_file = ""
        self.output_path = pathlib.Path(output_path)
        self.files = {}

        if clean:
            shutil.rmtree(str(self.output_path), ignore_errors=True)

        self.output_path.mkdir(parents=True, exist_ok=True)

    def parse(self, line):
        print(line)

        m = re.search(BEGIN_FILE_MARKER, line)

        if m is not None:
            self.begin_file(m.group(1))
            return

        m = re.search(END_FILE_MARKER, line)

        if m is not None:
            self.end_file()
            return

        self.append_line(line)

    def begin_file(self, name):
        if self.current_file != name:
            self.end_file()

        self.current_file = name

    def end_file(self):
        if self.current_file == "":
            return

        if self.immediate:
            self.emit_file(self.current_file)

        self.current_file = ""

    def append_line(self, line):
        if self.current_file != "":
            if self.current_file not in self.files:
                self.files[self.current_file] = ""

            self.files[self.current_file] += line + "\n"

    def emit_file(self, name):
        if not name in self.files:
            return

        contents = self.files[name]
        path = self.output_path.joinpath(name)

        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open(mode="w") as f:
            f.write(contents)

    def emit(self):
        self.end_file()

        if not self.immediate:
            for file in self.files.keys():
                self.emit_file(file)


class FunctionExtractor(ast.NodeVisitor):
    def __init__(self):
        self.functions = []

    def visit_FunctionDef(self, node):
        function_code = ast.unparse(node)
        self.functions.append(function_code)


def extract_functions(file_path: str, file_ext: str) -> List[str]:
    if file_ext != ".py":
        raise ValueError("Only Python files are supported in this implementation.")

    with open(file_path, "r") as file:
        file_content = file.read()

    tree = ast.parse(file_content)
    extractor = FunctionExtractor()
    extractor.visit(tree)

    return extractor.functions


pinecone.init(api_key=os.environ.get("PINECONE_API_KEY"), environment=os.environ.get("PINECONE_ENV", "us-west1-gcp"))

main()
