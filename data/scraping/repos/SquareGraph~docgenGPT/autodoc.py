"""Main class, AutoDoc and handy functions"""

from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Union, Type

from tqdm.autonotebook import tqdm
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PythonLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

import black

__all__ = ['SystemMessage', 'PyFile', 'AutoDoc']


class SystemMessage(Enum):
    """
    An enumeration representing system messages.
    """
    open: str = "I would like You to act as a senior programmer, with vast python knowledge and capabilities to understand complex code."
    docs: str = "could you create docstrings and type hints for all of classes and methods in provided code?"
    markdown: str = "could you create a markdown file with description of this code and example usage of classes and methods?"


@dataclass
class PyFile:
    """
    Represents a Python file.
    """
    path: str
    context: str
    page_content: str
    answer: Union[str, None] = None

    @classmethod
    def from_document(cls: Type['PyFile'], document: Document, context: str) -> 'PyFile':
        """
        Create a PyFile instance from a Document object.

        Args:
            document (Document): The document object.
            context (str): The context for the file.

        Returns:
            PyFile: A new PyFile instance.
        """
        path = document.metadata['source']
        content = document.page_content
        return cls(path=path, context=context, page_content=content, answer=None)

    def format_answer(self) -> str:
        """
        Formats the answer and returns it.

        Returns:
            str: The formatted answer.
        """
        try:
            codelines = [line for line in self.answer.strip().split('\n')[1:-1] if not '```' in line]
            return black.format_str('\n'.join(codelines), mode=black.FileMode())
        except (KeyError, AttributeError, ValueError) as e:
            return self.page_content

    def __post_init__(self):
        self.name = Path(self.path).name.split('.')[0]


class AutoDoc:
    """
    A class to automate the generation of docstrings for Python files.
    """

    def __init__(self, path_dir: Path, openai_api_key: str, context: Dict[str, str], model_name: str = 'gpt-3.5-turbo',
                 temperature: int = 0, **kwargs) -> None:
        """
        Initialize the AutoDoc class.

        Args:
            path_dir (Path): Directory path containing the Python files.
            openai_api_key (str): OpenAI API key.
            context (Dict[str, str]): Context for each file.
            model_name (str, optional): Name of the OpenAI model. Defaults to 'gpt-3.5-turbo'.
            temperature (int, optional): Temperature setting for the OpenAI model. Defaults to 0.
        """
        self.raw_paths = self.__get_python_paths(path_dir)
        self.documents = [PythonLoader(filepath).load()[0] for filepath in self.raw_paths]
        self.pyfiles = self.__get_pyfiles(context)
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature, openai_api_key=openai_api_key)
        self.vectorstore = Chroma.from_documents(documents=self.documents,
                                                 embedding=OpenAIEmbeddings(openai_api_key=openai_api_key))

    def __get_python_paths(self, path_dir: Path) -> List[str]:
        """
        Retrieves paths to Python files from a given directory.

        Args:
            path_dir (Path): Directory path.

        Returns:
            List[str]: List of paths to Python files.
        """
        return [str(file) for file in path_dir.iterdir() if file.suffix == '.py']

    def __get_pyfiles(self, context: Dict[str, str]) -> List[PyFile]:
        """
        Generates a list of PyFile instances from documents.

        Args:
            context (Dict[str, str]): Context for each file.

        Returns:
            List[PyFile]: List of PyFile instances.
        """
        pyfiles = []
        for doc in self.documents:
            name = Path(doc.metadata['source']).name.split('.')[0]
            pyfiles.append(PyFile.from_document(document=doc, context=context[name]))

        return pyfiles

    def retrive_docs(self, pyfile: PyFile) -> str:
        """
        Retrieves docs for a given PyFile instance.

        Args:
            pyfile (PyFile): PyFile instance.

        Returns:
            str: Retrieved docs.
        """
        ret = self.vectorstore.as_retriever(search_kwargs={"filter": {"source": pyfile.path}})
        chain = RetrievalQA.from_chain_type(self.llm, retriever=ret)
        prompt = f"{SystemMessage.open.value} According to the following description: {pyfile.context}, {SystemMessage.docs.value}."
        pyfile.answer = chain({"query": prompt})['result']
        return pyfile

    def __get_docstrings(self) -> Dict[str, str]:
        """
        Gathers docstrings for the Python files.

        Returns:
            Dict[str, str]: Dictionary with filenames as keys and docstrings as values.
        """
        updated = [self.retrive_docs(pyfile) for pyfile in tqdm(self.pyfiles,
                                                                total=len(self.pyfiles),
                                                                desc='Generating docstrings...')]
        return {item.name: item.format_answer() for item in updated}

    def generate_docstrings(self) -> None:
        """
        Generates docstrings and writes them to Python files in a new directory.
        """
        try:
            dirname = Path(self.raw_paths[0]).parent.parent / (str(Path(self.raw_paths[0]).parent.name) + "_with_docs")
            dirname.mkdir(parents=True, exist_ok=True)

            docstrings = self.__get_docstrings()

            for filename, content in docstrings.items():
                with open(f"{dirname / filename}.py", "w") as f:
                    f.write(content)

            print("Finished!")

        except (AttributeError, KeyError, ValueError) as e:
            print(e)
            pass