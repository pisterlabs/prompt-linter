from time import time
from typing import List, Dict, Union, Callable, Optional
from abc import abstractmethod
from langchain.schema import (
    BasePromptTemplate,
    Document,
    BaseDocumentTransformer,
)
from langchain.schema.language_model import BaseLanguageModel
from langchain.chat_models.base import BaseChatModel
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)


class BasePreprocessor:
    def __init__(
        self,
        prompt: BasePromptTemplate,
        llm: Union[BaseLanguageModel, BaseChatModel] = None,
        splitter: Optional[BaseDocumentTransformer] = None,
    ):
        self.prompt = prompt
        self.llm = llm or ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0,
            verbose=True,
        )
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            output_key="output",  # GPT 답변이 저장될 key; default='text'
        )
        self._splitter = splitter

    @abstractmethod
    def preprocess(
        self, docs: List[Document], fn: Optional[Callable] = None
    ) -> List[Document]:
        """Return a new list of Documents with preprocessed contents. Can apply `fn` to each 'page_content' before preprocessing.\
        refer: https://api.python.langchain.com/en/latest/chains/langchain.chains.llm.LLMChain.html"""
        pass

    @property
    @abstractmethod
    def splitter(self):
        pass

    def _split(self, docs: List[Document]):
        return self.splitter.split_documents(docs)

    def preprocess_and_split(
        self,
        docs: List[Document],
        fn: Optional[Callable] = None,
    ) -> List[Document]:
        self.file = open("./splits_output.txt", "w")

        start_preprocess = time()
        docs = self.preprocess(docs, fn)
        end_preprocess = time()
        print(
            f"☑️ Preprocessing took {(end_preprocess - start_preprocess):.3f} seconds for {len(docs)} document(s)."
        )

        start_split = time()
        docs = self._split(docs)
        end_split = time()
        print(
            f"☑️ Splitting into {len(docs)} newly split document(s) took {(end_split - start_split):.3f} seconds."
        )
        self.file.close()
        print(f"☑️ New splits saved to {self.file.name}.")
        return docs

    def save_output(self, output: Dict):
        from pprint import pprint

        pprint(output, stream=self.file)
