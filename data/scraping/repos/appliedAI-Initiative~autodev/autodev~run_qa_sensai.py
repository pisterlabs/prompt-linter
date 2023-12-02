import os

from langchain.text_splitter import PythonCodeTextSplitter

from autodev.util import logging
from autodev.qa.document_db import PythonDocumentDatabase
from autodev.llm import LLMType
from autodev.qa.qa_use_case import QuestionAnsweringUseCase

log = logging.getLogger(__name__)


class PythonDocumentDatabaseSensai(PythonDocumentDatabase):
    def __init__(self):
        sensai_src_path = os.path.join("..", "sensai", "src")
        if not os.path.exists(sensai_src_path):
            raise Exception(f"sensAI source directory was not found in {os.path.abspath(sensai_src_path)}; "
                f"clone the repository: git clone https://github.com/jambit/sensAI.git")
        super().__init__("sensai", sensai_src_path)


class UseCasePythonSensai(QuestionAnsweringUseCase):
    def __init__(self, llm_type: LLMType):
        queries = [
            "Give me information about class VectorModel",
            "What is the base class for regression models that use torch?",
            "What are my options to cache feature generation?",
            "What class allows me to compare the performance of regression models?",
            "How can I apply greedy clustering?",
        ]
        super().__init__(llm_type=llm_type, doc_db=PythonDocumentDatabaseSensai(),
            splitter=PythonCodeTextSplitter(chunk_size=llm_type.chunk_size()),
            queries=queries)


if __name__ == '__main__':
    logging.configure()

    use_case = UseCasePythonSensai(LLMType.OPENAI_CHAT_GPT4)

    use_case.run_example_queries()
