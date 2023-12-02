from asyncio import gather
from pathlib import Path

from langchain import LLMChain

from execution_pipeline.types import Executor


class PythonLangchainExecutor(Executor):

    def __init__(self, name: str, langchain: LLMChain):
        self._name = name
        self.llm_chain = langchain

    @property
    def name(self) -> str:
        return "llm_executor_python_langchain_" + self._name

    async def execute(self, code_path: Path, input_paths: list[Path]) -> list[str]:
        python_code = code_path.read_text()
        execution_coroutines = []
        for input_path in input_paths:
            input_text = input_path.read_text()
            execution_coroutine = self.llm_chain.arun(python_code=python_code, input_text=input_text)
            execution_coroutines.append(execution_coroutine)
        return list(await gather(*execution_coroutines))
