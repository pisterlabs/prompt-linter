from typing import Dict, List

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.llms.base import BaseLLM
from langchain_contrib.chains import ChoiceChain
from langchain_contrib.prompts import ChainedPromptTemplate, Templatable
from pydantic import BaseModel

from zamm.chains.general import LaxSequentialChain

from .filesystem import FileSystemTool
from .prompt import NEW_CONTENTS_PROMPT, REPLACE_CONTENTS_PROMPT, WHICH_FILE_PROMPT


class AskFileChain(LLMChain, BaseModel):
    """Asks LLM for path of file to edit."""

    @property
    def input_keys(self) -> List[str]:
        """Input keys to this chain."""
        return self.prompt.input_variables

    @property
    def output_keys(self) -> List[str]:
        return ["file_path"]

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        assert isinstance(self.llm, BaseLLM)
        file_path = (
            self.llm(self.prompt.format(**inputs), stop=["`", "\n"]).strip().strip("`")
        )
        return {"file_path": file_path}


class FileOutputChain(LLMChain, BaseModel):
    """Asks LLM for content of new file."""

    @property
    def input_keys(self) -> List[str]:
        """Input keys to this chain."""
        return self.prompt.input_variables + ["file_path"]

    @property
    def output_keys(self) -> List[str]:
        return self.prompt.input_variables + ["new_contents"]

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        assert isinstance(self.llm, BaseLLM)
        new_contents = self.llm(self.prompt.format(**inputs), stop=["\n```"])
        FileSystemTool().write_file(inputs["file_path"], new_contents)
        return {**inputs, "new_contents": new_contents}


class ReadFileChain(Chain):
    @property
    def input_keys(self) -> List[str]:
        """Input keys this chain expects."""
        return ["file_path"]

    @property
    def output_keys(self) -> List[str]:
        """Output keys this chain expects."""
        return ["file_exists", "old_contents"]

    def _validate_outputs(self, outputs: Dict[str, str]) -> None:
        assert "file_exists" in outputs, "Outputs doesn't say if file exists"

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        fs = FileSystemTool()
        return fs.read_file(file_path=inputs["file_path"]).as_llm_output(
            contents_key="old_contents"
        )


class EditFileChain:
    @classmethod
    def default(cls, llm: BaseLLM, prefix: Templatable) -> LaxSequentialChain:
        ask_file = AskFileChain(
            llm=llm, prompt=ChainedPromptTemplate([prefix, WHICH_FILE_PROMPT])
        )
        file_edit_chain = cls.for_file(llm, prefix)
        return LaxSequentialChain(
            chains=[ask_file, file_edit_chain],
            input_variables=ask_file.input_keys,
            output_variables=file_edit_chain.output_keys,
            return_all=True,
        )

    @classmethod
    def for_file(cls, llm: BaseLLM, prefix: Templatable) -> ChoiceChain:
        read_file = ReadFileChain()
        edit_file = FileOutputChain(
            llm=llm, prompt=ChainedPromptTemplate([prefix, REPLACE_CONTENTS_PROMPT])
        )
        new_file = FileOutputChain(
            llm=llm, prompt=ChainedPromptTemplate([prefix, NEW_CONTENTS_PROMPT])
        )
        return ChoiceChain(
            choice_picker=read_file,
            choice_key="file_exists",
            choices={
                "True": edit_file,
                "False": new_file,
            },
        )
