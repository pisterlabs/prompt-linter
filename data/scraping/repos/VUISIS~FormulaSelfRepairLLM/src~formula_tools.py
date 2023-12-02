from typing import Any, Optional

from os.path import abspath

from langchain.callbacks.manager import (
    CallbackManagerForToolRun,
)
from langchain.chat_models.base import BaseChatModel
from langchain.chains import SequentialChain
from langchain.tools.base import BaseTool
from langchain.utilities import PythonREPL
from .prompts import (
    DEBUG_FORMULA_CODE_LLM_DESC,
    DEBUG_FORMULA_CODE_LLM_PROMPT,
    DEBUG_FORMULA_CODE_LLM_RETURN,
    DECODE_FORMULA_CODE_EXPLAIN_PROMPT,
    DECODE_FORMULA_CODE_LLM_DESC,
    DECODE_FORMULA_CODE_LLM_RETURN,
    DECODE_FORMULA_CODE_PAINPOINTS_PROMPT,
    LOAD_FORMULA_CODE,
    QUERY_FORUMULA_CODE,
    DECODE_FORMULA_CODE_LLM_PROMPT,
)
import os
from pythonnet import load
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain, PromptTemplate
from langchain.tools import BaseTool, StructuredTool, Tool, tool
import json

env4ml = os.environ.get("FORMULA_KERNEL", "1")
if env4ml == "1":
    load("coreclr", runtime_config=os.path.abspath("../runtimeconfig.json"))
    import clr

    process_path = abspath("../CommandLine/CommandLine.dll")

    clr.AddReference(os.path.abspath(process_path))

from Microsoft.Formula.CommandLine import CommandInterface, CommandLineProgram
from System.IO import StringWriter
from System import Console

sink = CommandLineProgram.ConsoleSink()
chooser = CommandLineProgram.ConsoleChooser()
ci = CommandInterface(sink, chooser)

sw = StringWriter()
Console.SetOut(sw)
Console.SetError(sw)

if not ci.DoCommand("wait on"):
    raise Exception("Wait on command failed.")

if not ci.DoCommand("unload *"):
    raise Exception("Unload command failed.")


def _LoadFormulaCode(query: str):
    # Put the query string into a file ./temp.4ml
    with open("./temp.4ml", "w") as f:
        f.write(query)

    # Load the file into the FORMULA program
    if not ci.DoCommand("unload"):
        raise Exception("Unload command failed.")

    if not ci.DoCommand("load ./temp.4ml"):
        raise Exception("Load command failed.")

    output = sw.ToString()
    sw.GetStringBuilder().Clear()

    # if output contains the word "failed" case insensitive return false
    if "failed" in output.lower():
        return f"Failed to load FORMULA code, you probably have an syntax error in your code. \
            Please check your code and try again.\n\nHere is the output from the FORMULA program:\n{output}"

    return "Successfully loaded FORMULA code, make sure that you now query the code using \
        the QueryFormulaCode tool to assert that the code is working as expected now."


LoadFormulaCode = Tool.from_function(
    func=_LoadFormulaCode, name="LoadFormulaCode", description=LOAD_FORMULA_CODE
)


def _QueryFormulaCode(query: str):
    sw.GetStringBuilder().Clear()
    if not ci.DoCommand(query):
        raise Exception("Query command failed.")

    output = sw.ToString()
    sw.GetStringBuilder().Clear()

    if "not solvable" in output.lower():
        return f""" \
Your code is not solvable. This means that the code is broken and you need to fix it.
Here was the output from the FORMULA program:

{output}

Make sure to try to regenerate your code, using the DebugFormulaCodeLLM tool if needed, and then re run the program again
using the LoadFormulaCode tool / QueryFormulaCode Formula REPL tools.
"""

    return output


QueryFormulaCode = Tool.from_function(
    func=_QueryFormulaCode, name="QueryFormulaCode", description=QUERY_FORUMULA_CODE
)


class DecodeFormulaCodeLLM(BaseTool):
    name = "DecodeFormulaCodeLLM"
    description = DECODE_FORMULA_CODE_LLM_DESC
    llm: BaseChatModel

    def _run(
        self,
        query: str = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs,
    ) -> Any:
        parsed_code = kwargs
        code = parsed_code["code"]
        interpreter_output = parsed_code["interpreter_output"]

        code_description_template = DECODE_FORMULA_CODE_EXPLAIN_PROMPT

        code_painpoints_template = DECODE_FORMULA_CODE_PAINPOINTS_PROMPT

        prompt_template = PromptTemplate(
            input_variables=["code", "interpreter_output"],
            template=code_description_template,
        )
        code_understander_chain = LLMChain(
            llm=self.llm, prompt=prompt_template, output_key="explanation"
        )

        prompt_template = PromptTemplate(
            input_variables=["code", "interpreter_output", "explanation"],
            template=code_painpoints_template,
        )
        painpoints_chain = LLMChain(
            llm=self.llm, prompt=prompt_template, output_key="pain_points"
        )

        overall_chain = SequentialChain(
            chains=[code_understander_chain, painpoints_chain],
            input_variables=["code", "interpreter_output"],
            # Here we return multiple variables
            output_variables=["explanation", "pain_points"],
            verbose=True,
        )

        output = overall_chain(parsed_code)

        return_output = DECODE_FORMULA_CODE_LLM_RETURN.format(output=output['explanation'])

        return return_output

    async def _arun(
        self,
    ):
        raise NotImplementedError("custom_search does not support async")


class DebugFormulaCodeLLM(BaseTool):
    name = "DebugFormulaCodeLLM"
    description = DEBUG_FORMULA_CODE_LLM_DESC
    llm: BaseChatModel

    def _run(self, **kwargs) -> Any:
        parsed_code = kwargs
        prompt = DEBUG_FORMULA_CODE_LLM_PROMPT.format(**parsed_code)

        output = self.llm.predict(prompt)

        return_output = DEBUG_FORMULA_CODE_LLM_RETURN.format(output=output)

        return return_output

    async def _arun(
        self,
    ):
        raise NotImplementedError("custom_search does not support async")
