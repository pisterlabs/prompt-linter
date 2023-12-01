from pathlib import Path
from typing import Optional, Type

from decouple import config
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from aiudittool.utils import (
    extract_first_code_block,
    preprocess_solidity_code,
    save_to_text_file,
)

SYSTEM_MESSAGE = "You are a world class smart contract developer that creates EVM-compatible Solidity code given a description of a desired Smart Contract."
HUMAN_MESSAGE_TEMPLATE = (
    "Please write the code for a smart contract in Solidity 0.8.19 that conforms to the following description. "
    "Use Open Zeppelin libraries if appropriate. Comment the contract using natspec. Do not NOT use any constructor arguments. "
    "Description:\n"
    "{description}\n\n"
    "A voting app with an owner that can create a voting session. Use Openzeppelin for ownership and have signatures for voting"
    "Output a single code block within backticks containing the Solidity code."
)


smart_contract_writer_chain = LLMChain(
    llm=ChatOpenAI(openai_api_key=config("OPENAI_API_KEY"), temperature=0.0),  # type: ignore
    prompt=ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(SYSTEM_MESSAGE),
            HumanMessagePromptTemplate.from_template(HUMAN_MESSAGE_TEMPLATE),
        ]
    ),
)


class SmartContractWriterToolInput(BaseModel):
    title: str = Field(description="A title for the smart contract, e.g. 'ERC20 Token'")
    description: str = Field(
        description="""
        use this tool when you need to create a smart contract using the description.
        A detailed description of the smart contract, including its purpose, its functions, and its variables
        """
    )


class SmartContractWriterTool(BaseTool):
    name = "SmartContractWriter"
    description = "Useful for writing EVM-compatible smart contracts given a title and description. Returns the path to the .sol file."
    args_schema: Type[BaseModel] = SmartContractWriterToolInput

    def _run(
        self,
        title: str,
        description: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        chain_result = smart_contract_writer_chain.run(f"{title}\n{description}")
        code_block = extract_first_code_block(chain_result)
        code_block = preprocess_solidity_code(code_block)

        output_path = Path("smart_contracts")
        output_path.mkdir(exist_ok=True, parents=True)
        snake_case_title = title.lower().replace(" ", "_")
        file_name = f"{snake_case_title}.sol"

        save_to_text_file(code_block, output_path, file_name)
        absolute_path_str = str((output_path / file_name).absolute())
        return absolute_path_str

    async def _arun(
        self,
        description: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        raise NotImplementedError("This tool does not support async mode.")
