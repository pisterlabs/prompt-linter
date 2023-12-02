"""
A file which contains a prompt to get an LLM to analyse a smart contract, looking for vulnerabilities. The tool is just a wrapper for a LLM.
"""
from typing import Optional

from langchain.callbacks.manager import (AsyncCallbackManagerForToolRun,
                                         CallbackManagerForToolRun)
from langchain.llms import OpenAI
from langchain.tools import BaseTool


class SmartContractAnalysis(BaseTool):
    name = "smart_contract_analysis"
    description = "Smart contract expert who will provide information of any vulnerabilities in a smart contract, when provided with smart contract code. Does not accept addresses."

    def _run(self, query: str) -> str:
        """
        Call an LLM, using an engineered prompt, a query and previous chat history as context.
        """
        print(query)
        llm = OpenAI(temperature=0.0)
        prompt = f"""
        Analyse code for a solidity smart contract, highlight any known vulnerabilities.
        If there are no known vulnerabilities please say so.

        A re-entrancy vulnerability could look like:
        mapping (address => uint) private userBalances;

        function withdrawBalance() public {{
        uint amountToWithdraw = userBalances[msg.sender];
        (bool success, ) = msg.sender.call.value(amountToWithdraw)("");
        require(success);
        userBalances[msg.sender] = 0;
        }}


        {query}
        """
        return llm(prompt)
    
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("smart_contract_analysis does not support async")



