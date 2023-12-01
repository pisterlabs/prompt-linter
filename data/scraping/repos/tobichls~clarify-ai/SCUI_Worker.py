#############################################################################
# Code description
"""Script to generate the UI worker."""

#############################################################################
#############################################################################
# Modules & libraries
import os
import json
import requests
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from PyQt5.QtCore import QObject, pyqtSignal

#############################################################################
#############################################################################
# Import the different scripts
from SCUI_Variables import OpenAIkey

#############################################################################
#############################################################################
class SCUI_Worker(QObject):
    # Signals
    finished = pyqtSignal()
    smart_contract = pyqtSignal(str)
    output = pyqtSignal(str)
    display = pyqtSignal(str)

#############################################################################
# General functions
    def setContract(self, _contractID):
        self.contractID = _contractID

    def getSmartContract(self, contractID):
        """Get the smart contract from Hiro."""
        # https://docs.hiro.so/api/get-contract-info
        url = "https://api.mainnet.hiro.so/extended/v1/contract/"
        payload = {}
        headers = {'Accept': 'application/json'}

        response = requests.request("GET", url + contractID, headers=headers, data=payload)
        response_dict = json.loads(response.text) # convert to dictionary

        return response_dict['source_code']

    def askOpenAI(self):
        """Ask a question to OpenAI."""
        # https://python.langchain.com/docs/integrations/llms/openai
        try:
            os.environ["OPENAI_API_KEY"] = OpenAIkey.OPENAI_API_KEY

            template = """Please give me a non-technical copy of the following Clarity
                smart contract, without bullet points and using the documentation available
                on https://book.clarity-lang.org/:\n{smart_contract}"""

            prompt = PromptTemplate(template=template, input_variables=["smart_contract"])

            llm = OpenAI()
            llm_chain = LLMChain(prompt=prompt, llm=llm)
            smart_contract = self.getSmartContract(self.contractID)
            response = llm_chain.run(smart_contract)

            self.display.emit('Task completed.')
            self.output.emit(response.strip())
            self.smart_contract.emit(smart_contract.strip())
        except:
            self.display.emit('Something went wrong in the worker!')

        # finish thread
        self.finished.emit()