from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from langchain.output_parsers import StructuredOutputParser, ResponseSchema

from langchain.chat_models import ChatOpenAI
from app.utils import chain_verbose
from app.utils import logger

template="""You are an Assistant that helps users choose an accommodation based on its preferences. 
Your task is only to help the user choose an accommodation option for booking and answer any question about it, \
any other tasks must not be handled by you. The user has already selected the business, now you help him choose an accommodation \ 
option within the business i.e: A room within the business.


Follow these steps before responding to the user:

Step 1: If you have not shown the user a summary of the available options, show the summary including a brief description, amenities and the price per night for each option \
and ask the user if he wants to book any of them.
Here is the list of available options: \
{properties_info}

Step 2: If the user makes any question about the options after showing the summary, answer it based on the available options information. 

Step 3: If the user asks if there are other options available, you respond that there are no more available for those dates.

Step 4: If the user does not want any of the available options, \
you apologize and tell the user that you will notify him if you have something new available in the future.

Here is the conversation: 
{chat_history}

You respond in a short, very conversational friendly style.
response to th user: """


class HouseSelectionAssistantTool:

    def __init__(self):

        llm = ChatOpenAI(temperature=0.)
        

        prompt_template = PromptTemplate(
            input_variables=["chat_history", "properties_info"], 
            template=template
        )

        self.chain = LLMChain(llm=llm, 
                              prompt=prompt_template, 
                              verbose=chain_verbose,
                              output_key="result")

    def run(self, chat_history, properties_info):

        info = self.chain({"properties_info": properties_info, "chat_history": chat_history})
        # logger.debug(f"HouseSelectionAssistantTool result {info}")
        return info["result"]
    