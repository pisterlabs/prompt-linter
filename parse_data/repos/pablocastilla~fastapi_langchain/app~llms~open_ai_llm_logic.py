# the following module is the logic of the chatbot.
# it uses the langchain library to interact with the LLM.
# it is a class that implements the ILLM interface.
# it defines a template for the system message prompt.It gives the behaviour of the chatbot.
# The chatbot must try to get the name, the identification number and the symptons of the patient.
# After that it generates a json and sends it to the queue management system.

from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts.chat import SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser
from typing import Any
from app.llms import Illm
from app.llms.LLMDTOs import EmergencyRoomTriage
from app.llms.token_cost_process import CostCalcAsyncHandler, TokenCostProcess
from langchain.memory import ConversationBufferMemory
import json
from app.queue_management.iqueue_management_system import IQueueManagementSystem


parser = PydanticOutputParser(pydantic_object=EmergencyRoomTriage)

# system message prompt. It defines the instructions for the chatbot for behaving like someone in the emergency room.
# doing the triage.
template: str = """
            You are a doctor in an emergency room doing the triage.Be very concise.

            Interfact with the user patient until you have the name.

            After that, interact with the patient until you have its identification number.

            After that, interact with the patient until you have one description of the symptons.

            Do not ask for more information than the symptons.

            Do not use the internet as a reference.

            Do not create more conversations than the one you are having.

            Do not use surgery specialties.

            When you have the name, the identification number and the symptons decide the specialty and the function and answer following this format:
            {format_instructions}

            """

system_message_prompt = SystemMessagePromptTemplate.from_template(template)

human_template: str = """

            Previous conversation:
            {chat_history}

            Patient: {text}
            """
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

memory = ConversationBufferMemory(input_key="text", memory_key="chat_history", human_prefix="Patient", ai_prefix="Doctor")


class OpenAILLMLogic(Illm.ILLM):
    """
    Implementation of the client that is responsible for the logic of the LLM.
    """

    def __init__(self, queue_management_system: IQueueManagementSystem) -> None:
        self.queue_management_system: IQueueManagementSystem = queue_management_system

    def __enter__(self) -> "OpenAILLMLogic":
        self.token_cost_process = TokenCostProcess()
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        print(self.token_cost_process.get_cost_summary("gpt-3.5-turbo"))

    async def chat(self, prompt: str) -> str:
        """chat with the LLM.
        - get the prompt from the user
        - send the prompt to the LLM using the history
        - get the response from the LLM
        - if the response is a json then send it to the queue management system
        - return the response to the user
        """

        llm = ChatOpenAI(temperature=0, callbacks=[CostCalcAsyncHandler("gpt-3.5-turbo", self.token_cost_process)])

        chain = LLMChain(llm=llm, prompt=chat_prompt, memory=memory, verbose=True)

        result: str = await chain.arun({"text": prompt, "format_instructions": parser.get_format_instructions()})

        print(result)

        if self.is_json_convertible(result):
            json_result = json.loads(result)
            result = json_result["response_to_the_user"]
            self.queue_management_system.enqueue(json_result["name"], json_result["urgency"])
            return f"{result} ({json_result['urgency']},{json_result['medical_specialty']})"
        else:
            return result

    def is_json_convertible(self, text: str) -> bool:
        """Check if a string is json convertible."""
        try:
            json.loads(text)
            return True
        except json.JSONDecodeError:
            return False
