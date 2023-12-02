from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import List
from dotenv import load_dotenv
from langchain.output_parsers import OutputFixingParser
# from util import chatgpt_wrapper
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

load_dotenv()


# class Actor(BaseModel):
#     is_male: bool = Field(description="if the actor is male")
#     name: str = Field(description="name of the actor")
#
#
# actor_query = "Generate the filmography for a random actor."
#
# parser = PydanticOutputParser(pydantic_object=Actor)
#
# misformatted = "Please provide the information"
#
# # parser.parse(misformatted)
#
# new_parser = OutputFixingParser.from_llm(parser=parser, llm=ChatOpenAI())
#
# result = new_parser.parse(misformatted)
# print(result)
def chatgpt_wrapper(sys_prompt, text):
    load_dotenv()
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        # streaming=True,
        # callbacks=[StreamingStdOutCallbackHandler()] # not needed for wrapper
    )
    system_message_prompt = SystemMessagePromptTemplate.from_template(sys_prompt)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    # get a chat completion from the formatted messages
    messages = chat_prompt.format_prompt(text=text).to_messages()
    print(messages)
    result = llm(messages)
    return result.content


sys_template = "I would like you to analyze a conversation history between a patient and a physician in order to " \
               "identify the exact reason why the patient did not follow the prescribed measures. Remember, " \
               "you are doing qualitative research, the common excuses or vague answers shouldn't be considered as " \
               "the real reason. For example, an over-weight person who would not want to go swimming " \
               "in public pools " \
               "is afraid that his body would be judged by others in public pools, but he might not express this " \
               "clearly at first. Here is the context information of the conversation: The patient's name is Daniel " \
               "and he is 34 years old with a BMI of 27. His plan includes intermittent fasting with a eating window " \
               "until 1pm and swimming for 30 minutes/500 meters on Monday, Thursday, and Saturday evenings. The " \
               "measure he was supposed to take was to go swimming on Thursday evening, but the " \
               "{{'completed': false}} " \
               "indicates that he did not do it as intended. You will ONLY provide the output in JSON format, " \
               "with a boolean indicating whether a reason was found and a 'reason' field containing a one-sentence " \
               "summary of the patient's reason without any personal information, like name or age. If no reason can " \
               "be determined or the conversation history is not provided, " \
               "the 'reason_found' value will be False and the 'reason' will be 'null'."
print(sys_template)
sys_template = sys_template
result = chatgpt_wrapper(sys_template, input("type: \n"))
print(result)
