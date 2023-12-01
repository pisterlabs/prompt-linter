from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI

from langchain.llms import CerebriumAI


def get_openai_waiting_time_generator():
    prompt_template = "Make a variation of the following sentence: \
    {sentence} \
    The sentence should keep the same meaning but should be different. Keep it simple. \
    "
    prompt = PromptTemplate(input_variables=["sentence"], template=prompt_template)
    chatopenai = ChatOpenAI(
                model_name="gpt-3.5-turbo", temperature=1.2)
    llmchain_chat = LLMChain(llm=chatopenai, prompt=prompt)
    return llmchain_chat

def get_cerebrium_waiting_time_generator():
    prompt_template = "Make a variation of the following sentence: \
    {sentence} \
    The sentence should keep the same meaning but should be different. Keep it simple. \
    "
    prompt = PromptTemplate(input_variables=["sentence"], template=prompt_template)
    cereb = CerebriumAI(endpoint_url="https://run.cerebrium.ai/v")

    llmchain_chat = LLMChain(llm=cereb, prompt=prompt, output_key= "variation")
    return llmchain_chat

def get_azureo_waiting_time_generator():
    prompt_template = "Make a variation of the following sentence: \
    {sentence} \
    The sentence should keep the same meaning but should be different. Keep it simple. \
    "
    prompt = PromptTemplate(input_variables=["sentence"], template=prompt_template)
    cereb = AzureChatOpenAI(
            deployment_name="gpt35t",
            openai_api_version="2023-03-15-preview",
        )
    llmchain_chat = LLMChain(llm=cereb, prompt=prompt, output_key= "variation")
    return llmchain_chat