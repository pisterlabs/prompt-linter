from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from csirt_config_info import get_openai_config
# from langchain.schema import HumanMessage
def ask_GPT(reference_ticket, ticket):

    openai_api_key = get_openai_config()
    # Construct chat model
    chatModel = ChatOpenAI(
        model="gpt-3.5-turbo-16k-0613",
        openai_api_key=openai_api_key
    )

    # Build prompt
    # next_action_template = '{"mssmt__severity__c": {"critical or high or mid or low"}\n"mssmt__MSS_VulnerableTodo__c": [{"mssmt__Subject__c": "Todo subject", "mssmt__Description__c": "Todo description", "mssmt__Priority__c": "Todo priority"}]\"nmssmt__MSS_VulnerabilityNote__c":\n[{"mssmt__Content__c": Note description}]}'
    next_action_template = '{"severity": {"critical or high or mid or low"}\n"MSS_VulnerableTodo": [{"Subject": "Todo subject", "Description": "Todo description", "Priority": "Todo priority"}]\n"MSS_VulnerabilityNote":\n[{"Content": Note description}]}'
    template = 'Base on these ticket:\n{reference_ticket}\n"mssmt__MSS_VulnerableTodo__c" and "mssmt__MSS_VulnerabilityNote__c" are next actions.\nPlease create properly next actions without any explainations for this ticket:\n{ticket}\nNext action is in this following format:\n{next_action}'
    humanMessagePrompt = HumanMessagePromptTemplate.from_template(template=template)
    chatPrompTemplate = ChatPromptTemplate.from_messages([humanMessagePrompt])

    # Ask chatGPT
    chain = LLMChain(llm=chatModel, prompt=chatPrompTemplate)
    print(chain.prompt)
    answer = chain.run(
        reference_ticket=reference_ticket,
        ticket=ticket,
        next_action=next_action_template
    )
    return answer