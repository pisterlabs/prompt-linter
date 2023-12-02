from langchain.prompts import ChatPromptTemplate,PromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate,AIMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# def getChatTemplate(mem):
#     system_template = "As a generative ticket writing assistant, my goal is to output actionable {result_type} to be consumed by a scrum team."
#     system_prompt = SystemMessagePromptTemplate.from_template(system_template)

#     human_template = "Output the {result_type} based on this conversation: {conversation}"
#     human_prompt = HumanMessagePromptTemplate.from_template(human_template)

#     chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
#     formatted_chat_prompt = chat_prompt.format_prompt(result_type="acceptance criteria", conversation=mem).to_messages()
#     return formatted_chat_prompt

def getSystemMessageTemplate():
    system_template = """
    As an generative ticket writing assistant, your goal is to create and output actionable {result_type} for a {ticket_type} ticket. 
    Output the {result_type} for the {ticket_type} ticket so that it can be consumed by a scrum team."
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    return system_message_prompt

def getHumanMessageTemplate():
    human_template = """
    I need AI-generated {result_type} output for a {ticket_type} ticket. 
    The output should be formatted in {format_type} and be pastable into a Jira panel.
    Base the output off of the following conversation: {conversation}"
    f"{use_natural_language}"
    f"{output_template}"
    """
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    return human_message_prompt

def getPromptTemplate():
    chat_prompt = ChatPromptTemplate.from_messages([getSystemMessageTemplate(), getHumanMessageTemplate()])
    return chat_prompt

def formatPrompt(chat_prompt, input_conversation, input_ticket_type, input_result_type, input_format_type, input_natural_language="", input_output_template=""):
    prompt = chat_prompt.format_prompt(conversation=input_conversation
                                ,ticket_type=input_ticket_type
                                ,result_type=input_result_type
                                ,format_type=input_format_type
                                ,use_natural_language=input_natural_language
                                ,output_template=input_output_template).to_messages()
    return prompt

def getGherkinTemplate(conversation, ticket_type="user story", result_type="acceptance criteria", format_type="Gherkin"):
    return formatPrompt(getPromptTemplate(), conversation, ticket_type, result_type, format_type)

def getMarkdownTemplate(conversation, ticket_type="user story", result_type="acceptance criteria", format_type="Markdown"):
    return formatPrompt(getPromptTemplate(), conversation, ticket_type, result_type, format_type)

def getPlainTextTemplate(conversation, ticket_type="user story", result_type="acceptance criteria", format_type="Plain Text"):
    return formatPrompt(getPromptTemplate(), conversation, ticket_type, result_type, format_type)

def getSqlScriptTemplate(conversation, ticket_type="user story", result_type="acceptance criteria", format_type="SQL Script", natural_language="Do not output any natural language.", output_template="{ query_index: query_contents }"):
    return formatPrompt(getPromptTemplate(), conversation, ticket_type, result_type, format_type, natural_language, output_template)