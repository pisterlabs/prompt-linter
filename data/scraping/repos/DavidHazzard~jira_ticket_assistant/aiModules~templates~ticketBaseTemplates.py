from langchain.prompts import ChatPromptTemplate,PromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate,AIMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage

def getSystemMessageTemplate():
    system_template = "You are an AI Jira ticket writing assistant that specializes in ticket generation and refinement. You are currently assisting a {client} stakeholder by constructing {result_type} for a {ticket_type} ticket."
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    return system_message_prompt

def getHumanMessageTemplate():
    human_template = "I am a {role} stakeholder for {client}. I need to write the {result_type} ticket for a {ticket_type}."
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    return human_message_prompt

def getPromptTemplate():
    chat_prompt = ChatPromptTemplate.from_messages([getSystemMessageTemplate(), getHumanMessageTemplate()])
    return chat_prompt

def formatPrompt(chat_prompt, input_client, input_role, input_ticket_type, input_result_type):
    prompt = chat_prompt.format_prompt(client=input_client
                                ,role=input_role
                                ,ticket_type=input_ticket_type
                                ,result_type=input_result_type).to_messages()
    return prompt

def getAITemplate():
    ai_template = "Hi there! I'm Reepicheep, your ticket writing assistant. What type of ticket can I help you write today?"
    ai_message_prompt = AIMessagePromptTemplate.from_template(ai_template)
    return ai_message_prompt

def getUserStoryTemplate(client="business team", role="business analyst", ticket_type="user story", result_type="acceptance criteria"):
    return formatPrompt(getPromptTemplate(), client, role, ticket_type, result_type)

def getBugReportTemplate(client="business team", role="business analyst", ticket_type="bug report", result_type="expected behavior"):
    return formatPrompt(getPromptTemplate(), client, role, ticket_type, result_type)

def getTestPlanTemplate(client="business team", role="software quality assurance engineer", ticket_type="test plan", result_type="test plan"):
    return formatPrompt(getPromptTemplate(), client, role, ticket_type, result_type)

def getTestCasesTemplate(client="business team", role="software quality assurance engineer", ticket_type="test cases", result_type="test cases"):
    return formatPrompt(getPromptTemplate(), client, role, ticket_type, result_type)

def getDbQueryTemplate(client="business team", role="software quality assurance engineer", ticket_type="database query", result_type="database query"):
    return formatPrompt(getPromptTemplate(), client, role, ticket_type, result_type)

def getRegressionRiskTemplate(client="business team", role="software developer", ticket_type="regression risk", result_type="regression risk"):
    return formatPrompt(getPromptTemplate(), client, role, ticket_type, result_type)