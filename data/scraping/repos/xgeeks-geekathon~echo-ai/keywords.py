from keybert import KeyBERT

from langchain.chat_models import ChatOpenAI
from langchain.agents import (
    load_tools,
    AgentType,
    initialize_agent
)
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    HumanMessagePromptTemplate,
)

def detectKeywords(doc):
    kw_model = KeyBERT()
    keyword_list = kw_model.extract_keywords(doc, stop_words=None, use_mmr=True, diversity=0.1)
    keywords = [keyword[0] for keyword in keyword_list]

    return keywords

def createDefinitions(doc, summary):
    print("Getting keywords...")
    keywords_list = detectKeywords(doc)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)

    tool_names = ["serpapi"]
    tools = load_tools(tool_names)

    agent_chain = initialize_agent(tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION)
    
    result_list = []

    for keyword in keywords_list:
        agent_instructions = f"""
        Given the following context {summary}, what is {keyword}?
        """
        description = agent_chain.run(agent_instructions)

        result_list.append(formatOutput(description).content)
    
    result = "\n\n".join(result_list)

    print("Done!")
    
    return result

def formatOutput(description):
    promptTemplate = PromptTemplate(
        template = """
        Given the following description extract the keyword and provide a concise and professional definition:

        Description: {description}

        The output should be in the following format:
        (Capitalized keyword) : (definition)
        """,
        input_variables=["description"]
    )

    human_message_prompt = HumanMessagePromptTemplate(prompt=promptTemplate)

    chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])
    chat_messages = chat_prompt.format_prompt(description= description).to_messages()

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)

    return llm(chat_messages)