# first attempts of custom agents
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.agents.tools import Tool
from langchain import LLMMathChain, LLMChain, PromptTemplate
from langchain.agents import AgentType, Agent, AgentExecutor, ConversationalChatAgent, BaseMultiActionAgent, AgentOutputParser, initialize_agent
from indexRetriever import answer_retriever, question_generator
from questionGenerator import random_question_tool, answer_comparison
from factchecker import fact_check
from dotenv import load_dotenv

def generate_agent007(memory):
    
    answer_retrieve = answer_retriever()
    # answer_compare = Tool.from_function(
    #     answer_comparison,
    #     name = "answer comparison",
    #     description="Rufe die Funktion auf, um die Antwort des Studierenden auf die Frage des Tutors mit der Antwort des Tutors zu vergleichen und um mitzuteilen, ob die Antwort korrekt war. Der Input sollte in json-Format sein mit question: und answer: als key-values.",
    #     return_direct=True,
    # )
    generate_question = question_generator()
    
    answer_lecture_question = Tool(
        name="answer lecture Question",
        func=answer_retrieve.run,
        description="Rufe die Funktion auf, um Fragen zur Vorlesung Datenbanken und Informationssysteme zu beantworten.",
        return_direct=True,        
    )
    tutor_question = Tool.from_function(
        random_question_tool, 
        name="tutor Question", 
        description="Rufe die Funktion auf, um eine zufällige Frage zu stellen, die ein Tutor stellen könnte. Übersetze sie aber davor ins Deutsche.",
        return_direct=True
    )
    specific_question = Tool(
        name="specific Question",
        func=generate_question.run,
        description="Rufe die Funktion auf, um eine Frage zu einem spezifischen Thema zu generieren. Übersetze sie aber davor ins Deutsche.",
        return_direct=True,
    )
    
    tools = [
        answer_lecture_question,
        tutor_question,
        specific_question
        #answer_compare
    ]
    
    prefix="""Du bist Tutor für die Vorlesung Datenbanken und Informationssysteme an der Universität RWTH Aachen. Dein Ziel ist die Studenten bei der Vorlesung zu unterstützen und Fragen zur Vorlesung zu beantworten, indem du eine Konversation mit ihnen führst.
    Ebenso kannst du den Studierenden Übungsaufgaben stellen und ihre Antworten korrigieren. Da die Vorlesung auf Deutsch gehalten wird, solltest du auch auf Deutsch antworten. Du kannst nur Fragen zur Vorlesung beantworten. Die Antwort auf Inhalte, die nicht Teil der Vorlesung sind, sollen verweigert werden.
    Sei immer freundlich und wenn du eine Frage nicht beantworten kannst, gib dies zu. 
    Zusammengefasst ist der Tutor ein mächtiges System, das bei einer Vielzahl von Aufgaben helfen kann und wertvolle Einblicke und Informationen zu einer Vielzahl von Themen liefern kann. Ob du Hilfe bei einer bestimmten Frage benötigst oder einfach nur ein Gespräch über ein bestimmtes Thema führen möchtest, der Tutor ist hier, um zu helfen."""
    
    FORMAT_INSTRUCTIONS = """RESPONSE FORMAT INSTRUCTIONS
    ----------------------------

    When responding to me, please output a response in one of two formats:

    **Option 1:**
    Use this if you want the human to use a tool.
    Markdown code snippet formatted in the following schema:

    ```json
    {{{{
        "action": string, \\ The action to take. Must be one of {tool_names}
        "action_input": string \\ The input to the action
    }}}}
    ```

    **Option #2:**
    Use this if you want to respond directly to the human. Markdown code snippet formatted in the following schema:

    ```json
    {{{{
        "action": "Final Answer", \\ Name the tool name used {tool_names}
        "action_input": string \\ You should put what you want to return to use here
    }}}}
    ```"""
    
    suffix="""TOOLS
    ------
    Der Tutor kann tools aufrufen, die für die Beantwortung der Fragen nützlich sind.

    {{tools}}
    
    {format_instructions}
    
    CHAT HISTORY
    ------------
    Der Tutor hat auch Zugriff auf den vorherigen Chat-Verlauf, um noch besser auf die Fragen der Studierenden eingehen zu können.
        
    USER'S INPUT
    --------------------
    Hier ist die Eingabe vom User:
    {{{{input}}}}"""

    llm = ChatOpenAI(temperature=0, model="gpt-4")
    agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, system_message=prefix, human_message=suffix, tools=tools, verbose=True)
    agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory, handle_parsing_errors=True)

    return agent_chain

