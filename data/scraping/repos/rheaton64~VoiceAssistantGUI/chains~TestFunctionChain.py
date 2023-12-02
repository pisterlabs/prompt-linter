from langchain import LLMChain
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from dotenv import load_dotenv
load_dotenv(override=True)
def load_test_function_chain():
    system_message = SystemMessagePromptTemplate.from_template("""You are an expert programmer and QA engineer that can write comprehensive unit tests for any Python function or tool that I ask you to.
    You create unit tests for the given function, and write them in well-documented Python code.
    The tests should run as-is, and should not require any edits or modifications from the user.
    The tests will be run on the 'tool' that wraps the given function. This tool will be used by an AI agent to execute the function, and will always be used to execute the function.
    Be sure to exhaustively test the tool, and ensure that it works in all cases.
    Please follow the instruction below to analyze a function and create unit tests for it.
    INSTRUCTIONS:

    ====================
    1. Understand the given function, and reason about its behaviour.
    2. Consider the different cases that the function may encounter, especially edge cases.
    3. List all of the cases that you have considered, and that you will test for.
    4. Create unit tests for the function, ensuring that all cases are covered.
    ====================

    Remember, you don't need to write the function, just the unit tests. The tool definition will be appended to the beginning of the code that you write, so DO NOT import the tool.
    The input will contain the function, its tool, and possible some additional explanation. You should only write unit tests for the tool.
    Be sure to only respond with the new code, nothing else.
    Begin!""")

    human_message = HumanMessagePromptTemplate.from_template("""Here is the function and corresponding tool that you must test:

    {input}""")
    chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

    llm = ChatOpenAI(model='gpt-4-0613', temperature=0, streaming=True, callbacks=[StreamingStdOutCallbackHandler()])

    return LLMChain(llm=llm, prompt=chat_prompt)
