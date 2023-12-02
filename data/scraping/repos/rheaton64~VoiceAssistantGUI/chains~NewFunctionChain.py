from langchain import LLMChain
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from chains.NEW_FUNC import instructions
from dotenv import load_dotenv
load_dotenv(override=True)
def load_new_function_chain():
    escaped_instructions = instructions.replace('{', '{{').replace('}', '}}')
    system_message = SystemMessagePromptTemplate.from_template("""You are an expert programmer that can write any Python function I ask you to.
    You must also define a custom tool for each function that you write.
    This tool will be used by an AI agent to execute your function, so make sure that it is easy to use, general, and robust.
    Your function may be used in the creation of another function, so ensure that it is well documented and easy to understand.
    Your function should not be too specific, and should be able to be used in a variety of situations.
    Please follow the instruction below to define a function and its tool.
    INSTRUCTIONS:
                                                            
    ====================
    {}
    ====================

    Remember, try your best to make the function simple, reusable, and well documented.
    Be sure to only respond with the new code, nothing else.
    Begin!""".format(escaped_instructions))

    human_message = HumanMessagePromptTemplate.from_template("""{input}""")
    chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

    llm = ChatOpenAI(model='gpt-4-0613', temperature=0, streaming=True, callbacks=[StreamingStdOutCallbackHandler()])

    return LLMChain(llm=llm, prompt=chat_prompt)

    