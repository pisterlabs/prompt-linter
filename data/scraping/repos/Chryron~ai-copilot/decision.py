from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, AIMessage, HumanMessage

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler 

from langchain.prompts import FewShotPromptTemplate

from langchain import PromptTemplate

from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from pydantic import BaseModel, Field, validator
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

def process_input(query:str):
    context = identify_context(query=query)
    
    if context == "new":
        result = new_code_function(query=query)
    elif context == "action":
        result = perform_action(query=query)
    elif context == "unrelated":
        result = unrelated(query=query)
    
    return result


def new_code_function(query:str):
    function_name = find_existing_code_function(query=query)
    if function_name is None:
        return "New function created!"
    return f"Function {function_name} already exists!"

def perform_action(query:str):
    pass

def unrelated(query:str):
    pass

def find_existing_code_function(query:str):
    system_message_template = """
    You are a senior software engineer in charge of identifying if a user's feature request is already implemented in the codebase.
    Your task is to ONLY respond with the name of the function that implements the feature request ONLY IF IT EXISTS. DO NOT WRITE ANY CODE TO IMPLEMENT THE FEATURE REQUEST.
    
    {format_instructions}

    Here all the functions in the codebase (if none of the provided functions implement the feature request, respond with None):
    """
    
    examples = [{
        "function": """def hello_world():
        \"\"\"Prints 'Hello, World!' to the console.\"\"\"
        print("Hello, World!")"""
    }]
    example_prompt = PromptTemplate.from_template("{function}")

    class ChatBotOutput(BaseModel):
        query: str = Field(description="the user's query")
        already_exists: bool = Field(description="whether or not the function exists in the codebase")
        function_name: Optional[str] = Field(description="the name of the function that implements the user's feature request. If none of the provided functions implement the feature request, respond with None")

    parser = PydanticOutputParser(pydantic_object = ChatBotOutput)

    prompt_with_examples = FewShotPromptTemplate(
        input_variables=["format_instructions"],
        output_parser=parser,
        examples=examples,
        example_prompt=example_prompt,
        prefix=system_message_template,
        suffix=""
    )
    system_message_with_examples = prompt_with_examples.format_prompt(format_instructions=parser.get_format_instructions())
    
    
    system_message = SystemMessage(content = system_message_with_examples.text)
    human_message = HumanMessage(content = query)

    
    chat = ChatOpenAI(model_name="gpt-3.5-turbo")
    result = chat([system_message, human_message])
    try:
        parsed_result = parser.parse(result.content)
    except Exception as e:
        fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=ChatOpenAI())
        parsed_result = fixing_parser.parse(result.content)
    return parsed_result.function_name

def identify_context(query:str):
    examples = [{
        "query": "My name is John", 
        "context": "unrelated",
        },
        {
        "query": "reverse a linked list", 
        "context": "action",
        },
        {
        "query": "What is the time complexity of quicksort?",
        "context": "unrelated",
        },
        {
        "query": "write a function that returns the sum of two numbers", 
        "context": "new",
        },
        {
        "query": "write a function that reverses a linked list", 
        "context": "new",
        },
        {
        "query": "run the function you just wrote", 
        "context": "action",
        },
        ]
    
    example_template = """"query": {query}, 
    "context": {context}"""
    example_prompt = PromptTemplate.from_template(example_template)

    system_message_template = """
    You are an LLM in charge of identifying the context of a user's query. 
    
    There are three possible contexts to the user's query:
    1. The user is providing an instruction to write a new code function. [new]
    2. The user is providing an instruction to perform an action to a codebase. [action]
    3. The user has an natural language query unrelated to making changes to a codebase. [unrelated]
    Your task is to ONLY respond with the CONTEXT of the user's query. DO NOT RESPOND TO THE QUERY ITSELF.
    {format_instructions}

    Here are some examples of queries and their contexts:
    """

    class ChatBotOutput(BaseModel):
        query: str = Field(description="the user's query")
        context: Optional[str] = Field(description="the context of the user's query, one of [new, action, unrelated]")

        @validator('context')
        def context_must_be_valid(cls, v):
            if v not in ["new", "action", "unrelated"]:
                raise ValueError("context must be one of [new, action, unrelated]")
            return v
    
    parser = PydanticOutputParser(pydantic_object = ChatBotOutput)

    prompt_with_examples = FewShotPromptTemplate(
        input_variables=["format_instructions"],
        output_parser=parser,
        examples=examples,
        example_prompt=example_prompt,
        prefix=system_message_template,
        suffix=""
    )

    system_message_with_examples = prompt_with_examples.format_prompt(format_instructions=parser.get_format_instructions())
    
    
    system_message = SystemMessage(content = system_message_with_examples.text)
    human_message = HumanMessage(content = query)

    
    chat = ChatOpenAI(model_name="gpt-3.5-turbo")
    result = chat([system_message, human_message])
    try:
        parsed_result = parser.parse(result.content)
    except Exception as e:
        fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=ChatOpenAI())
        parsed_result = fixing_parser.parse(result.content)
    return parsed_result.context

