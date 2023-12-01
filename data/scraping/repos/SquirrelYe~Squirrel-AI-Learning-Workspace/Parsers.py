from langchain.output_parsers import CommaSeparatedListOutputParser, DatetimeOutputParser, EnumOutputParser, OutputFixingParser
from langchain.output_parsers import PydanticOutputParser, RetryOutputParser, RetryWithErrorOutputParser, StructuredOutputParser
from langchain.output_parsers import ResponseSchema
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

from enum import Enum
from pydantic import BaseModel, Field, validator
from typing import List

model_name = "text-davinci-003"
chat_model_name = "gpt-3.5-turbo"
temperature = 0.0
model = OpenAI(model_name=model_name, temperature=temperature)
chat_model = ChatOpenAI(model_name=chat_model_name, temperature=temperature)

# ############### CommaSeparatedListOutputParser ################
# 当您想要返回以逗号分隔的项目列表时，可以使用此输出解析器。
def ListParserDemo():
    output_parser = CommaSeparatedListOutputParser()
    format_instructions = output_parser.get_format_instructions()
    prompt = PromptTemplate(
        template="List five {subject}.\n{format_instructions}",
        input_variables=["subject"],
        partial_variables={"format_instructions": format_instructions}
    )
    _input = prompt.format(subject="ice cream flavors")
    output = model(_input)
    result = output_parser.parse(output)
    print(_input)
    print(result)
    # ['Vanilla', 'Chocolate', 'Strawberry', 'Mint Chocolate Chip', 'Cookies and Cream']
    

# ############### DatetimeOutputParser ################
def DatetimeOutputParserDemo():
    output_parser = DatetimeOutputParser()
    template = """Answer the users question:
    {question}
    {format_instructions}"""
    prompt = PromptTemplate.from_template(
        template,
        partial_variables={"format_instructions": output_parser.get_format_instructions()},
    )
    chain = LLMChain(prompt=prompt, llm=OpenAI())
    output = chain.run(question = "around when was bitcoin founded?")
    result = output_parser.parse(output)
    print(prompt.format(question="around when was bitcoin founded?"))
    # Answer the users question:
    # around when was bitcoin founded?
    # Write a datetime string that matches the 
    #         following pattern: "%Y-%m-%dT%H:%M:%S.%fZ". Examples: 1084-12-04T02:48:41.297443Z, 1440-06-17T23:58:05.302963Z, 0664-07-16T15:12:19.847775Z
    
    print(output, result)
    # 2008-01-03T18:15:05.000000Z 2008-01-03 18:15:05


# ############### EnumOutputParser ################
def EnumOutputParserDemo():
    class Colors(Enum):
        RED = "red"
        GREEN = "green"
        BLUE = "blue"
        
    parser = EnumOutputParser(enum=Colors)
    parser.parse("red")
    parser.parse(" green")
    parser.parse("blue\n")
    parser.parse("yellow") # ValueError: 'yellow' is not a valid EnumOutputParserDemo.<locals>.Colors


# ############### OutputFixingParser ################
# # 该输出解析器包装另一个输出解析器，如果第一个解析器失败，它会调用另一个 LLM 来修复任何错误。
# 但除了抛出错误之外，我们还可以做其他事情。具体来说，我们可以将格式错误的输出以及格式化的指令传递给模型并要求其修复。
# 对于此示例，我们将使用上面的 Pydantic 输出解析器。如果我们传递一个不符合模式的结果，会发生以下情况：
def OutputFixingParserDemo():
    class Actor(BaseModel):
        name: str = Field(description="name of an actor")
        film_names: List[str] = Field(description="list of names of films they starred in")

    parser = PydanticOutputParser(pydantic_object=Actor)
    misformatted = "{'name': 'Tom Hanks', 'film_names': ['Forrest Gump']}"
    # parser.parse(misformatted)
    
    new_parser = OutputFixingParser.from_llm(parser=parser, llm=ChatOpenAI())
    result = new_parser.parse(misformatted)
    print(result.json())


# ############### PydanticOutputParser ################
# 使用 Pydantic 声明您的数据模型。
# Pydantic 的 BaseModel 就像 Python 数据类，但具有实际的类型检查 + 强制。
# 该输出解析器将输出解析为 Pydantic 对象。这对于将模型的输出转换为 Python 对象非常有用。
def PydanticOutputParserDemo():
    # 定义您想要的数据结构。
    class Joke(BaseModel):
        setup: str = Field(description="question to set up a joke")
        punchline: str = Field(description="answer to resolve the joke")

        # 你可以很容易地使用 Pydantic 添加自定义验证逻辑。
        @validator("setup")
        def question_ends_with_question_mark(cls, field):
            if field[-1] != "?":
                raise ValueError("Badly formed question!")
            return field

    parser = PydanticOutputParser(pydantic_object=Joke)
    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    _input = prompt.format_prompt(query="Tell me a joke.")
    output = model(_input.to_string())
    print(parser.parse(output).json())
    # {"setup": "Why did the chicken cross the road?", "punchline": "To get to the other side!"}
    
    # 另一个示例，使用复合字段
    class Actor(BaseModel):
        name: str = Field(description="name of an actor")
        film_names: List[str] = Field(description="list of names of films they starred in")

    parser = PydanticOutputParser(pydantic_object=Actor)
    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    _input = prompt.format_prompt(query="Generate the filmography for a random actor.")
    output = model(_input.to_string())
    print(parser.parse(output).json())
    # {"name": "Tom Hanks", "film_names": ["Forrest Gump", "Saving Private Ryan", "The Green Mile", "Cast Away", "Toy Story"]}


# ############### PydanticOutputParser ################
# 该输出解析器将重试模型，直到它返回一个有效的结果。
# 重试次数可以通过 max_retries 参数进行配置。
# 重试间隔可以通过 retry_interval 参数进行配置。
def RetryOutputParserDemo():
    class Action(BaseModel):
        action: str = Field(description="action to take")
        action_input: str = Field(description="input to the action")

    parser = PydanticOutputParser(pydantic_object=Action)
    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    prompt_value = prompt.format_prompt(query="who is leo di caprios gf?")
    bad_response = '{"action": "search"}'
    # parser.parse(bad_response)
    
    # 下面的Parser可以替换为RetryWithErrorOutputParser
    retry_parser = RetryOutputParser.from_llm(
        parser=parser,
        llm=model
    )
    print(retry_parser)
    print(retry_parser.parse_with_prompt(bad_response, prompt_value).json())


# ############### StructuredOutputParser ################
# 该输出解析器将输出解析为结构化的数据。
# 该解析器使用一个模板，该模板指定了如何将输出解析为结构化的数据。
def StructuredOutputParserDemo():
    response_schemas = [
        ResponseSchema(name="answer", description="answer to the user's question"),
        ResponseSchema(name="source", description="source used to answer the user's question, should be a website.")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    prompt = PromptTemplate(
        template="answer the users question as best as possible.\n{format_instructions}\n{question}",
        input_variables=["question"],
        partial_variables={"format_instructions": format_instructions}
    )
    
    _input = prompt.format_prompt(question="what's the capital of france?")
    output = model(_input.to_string())
    print(output_parser.parse(output))
    # {'answer': 'Paris', 'source': 'https://www.worldatlas.com/articles/what-is-the-capital-of-france.html'}

    # ### 在聊天模型中使用它的示例 ###
    prompt = ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template("answer the users question as best as possible.\n{format_instructions}\n{question}")  
        ],
        input_variables=["question"],
        partial_variables={"format_instructions": format_instructions}
    )
    _input = prompt.format_prompt(question="what's the capital of france?")
    output = chat_model(_input.to_messages())
    print(output_parser.parse(output.content))



if __name__ == "__main__":
    # ListParserDemo()
    # DatetimeOutputParserDemo()
    # EnumOutputParserDemo()
    # OutputFixingParserDemo()
    # PydanticOutputParserDemo()
    # RetryOutputParserDemo()
    StructuredOutputParserDemo()