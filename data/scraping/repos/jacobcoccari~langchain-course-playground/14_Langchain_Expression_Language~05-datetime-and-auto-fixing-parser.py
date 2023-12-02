# Imports done at the beginning of each file.
import os
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
from langchain.chat_models import ChatOpenAI
model = ChatOpenAI(openai_api_key=openai_api_key)

from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import DatetimeOutputParser
from langchain.output_parsers import OutputFixingParser

def main():
    output_parser = DatetimeOutputParser()
    template = """Answer the users question:

    {question}

    {format_instructions}"""

    prompt = ChatPromptTemplate.from_template(
        template,
        partial_variables={"format_instructions": output_parser.get_format_instructions()},
    )
    auto_fixing_parser = OutputFixingParser.from_llm(parser=output_parser, llm=model)
    chain = prompt | model | auto_fixing_parser
    output = chain.invoke({"question": "around when was bitcoin founded?"})
    print(output)


if __name__ == "__main__":
    main()