import os
from dotenv import load_dotenv

from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import DatetimeOutputParser
from langchain.chat_models import ChatOpenAI


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(
    openai_api_key=api_key,
    temperature=0,
)


def main():
    output_parser = DatetimeOutputParser()
    system_prompt = "Only respond in datetime formats."
    human_prompt = "Answer the user's question: \n {question} \n {format_instructions}"
    prompt_template = ChatPromptTemplate.from_messages(
        [
            system_prompt,
            human_prompt,
        ]
    )
    input = prompt_template.format_prompt(
        question="On what date did world war two end?",
        format_instructions=output_parser.get_format_instructions(),
    ).to_messages()
    output = model(input).content
    result = output_parser.parse(output)
    print(result)


if __name__ == "__main__":
    main()
