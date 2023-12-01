import json
import os

from dotenv import load_dotenv
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser

from quiz_pydantic import Quiz
from quiz_prompt_template import quiz_pydantic_prompt


def lambda_handler():
    with open('summary.json') as file:
        summary = json.load(file)
    quiz = make_quiz(summary['output_text'])
    print(quiz)


# Load and run summarization chain
# https://python.langchain.com/en/latest/modules/chains/index_examples/summarize.html
# The recommended way to get started using a summarization chain is: map_reduce
# different chain types: stuff, map_reduce, and refine
# https://docs.langchain.com/docs/components/chains/index_related_chains
def make_quiz(summary):
    # Load environment variables from .env file
    load_dotenv()
    # Initialize OpenAI model and text splitter
    # model_name="gpt-4-32k",
    llm = ChatOpenAI(model_name="gpt-4", openai_api_key=os.getenv("OPENAI_API_KEY"))
    # Set up a parser + inject instructions into the prompt template.
    parser = PydanticOutputParser(pydantic_object=Quiz)
    prompt = quiz_pydantic_prompt(parser)
    chain = LLMChain(llm=llm, verbose=True, prompt=prompt)

    config = {
        "text": summary,
        "language": 'English',
        "question_count": 5,
        "alternative_count": 4,
        "difficulty_level": map_difficulty_level(3)
    }
    output = chain.run(config)
    return parser.parse(output).json()


def map_difficulty_level(value):
    levels = {
        1: "10 years old, Question points: 1",
        2: "15 years old, Question points: 2-3",
        3: "adult, Question points: 4-5",
        4: "professional, Question points: 6-8",
        5: "master, Question points: 9-10"
    }
    level = levels.get(value, levels[2])
    return level


lambda_handler()
