import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
import pprint as pp

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


def main():
    llm = OpenAI(
        openai_api_key=api_key,
        model="gpt-3.5-turbo-instruct",
    )
    result = llm.generate(
        [
            "Write a limerick about taylor swift",
            "tell me a joke",
        ]
    )
    # This time, it returns an LLM result object as opposed to just a string
    print(result)
    # print(result.generations)
    # print(type(result.generations))
    # print(result.schema())
    # print(result.llm_output)


if __name__ == "__main__":
    main()
