import readline
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms.openai import OpenAI
from utils import repl

def main():
    llm = OpenAI(temperature=0, model_name="text-davinci-003")
    print(f"ask me anything, this is using openai's {llm.model_name} model for completion")

    repl(lambda user_input: llm(user_input))

if __name__ == "__main__":
    main()
