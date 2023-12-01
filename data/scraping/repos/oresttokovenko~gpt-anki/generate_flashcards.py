from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate

from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import List
import pandas as pd
import os

# loading api key and defining vars

load_dotenv("../.env")
deck_name = os.environ.get("DECK_NAME")
csv_file_path = "../csv/" + deck_name + ".csv"

openai_api_key = os.environ.get("OPENAI_API_KEY")
model = "gpt-4"
temperature = 0.0


class FlashCard(BaseModel):
    question: str = Field(description="The question for the flashcard")
    answer: str = Field(description="The answer for the flashcard")

class FlashCardArray(BaseModel):
    flashcards: List[FlashCard]

def create_flashcards_from_text(input_text: str, user_prompt: str, csv_file_path: str):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model=model, temperature=temperature)

    print("Creating flashcards...")

    pydantic_parser = PydanticOutputParser(pydantic_object=FlashCardArray)

    format_instructions = pydantic_parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_template(template=user_prompt)

    messages = prompt.format_messages(input_text=input_text, format_instructions=format_instructions)

    output = llm(messages)

    flashcards = pydantic_parser.parse(output.content)

    list_of_flashcards = [card.dict() for card in flashcards.flashcards]

    df = pd.DataFrame(list_of_flashcards)

    if os.path.isfile(csv_file_path):
        df.to_csv(csv_file_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_file_path, mode="w", header=False, index=False)

def main():
    try:
        with open("input.txt", "r") as f:
            input_text = f.read()

        with open("prompt.txt", "r") as f:
            user_prompt = f.read()

        create_flashcards_from_text(input_text, user_prompt, csv_file_path)

    except Exception as e:
        print(f"Error occurred: {e}")
        return


if __name__ == "__main__":
    main()
