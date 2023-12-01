from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv('.env')
jlpt_level = os.environ.get("JLPT_LEVEL")


llm = OpenAI()
chat_model = ChatOpenAI()

dictionary_df = pd.read_csv("dictionary.csv")
selected_words = dictionary_df.sample(n=50)
kanji_meaning_pairs = list(zip(selected_words['kanji'], selected_words['meaning']))
long_string = ', '.join(f"{kanji} ({meaning})" for kanji, meaning in kanji_meaning_pairs)

num_sentences = 10

prompt = f"You are a Japanese language teacher creates sentences in Japanese and English for students. Given a list of Japanese words and their English meaning, create sentences that are for difficulty level {jlpt_level}. Use a few  words in your sentences: {long_string}. The sentence should be short. Please provide the translated version of the sentence also."
for _ in range(num_sentences):
    response = llm.predict(prompt)
    print(response)

