
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from pathlib import Path
import json

from utils import load_and_convert_to_json, get_credentials

OPEN_AI_API, ACTIVELOOP_TOKEN = get_credentials()

json_data = load_and_convert_to_json("data/processed/movie_and_plot.csv")

prompt = PromptTemplate(
    input_variables=['movie_info'],
    template = Path("prompts/create_attributes.prompt").read_text()
)

llm = ChatOpenAI(openai_api_key=OPEN_AI_API, model="gpt-3.5-turbo", temperature=0.3)

chain = LLMChain(llm=llm, prompt=prompt)

new_data = {}

for i, movie in enumerate(json_data):
    info = ""
    for key, value in movie.items():
        info += str(value) + ","

    attribs = chain.run(movie_info=info)
    new_data[movie["title"]] = attribs

    print(f"Iter: {i} | Movie: {movie['title']}")


with open("data/processed/movie_attributes.json", "w") as f:
    json.dump(new_data, f)


