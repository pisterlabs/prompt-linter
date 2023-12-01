import os
from typing import List

from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
import orjson

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = OpenAI(
    model_name="text-davinci-003", openai_api_key=OPENAI_API_KEY, max_tokens=3000
)


class Location(BaseModel):
    city: str = Field(description="city of a location")
    country: str = Field(description="country of a location")


class LocationDict(BaseModel):
    locations: dict[str, List[Location]] = Field(
        description="dictionary of lists of locations with the original text as key"
    )


# location_query = "Extract the city and the country from a location in a json format."
location_query = """I have a list of locations but it is really badly designed.
I want you to extract the city and the country from a location in a json format.
I don't want a zipcode. Only city and country. Cities and countries should only be strings.
Countries should not be acronyms: for example "USA" should be change to "United States".
Infer the country if needed.
"""

parser = PydanticOutputParser(pydantic_object=LocationDict)

print(parser.get_format_instructions())

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\nFormat the following locations:\n{locations}\n",
    input_variables=["query", "locations"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


def clean_locations(locations: list[str]):
    _input = prompt.format_prompt(query=location_query, locations=locations)
    print(_input.to_string())

    output = llm(_input.to_string())
    print(output)

    data = parser.parse(output)
    data

    for value in data.locations.values():
        for location in value:
            if location.city == "Zürich":
                location.city = "Zurich"

    # data.json(indent=2, ensure_ascii=False)
    return orjson.loads(data.json())


if __name__ == "__main__":
    locations = [
        "St-Sulpice / VD",
        "Geneva",
        "Paris",
        "Yverdon",
        "GENEVA",
        "EPFL Innovation Park",
        "Neuchâtel",
        "Josefstrasse 219, Zurich",
        "Zurich",
        "EDF R&D Renardières",
        "Lausanne, EPFL Innovation Park",
        "Fribourg",
        "Renens",
        "Antony, France",
        "Genève",
        "Le Bourget du Lac (France,73)",
        "Swisscom Digital Lab, EPFL Innovation Park, Bat F, 1015 Lausanne",
        "Zürich, Nyon or Homeoffice",
        "Oracle Labs Zurich, Switzerland (other locations or work from home available upon agreement)",
        "Geneve",
        "remote in Switzerland (we also have an office in Zürich)",
        "Zürich",
        "Chilly-Mazarin",
        "Sophia Antipolis - France",
    ]
    clean_locations(locations)
