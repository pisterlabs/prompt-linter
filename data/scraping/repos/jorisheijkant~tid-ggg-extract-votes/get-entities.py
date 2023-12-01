import os
import json
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI


# Set the chunks folder path and the city name
target_folder = "data/voting-test"
chunk_folder = f"{target_folder}/chunks"

lines = []
# Read the json files and add the lines to the lines array
for filename in os.listdir(chunk_folder):
    if filename.endswith(".json"):
        with open(f"{chunk_folder}/{filename}") as f:
            json_file = json.load(f)
            # Pick the first five lines from the file and add them to the lines array
            for index, line in enumerate(json_file):
                if(index < 5):
                    lines.append(line)
                
print(f"Found {len(lines)} lines in the chunks folder that we are going to scan for parties")

# Import the keys and url from the constants file
import constants
os.environ["OPENAI_API_KEY"] = constants.APIKEY
os.environ["OPENAI_APIKEY"] = constants.APIKEY

# Set up LLM and prompt 
# TODO: dynamically set the city here in the prompt
prompt_template = """
Je bekijkt een stuk tekst uit een gemeenteraadsvergadering van Helmond of ander gemeentelijk document uit Helmond.
Je gaat op zoek naar politieke partijen of afkortingen van politieke partijen in dit stuk tekst.
Als je deze vind, dan geef je een json object terug met de volgende structuur:
- "text": de tekst die je gevonden hebt
- "party": de volledige naam van de politieke partij
- "abbreviation": de afkorting van de politieke partij
Gebruik hierbij eventueel je kennis van de Nederlandse politiek en de politiek in Helmond om het object aan te vullen als je alleen een partij of afkorting vindt.
Vind je geen partijen? Geef dan Null (met hoofdletter) terug.
Let op: B&W, Burgmeesters en Wethouders, Gemeenteraad en Raad zijn geen politieke partijen.

Context: 
{context}
"""

llm = ChatOpenAI(model="gpt-4")
llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template)
)

parties_found = []
unique_parties = []

# Loop over the lines
for line in lines:
    print(f"Processing line: {line}")
    result = llm_chain(line["text"])
    if(result):
        result_text = result["text"]
        if(result_text != "Null"):
            parties_found.append(result_text)
        else:
            print("No parties found")

print(parties_found)

for parties_line in parties_found:
    json_line = json.loads(parties_line)
    for p in json_line:
        # Check if unique_parties already contains the party, checking for an object with the same abbreviation
        if(type(p) is dict and p["abbreviation"] and p["abbreviation"] != "Null"):
          if not any(party["abbreviation"] == p["abbreviation"] for party in unique_parties):
              unique_parties.append({
                  "party": p["party"],
                  "abbreviation": p["abbreviation"]
              })
          else:
              print(f"Party {p['party']} already in array")

print(unique_parties)

# Write out the parties to a file
with open(f"{target_folder}/parties.json", "w") as f:
    json.dump(unique_parties, f, indent=4, ensure_ascii=False)