import os, requests
from dotenv import load_dotenv
from query import query

# 3rd party
import openai


load_dotenv()


def generate_prompt(user_prompt):
    return """You are a creative and witty wordsmith well known for your ability to coin novel, humorous new names.
    Please generate a list of 10 {} related Ethereum .ens names.
    Avoid all use of controversial concepts relating to religion, profanity, sex or violence.""".format(
        user_prompt.capitalize()
    )


openai.api_key = os.getenv("OPENAI_API_KEY")

user_prompt = input("Search bar:")


response = openai.Completion.create(
  model="gpt-3.5-turbo-instruct",
  prompt=generate_prompt(user_prompt),
  temperature=1.1,
  max_tokens=120
)

result = response.choices[0].text
token_stats = response.usage

raw_names = result.split("\n")
names = []

for name in raw_names:
    if name: names.append(name.split(".")[1].strip())

print(names)


url = "https://graphigo.prd.space.id/query"

# Variables for the GraphQL query
variables = {
    "input": {
        "query": names[0],
        "buyNow": 1,
        "domainStatuses": ["REGISTERED", "UNREGISTERED"],
        "first": 30
    }
}

# Send POST request to the API endpoint with the GraphQL query and variables
response = requests.post(url, json={"query": query, "variables": variables})
space_id_response = None

# Check if the request was successful
if response.status_code == 200:
    data = response.json()
    # print("API Response:")
    space_id_response = data["data"]["domains"]["exactMatch"]
    # print(space_id_response)
else:
    print(f"Error: Unable to fetch data from the API. Status code: {response.status_code}")
    print(response)

links = []
prices = []

if not space_id_response:
    print("spaceID no response, exiting program")
    exit()

for id in space_id_response:
    links.append(f"https://space.id/name/{str(id['tld']['tldID'])}/" + str(id["tokenId"] + "?name=" + str(id["name"] + "&tldName=" + str(id['tld']['tldName']))))
    prices.append(id["listPrice"])

print("links: ", links)
print("prices: ", prices)



