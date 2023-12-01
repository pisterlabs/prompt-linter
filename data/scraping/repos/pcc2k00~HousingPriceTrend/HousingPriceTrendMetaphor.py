import openai
import yaml

from metaphor_python import Metaphor

with open("pass.yml") as f:
    content = f.read()

my_credentials = yaml.load(content, Loader=yaml.FullLoader)

openai.api_key = my_credentials["openAi"]
metaphor = Metaphor(my_credentials["metaphor"])

USER_QUESTION = "Recent housing price in Seattle"

SYSTEM_MESSAGE = "You are a helpful assistant that generates search queiries based on user questions. Only generate one search query."
completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": USER_QUESTION},
    ],
)
query = completion.choices[0].message.content
search_response = metaphor.search(
    query, use_autoprompt=True, start_published_date="2023-07-01"
)
contents_result = search_response.get_contents()
first_result = contents_result.contents[0]

SYSTEM_MESSAGE = "You are a helpful assistant that summarizes the content of a webpage. Summarize the users input."
completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": first_result.extract},
    ],
)
summary = completion.choices[0].message.content
print(f"Summary for {first_result.title}: {summary}")
