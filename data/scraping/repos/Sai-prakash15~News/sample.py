import os
import openai
from metaphor_python import Metaphor

openai.api_key = os.getenv("OPENAI_API_KEY")

metaphor = Metaphor(os.getenv("METAPHOR_API_KEY"))

USER_QUESTION = "Adele songs"

SYSTEM_MESSAGE = "You are a helpful assistant that generates search queiries based on user questions. Only generate one search query."

# completion = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role": "system", "content": SYSTEM_MESSAGE},
#         {"role": "user", "content": USER_QUESTION},
#     ],
# )


# query = completion.choices[0].message.content
search_response = metaphor.search(
    USER_QUESTION, use_autoprompt=True
)
print(search_response)
print(f"URLs: {[(result.title, result.url) for result in search_response.results]}\n")
