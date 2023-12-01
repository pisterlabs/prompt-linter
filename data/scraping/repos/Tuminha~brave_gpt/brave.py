import openai
from langchain.document_loaders import BraveSearchLoader
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")

query = "OpenAI"  # Or any other query

# Initialize the loader with your query and API key
loader = BraveSearchLoader(query=query, api_key=BRAVE_API_KEY, search_kwargs={"count": 3})

# Load the documents (perform the search)
docs = loader.load()

# Initialize an empty list to store the formatted information
info_list = []

# Iterate over each document
for doc in docs:
    title = doc.metadata['title']
    link = doc.metadata['link']
    snippet = doc.page_content
    # Format the information
    info = f"Title: {title}\nLink: {link}\nSnippet: {snippet}\n"
    # Add the formatted information to the list
    info_list.append(info)

# Combine all the formatted information into a single string
info_str = "\n".join(info_list)

# Initialize the OpenAI API
openai.api_key = OPENAI_API_KEY

# Construct the prompt for the OpenAI model
prompt = f"I found the following information about {query}:\n{info_str}\nBased on this information, can you tell me more about {query}?"

# Generate a response from the OpenAI model
response = openai.Completion.create(prompt=prompt, engine="text-davinci-003", max_tokens=100)

# Print the generated text
print(response['choices'][0]['text'])
