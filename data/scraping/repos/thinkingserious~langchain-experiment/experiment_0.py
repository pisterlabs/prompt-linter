from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import BaseOutputParser
from utils import CommaSeparatedListOutputParser, print_separator
from dotenv import load_dotenv
import os

# Load the environment variables from the .env file
load_dotenv()

# Print a separator to indicate the start of the translator experiment
print_separator("STARTING EXPERIMENT - TRANSLATOR")

# Define a template for a chat prompt that translates input_language to output_language
template = "You are a helpful assistant that translates {input_language} to {output_language}."
# Create a system message prompt from the template
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
# Define a template for a human message prompt
human_template = "{text}"
# Create a human message prompt from the template
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
# Create a chat prompt from the system and human message prompts
chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt])
# Create a ChatOpenAI object to handle the translation
chat_model = ChatOpenAI()
# Test the chat model by translating "I love programming" from English to French, Mandarin, and Spanish
print(chat_model.predict_messages(chat_prompt.format_messages(input_language="English",
      output_language="French", text="I love programming.")).content)
print(chat_model.predict_messages(chat_prompt.format_messages(input_language="English",
      output_language="Mandarin", text="I love programming.")).content)
print(chat_model.predict_messages(chat_prompt.format_messages(input_language="Mandarin",
      output_language="Spanish", text="我喜欢编程。 (Wǒ xǐhuān biānchéng.)")).content)

# Print a separator to indicate the start of the list generator experiment
print_separator("STARTING EXPERIMENT - LIST GENERATOR")

# Define a template for a chat prompt that generates comma-separated lists
template = """You are a helpful assistant who generates comma separated lists.
A user will pass in a category, and you should generate 5 objects in that category in a comma separated list.
ONLY return a comma separated list, and nothing more."""
# Create a system message prompt from the template
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
# Define a template for a human message prompt
human_template = "{text}"
# Create a human message prompt from the template
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
# Create a chat prompt from the system and human message prompts
chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt])
# Create an LLMChain object that uses the ChatOpenAI model, the chat prompt, and the CommaSeparatedListOutputParser object
chain = LLMChain(
    llm=ChatOpenAI(),
    prompt=chat_prompt,
    output_parser=CommaSeparatedListOutputParser()
)
# Define the categories
categories = [
    "colors",
    "books",
    "movies",
    "tv shows",
    "programming languages",
    "code editors",
    "APIs"
]
# Generate a list of items for each category using the LLMChain object
items = [chain.run(category) for category in categories]
# Print the items for each category
for category, items_list in zip(categories, items):
    if category == "TV shows":
        category = "TV Shows"
    elif category == "APIs":
        category = "APIs"
    else:
        category = category.title()
    print(f"{category}: {items_list}")

# Output

"""
# >> ['hi', 'bye']
# >> =========================================
# >> J'adore la programmation.
# >> 我喜欢编程。 (Wǒ xǐhuān biānchéng.)
# >> Me gusta programar.
# >> =========================================
# >> Colors: ['red', 'blue', 'green', 'yellow', 'orange']
# >> Books: ['The Catcher in the Rye', 'To Kill a Mockingbird', '1984', 'The Great Gatsby', 'Pride and Prejudice']
# >> Movies: ['The Godfather', 'The Shawshank Redemption', 'Pulp Fiction', 'The Dark Knight', 'Fight Club']
# >> TV Shows: ['Friends', 'Game of Thrones', 'Breaking Bad', 'The Office', 'Stranger Things']
# >> Programming Languages: ['Python', 'Java', 'C++', 'JavaScript', 'Ruby']
# >> Code Editors: ['Sublime Text', 'Visual Studio Code', 'Atom', 'Notepad++', 'Brackets']
# >> APIs: ['Twilio', 'Stripe', 'Google Maps', 'OpenWeatherMap', 'Spotify']
"""
