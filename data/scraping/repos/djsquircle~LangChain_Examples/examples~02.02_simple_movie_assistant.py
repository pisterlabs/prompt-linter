# Import necessary libraries and modules
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# Load environment variables
load_dotenv()

# Initialize the language model
language_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=.95)

# Define the system message prompt template
system_template = (
    "You are an AI assistant specialized in creating fictional narratives "
    "based on movies. Use your knowledge of film to craft imaginative scenes."
)
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

# Define the human message prompt template
human_template = (
    "Find information about the movie {movie_title}, and write a short scene "
    "inspired by it. The scene should be completely new including characters, with a "
    "unique twist and a 'choose your own adventure' style choice at the end."
)
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

# Combine system and human message prompts
chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)

# Define the movie title to be used in the prompt
movie_title = "Blade Runner"

# Generate the response using the language model
response = language_model(
    chat_prompt.format_prompt(movie_title=movie_title).to_messages()
)

# Output the generated content
print(response.content)
