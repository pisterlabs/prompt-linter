from dotenv import load_dotenv
from langchain.llms import OpenAI


def generate_character_names(genre, setting):
    """
    Generate names for fictional characters in a role-playing game.
    
    :param genre: The genre of the role-playing game.
    :param setting: The setting of the role-playing game.
    :return: AI-generated character names.
    """

    # Define the prompt template for creating character names
    template = """
    As an AI storytelling assistant, I am creating fictional characters in a {genre} role-playing game set in {setting}.
    Here are some character names and short backstories:
    """

    # Input data for the prompt
    input_data = {"genre": genre, "setting": setting}

    # Format the prompt by substituting the variables in the template
    prompt_text = template.format(**input_data)

    # Initialize the language model
    llm = OpenAI(model_name="text-davinci-003", temperature=0.7)

    # Generate character names using the language model
    response = llm.predict(prompt_text)

    # Return the AI-generated character names
    return response


# Load environment variables
load_dotenv()

# Define the genre and setting of the role-playing game
genre = "techno-fantasy"
setting = "The Web3 Continuum Conference at the Budja Lounge during the Bitcoin conference in Miami, 2031"

# Generate and print character names
character_names = generate_character_names(genre, setting)
print(character_names)
