import json
import logging
import os
import backoff
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
from tqdm import tqdm
import concurrent.futures

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load API key from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
number_of_simulations = 1000
concurrent_simulations = 15

# Specify the folder to save the JSON files
output_folder = "../../data/simulations"
os.makedirs(output_folder, exist_ok=True)

# Define the path to the prompt file
system_prompt_file = "../../docs/prompts/prompt-simulation-lextranscripts-system.md"
with open(system_prompt_file, 'r') as file:
    system_prompt = file.read()
user_prompt_file = "../../docs/prompts/prompt-simulation-lextranscripts-user.md"
with open(user_prompt_file, 'r') as file:
    user_prompt = file.read()

# Backoff function for handling rate limits
@backoff.on_exception(backoff.expo, OpenAIError)
def completions_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)

def get_response():
    # Generate the conversation using OpenAI's chat completion API
    response = completions_with_backoff(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.7,
        timeout=30,
    )
    return response

# Function to handle each simulation
def handle_simulation(i):
    try:
        response = get_response()

        # Extract the generated message from the API response
        message = response.choices[0].message.content

        # Parse the message content as JSON
        message_json = json.loads(message)

        # Save the conversation as a JSON file
        filename = f"{output_folder}/simulation_002{i+1}.json"
        with open(filename, 'w') as file:
            json.dump(message_json, file, indent=4)  # Pretty print

        logging.info(f"Simulation {i+1} completed and saved as {filename}")

    except Exception as e:
        logging.error(f"Error in simulation {i+1}: {e}")

# Init the OpenAI client
try:
    client = OpenAI(api_key=api_key)
except Exception as e:
    logging.error(f"Failed to initialize OpenAI client: {e}")
    exit(1)

# Run the simulations concurrently
with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_simulations) as executor:
    list(tqdm(executor.map(handle_simulation, range(number_of_simulations)), total=number_of_simulations, desc="Simulating conversations"))
