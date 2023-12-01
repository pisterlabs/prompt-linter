import configparser
import openai
from modules.errors import robust_api_call
from tqdm import tqdm

# Load the configuration file
config = configparser.ConfigParser()
config.read('modules/suite_config.ini')

# Access variables
use_tqdm = config.getboolean('General', 'UseTqdm')
model = config['Models']['CategorizeHeadlines']
categories = config['Headlines']['Categories']
openai_api_key = config['OPENAI']['OPENAI_API_KEY']

openai.api_key = openai_api_key
def categorize_headlines(headlines):
    categorized_headlines = []

    for headline in tqdm(headlines, desc="Categorizing headlines"):
        # Create the prompt
        prompt = f"Classify the following headline into one of these categories: {categories}. Only type the category name as the response. Headline: '{headline[0]}'"

        response = robust_api_call(lambda: openai.Completion.create(
            engine=model,
            prompt=prompt,
            temperature=0,
            max_tokens=60,
        ))

        if response is not None:
            category = response['choices'][0]['text'].strip()
            print(f"Model response: {category}") 
            categorized_headlines.append((headline[0], category, headline[1], headline[2], headline[3]))
        else:
            # If the API call failed after all retries, add a None category and continue with the next headline
            print("Failed to get category for a headline after all retries. Skipping...")
            categorized_headlines.append((headline[0], None, headline[1], headline[2], headline[3]))

    return categorized_headlines

