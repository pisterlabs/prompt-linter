import time
import configparser
import os
import openai
import json
import csv

__OUTPUT_DIR = 'output_pdf'
__CONTENT_PROMPT_LENGTH = 4000
MAX_RETRIES = 5  # Maximum number of retries for each API call
RETRY_DELAY = 10  # Delay in seconds before retrying

# Read the CSV file and extract the topic-question pairs
def read_csv_pairs(filename):
    with open(filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        pairs = [(row['topic'], row['question']) for row in reader]
    return pairs

# Read the last processed state
def read_last_processed_state():
    if os.path.exists('last_processed_state.txt'):
        with open('last_processed_state.txt', 'r') as f:
            return f.read().strip()
    return None

# Save the last processed state
def save_last_processed_state(state):
    with open('last_processed_state.txt', 'w') as f:
        f.write(state)

# Updated prompt template
prompt_template = """
Op basis van de volgende inhoud uit het document "{pdf_name}", identificeer alle secties gerelateerd aan {topic}.
Voor elke sectie, geef een citaat met de relevante tekst en paginanummer.
Evalueer daarnaast of {question}.

De inhoud die geëvalueerd moet worden is:
"""

config_parser = configparser.ConfigParser()
config_parser.read('openaiconfig.ini')
openai_secret = config_parser['keys']['openaikey']
openai.api_key = openai_secret

with open('docs/extracted_texts.json', 'r', encoding='utf-8') as json_file:
    pdf_texts = json.load(json_file)

if not os.path.exists(__OUTPUT_DIR):
    os.makedirs(__OUTPUT_DIR)

topic_question_pairs = read_csv_pairs('query.csv')
last_processed_state = read_last_processed_state()

for pdf_name, data in pdf_texts.items():
    for topic, question in topic_question_pairs:
        state = f"{pdf_name}-{topic}"
        if state == last_processed_state:
            continue  # Skip already processed combinations

        print(pdf_name)
        base_file_name = os.path.splitext(pdf_name)[0]
        text = ""
        for page_data in data:
            text += f"Page {page_data['page_number']}:\n{page_data['text']}\n\n"

        text_parts = [text[i * __CONTENT_PROMPT_LENGTH:(i + 1) * __CONTENT_PROMPT_LENGTH] for i in range((len(text) + __CONTENT_PROMPT_LENGTH - 1) // __CONTENT_PROMPT_LENGTH)]

        for idx, part in enumerate(text_parts):
            outfile = os.path.join(__OUTPUT_DIR, f"{base_file_name}-{topic}-{idx:04d}.txt")
            if not os.path.exists(outfile):
                for attempt in range(MAX_RETRIES):
                    try:
                        print('calling OpenAI')
                        prompt = prompt_template.format(pdf_name=pdf_name, topic=topic, question=question)
                        response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo-16k",
                            messages=[
                                {
                                    "role": "user",
                                    "content": prompt + part
                                }
                            ],
                            temperature=0.5,
                            max_tokens=12000,
                            top_p=1,
                            frequency_penalty=0,
                            presence_penalty=0
                        )

                        print('writing files')
                        with open(outfile, 'w', encoding='utf-8') as f:
                            result = response['choices'][0]['message']['content']
                            f.write(result)
                        break  # If the API call is successful, break out of the retry loop
                    except openai.error.RateLimitError as e:
                        print(f"Rate limit error: {e}. Waiting for a few seconds before retrying...")
                        time.sleep(RETRY_DELAY)
                    except openai.error.Timeout:
                        print("Request timed out. Waiting for a few seconds before retrying...")
                        time.sleep(RETRY_DELAY)
            else:
                print(f"Output file {outfile} exists, skipping")

        # Save the state after processing each topic for the current document
        save_last_processed_state(state)
        print('\n')
