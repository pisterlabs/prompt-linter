import json
import logging
import os
import re
from pprint import pprint

import openai

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set API key
openai.api_key = os.getenv("OPENAI_API_KEY")

ChatCompletionConfig = {
    "model": "gpt-4",
    "temperature": 0.5,
    "stop": "</code>"
}

# Function to extract and validate the output JSON


def extract_events_from_json(input_json_string):

    logging.info('Starting extraction for input data...')

    # Construct the initial prompt for the API
    prompt = (f"You are a service that translates a JSON into another JSON based on historical events. "
              f"Given the JSON below, extract significant events from the \"history_text_cleaned\" "
              f"and \"short_description_text_cleaned\" fields.\n\n"
              f"Describe the events with short concise sentences in a narrative manner.\n\n"
              f"<code>"
              f"{input_json_string}"
              f"</code>\n\n"
              "Please generate the expected output as a valid JSON format and place it inside a code block.\n\n"
              "Expected Output Format:\n\n"
              "<code>\n"
              "{\n"
              "  \"id\": \"Extracted from the input JSON.\",\n"
              "  \"events\": [\n"
              "    {\n"
              "      \"description\": \"A brief summary of the event.\",\n"
              "      \"shortDescription\": \"A summary of the event using strictly no more than 3 words.\",\n"
              "      \"time\": \"Strict ISO 8601 date format 'YYYY-MM-DD' or a duration object with 'from' and 'to' dates, both strictly in 'YYYY-MM-DD' format. No other characters or suffixes allowed.\",\n"
              "      \"approximateDate\": \"true if the date is approximate, false otherwise.\"\n"
              "    }\n"
              "  ]\n"
              "}\n"
              "</code>\n\n"
              "Please make sure your response contains a valid JSON structure wrapped between in a html code block <code>...</code>.\n\n"
              "Return only code block, no text.\n\n")

    messages = [
        {"role": "system", "content": "You are a specialist in converting and extracting information from JSON based on historical events."},
        {"role": "user", "content": prompt}
    ]

    # Fetch the API
    response = openai.ChatCompletion.create(
        model=ChatCompletionConfig['model'],
        temperature=ChatCompletionConfig['temperature'],
        stop=ChatCompletionConfig['stop'],
        messages=messages
    )

    # Extract and validate the response JSON
    output_text = response.choices[0].message['content'].strip()

    # Add stop token to the end of the output text
    output_text += ChatCompletionConfig['stop']

    pprint(output_text)
    messages.append({"role": "assistant", "content": output_text})

    match = re.search(r'<code>([\s\S]+?)</code>', output_text)
    # Extract JSON inside code block using regex
    if not match:
        logging.error('No JSON found inside code block.')
        return refetch_api_with_error_message(messages, "No JSON found inside code block.", input_json_string)

    json_str_in_code_block = match.group(1)

    try:
        output_json = json.loads(json_str_in_code_block)
        logging.info('JSON extraction successful.')
        return output_json
    except json.JSONDecodeError as e:
        logging.error(f'Error decoding JSON: {e}')
        return refetch_api_with_error_message(messages, str(e), input_json_string)
# Function to refetch the API on error


def refetch_api_with_error_message(messages, error_message, input_json_string):

    logging.info('Refetching with error message...')

    retry_prompt = (f"The JSON structure returned was incorrect because of the following reason: {error_message}. "
                    f"Given the original data, can you generate the expected JSON format as specified earlier?\n\n"
                    f"<code>"
                    f"{input_json_string}"
                    f"</code>\n\n"
                    "Expected Output Format:\n\n"
                    "<code>\n"
                    "{\n"
                    "  \"id\": \"Extracted from the input JSON.\",\n"
                    "  \"events\": [\n"
                    "    {\n"
                    "      \"description\": \"A brief summary of the event.\",\n"
                    "      \"shortDescription\": \"A summary of the event using strictly no more than 3 words.\",\n"
                    "      \"time\": \"Strict ISO 8601 date format 'YYYY-MM-DD' or a duration object with 'from' and 'to' dates, both strictly in 'YYYY-MM-DD' format. No other characters or suffixes allowed.\",\n"
                    "      \"approximateDate\": \"true if the date is approximate, false otherwise.\"\n"
                    "    }\n"
                    "  ]\n"
                    "}\n"
                    "</code>\n\n"
                    "Please make sure your response contains a valid JSON structure wrapped between in a html code block <code>...</code>.\n\n"
                    "Return only code block, no text.\n\n")

    messages.append({"role": "user", "content": retry_prompt})
    pprint(messages)

    response = openai.ChatCompletion.create(
        model=ChatCompletionConfig['model'],
        temperature=ChatCompletionConfig['temperature'],
        stop=ChatCompletionConfig['stop'],
        messages=messages
    )

    output_text = response.choices[0].message['content'].strip()

    # add stop token to the end of the output text
    output_text += ChatCompletionConfig['stop']

    pprint(output_text)

    # Extract JSON inside code block using regex, if present
    match = re.search(r'<code>([\s\S]+?)</code>', output_text)
    json_to_parse = match.group(1) if match else output_text

    try:
        output_json = json.loads(json_to_parse)
        logging.info('JSON extraction successful in refetch.')
        return output_json
    except json.JSONDecodeError as e:
        logging.error(f'Error decoding JSON on refetch: {e}')
        return None


# Read the input JSON file
with open('output/filtered_dump_2.json', 'r') as file:
    data = json.load(file)

# Process each item in the input JSON
for item in data:
    # Remove the "main_coords_parsed" entry before constructing the prompt
    if 'main_coords_parsed' in item:
        del item['main_coords_parsed']

    output_file_path = f"generated_output/{item['id']}.json"

    # Check if the file already exists
    if os.path.exists(output_file_path):
        logging.warning(
            f"A file with ID {item['id']} already exists. Skipping...")
        continue

    # Extract events from the JSON and save the response to a separate JSON file
    output_data = extract_events_from_json(json.dumps(item))

    if output_data:
        with open(output_file_path, 'w') as outfile:
            json.dump(output_data, outfile, indent=4)

logging.info('Script execution completed.')
