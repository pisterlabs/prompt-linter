import json
import os
import time
import random
import openai
import sys
from time import sleep

# Set up your GPT-3.5 Turbo API key
openai.api_key = "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"

project_root = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(project_root))

unwanted_phrases = [
    "write a response for an instruction",
    "FedEx",
    "Here are some best practices",
    "LOADING",
    "loading",
    "error",
    "403",
    "forbidden",
    "Twitter",
    "Forbidden",
    "Page Not Found",
    "Google Colab",
    "Just a moment",
    "We’ve detected that JavaScript is disabled",
    "LinkedIn",
    "The provided instruction",
    "The provided text",
    "Just a moment..."
]

def contains_unwanted_phrases(text):
    return any(phrase.lower() in text.lower() for phrase in unwanted_phrases)


def identify_keywords(content):
    conversation = [
        {"role": "system", "content": "You are a helpful assistant that identifies keywords in a text."},
        {"role": "user", "content": f"What are the three main keywords in the following text?\n\n{content[:2000]}\nPlease format your response as: 'Keywords: keyword1, keyword2, keyword3...' where keywords are only one word long never more than one word." }
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation,
        temperature=0.4,
        max_tokens=200
    )

    # Add a delay between 250ms and 400ms
    time.sleep(random.uniform(0.25, 0.4))

    return response.choices[0].message.content.strip()



def load_processed_records(filename):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f)
    else:
        return set()

def save_processed_record(filename, record_id):
    with open(filename, "a", encoding="utf-8") as f:
        f.write(record_id + "\n")

def process_data(record):
    content = record['content']
    website_name = record['website_name']
    path_info = record['path_info']
    combined_content = f"{website_name} {path_info} {content}"
    
    # Check if the record contains unwanted phrases and skip if it does
    if contains_unwanted_phrases(combined_content):
        print(f"Skipping record with unwanted phrases: {record['url']}")
        return None

    for attempt in range(1, 6):
        try:
            print(f"Extracting keywords for record {record['url']} (Attempt: {attempt})...")
            keywords_str = identify_keywords(combined_content)

            keywords = []
            if 'keywords' in keywords_str.lower():
                keywords = [v.strip() for v in keywords_str.split(':')[1].split(',')]

            print(f"Received keywords for record {record['url']}.")
            record['keywords'] = keywords
            return record
            
        except Exception as e:
            print(f"Error processing record {record['url']} (Attempt {attempt}): {e}")
            time.sleep(random.uniform(2, 4)) # Wait randomly between 2 and 4 seconds before retrying
                
    return None


def process_file(filename, processed_records_file, output_filename='keywords.json'):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed_records = load_processed_records(processed_records_file)

    if not os.path.exists(output_filename):
        print(f"Output file '{output_filename}' not found, creating a new one...")
        with open(output_filename, 'w') as f:
            json.dump([], f)

    for record in data:
        if record['url'] in processed_records:
            print(f"Skipping already processed record: {record['url']}")
            continue

        processed_record = process_data(record)
        if processed_record is not None:
            save_processed_record(processed_records_file, processed_record['url'])
            with open(output_filename, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            existing_data.append(processed_record)
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=4)
            print(f"Updated the output file with record {record['url']}")
        else:
            print(f"Failed to process record: {record['url']}. Continuing with next record.")

def main():
    process_file('summarised.json', 'keywords_processed.txt')

if __name__ == "__main__":
    main()