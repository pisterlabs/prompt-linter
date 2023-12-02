import json
import time
import os
import openai

# Set up your GPT-3.5 Turbo API key
openai.api_key = "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"

def summarize(content):
    conversation = [
        {"role": "system", "content": "You are a helpful assistant that summarizes text succintly and never prefaces a summarization."},
        {"role": "user", "content": f"Please summarize the following text. Only summarize the text and do not preface your summarization in any way. For example you never say: The text is about... and only summarize the content.\n\n{content[:3950]}"}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation,
        temperature=0.4,
        max_tokens=1000
    )

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
    # Handle potential API errors with a retry loop
    for _ in range(5):  # Try up to 5 times
        try:
            print(f"Sending record {record['url']} for summarization...")
            summary = summarize(content)
            print(f"Received summary for record {record['url']}.")
            record['content'] = summary
            return record  # return the updated record
        except Exception as e:
            print(f"Error processing record {record['url']}: {e}")
            time.sleep(5)  # Wait for 5 seconds before retrying
    return None  # Failed after 5 attempts

def process_file(filename, processed_records_file, output_filename='summarised.json'):
    # Load data
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed_records = load_processed_records(processed_records_file)

    # Check if output file exists, if not, create an empty one
    if not os.path.exists(output_filename):
        print(f"Output file '{output_filename}' not found, creating a new one...")
        with open(output_filename, 'w') as f:
            json.dump([], f)

    # Process data
    for record in data:
        if record['url'] in processed_records:
            print(f"Skipping already processed record: {record['url']}")
            continue

        processed_record = process_data(record)
        if processed_record is not None:
            save_processed_record(processed_records_file, processed_record['url'])
            # Load the existing data from output file
            with open(output_filename, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            # Append the new processed data
            existing_data.append(processed_record)
            # Write the updated data back to the output file
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=4)
            print(f"Updated the output file with record {record['url']}")
        else:
            print(f"Failed to process record: {record['url']}. Continuing with next record.")

def main(input_filename='scraped_pages.json', processed_records_filename='processed_records.txt', output_filename='summarised.json'):
    process_file(input_filename, processed_records_filename, output_filename)

if __name__ == "__main__":
    main()