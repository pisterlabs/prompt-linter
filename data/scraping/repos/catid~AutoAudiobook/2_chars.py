import os, argparse, json, shutil

# OpenAI
import api_key
from openai import OpenAI
client = OpenAI(api_key=api_key.api_key)

def extract_two_member_objects(json_obj, result=None):
    """
    Walks through a JSON object and extracts all objects that have exactly two members,
    with strings for both keys and values.

    :param json_obj: The JSON object to walk through.
    :param result: A list to store the extracted objects.
    :return: A list of extracted objects.
    """
    if result is None:
        result = []

    if isinstance(json_obj, dict):
        # Check if the object has exactly two string key-value pairs
        keys = list(json_obj.keys())
        if len(keys) == 2 and all(isinstance(k, str) and isinstance(v, str) for k, v in json_obj.items()):
            new_obj = {"name": json_obj[keys[0]], "dialog": json_obj[keys[1]]}
            result.append(new_obj)
        # Recursively walk through each value which is a dictionary or a list
        for value in json_obj.values():
            extract_two_member_objects(value, result)
    elif isinstance(json_obj, list):
        # Recursively walk through each item in the list
        for item in json_obj:
            extract_two_member_objects(item, result)

    return result


def assign_characters_split(args, content):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that produces JSON."},
        {"role": "user", "content": f"""
Read the following text and classify each part with 'narrator' or the name of the character speaking.
Provide the output in JSON format, where each piece of dialogue or narration is a separate entry with the speaker and text keys. Here's the text:
{content}"""}
    ]
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
        temperature=0.1,
        max_tokens=4096,
        n=1,
        response_format={"type": "json_object"}
    )

    content = response.choices[0].message.content.strip()

    raw_json = json.loads(content)

    sanitized_json = extract_two_member_objects(raw_json)

    return sanitized_json


def assign_characters(args):
    if os.path.exists(args.output_folder):
        shutil.rmtree(args.output_folder)
    os.makedirs(args.output_folder)

    for filename in os.listdir(args.split_folder):
        if filename.endswith('.txt'):  # Check if the file is a text file
            with open(os.path.join(args.split_folder, filename), 'r') as file:

                try:
                    content = file.read()

                    base = os.path.splitext(filename)[0]
                    output_filename = base + ".json"

                    json_object = assign_characters_split(args, content)

                    assigned_filename = os.path.join(args.output_folder, output_filename)
                    with open(assigned_filename, 'w', encoding='utf-8') as assigned_file:
                        assigned_file.write(json.dumps(json_object))

                    print(f"Successfully processed {filename} -> {output_filename}")

                except Exception as e:
                    print(f"Failed to process {filename}: {e}")


def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Assign character names to each part of the text.")
    parser.add_argument(
        '--split-folder', 
        type=str, 
        nargs='?',  # Indicates the argument is optional
        default='splits',  # Default file name
        help="Directory with input splits."
    )
    parser.add_argument(
        '--output-folder', 
        type=str, 
        nargs='?',  # Indicates the argument is optional
        default='assigned',  # Default folder name
        help="Characters assigned to each split."
    )

    # Parse the command line arguments
    args = parser.parse_args()

    assign_characters(args)

if __name__ == "__main__":
    main()
