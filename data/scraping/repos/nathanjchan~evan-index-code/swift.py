import os
import re
import openai
import argparse
from datetime import datetime

# Read the API key from api_key.txt
with open("api_key.txt", "r") as key_file:
    openai.api_key = key_file.readline().strip()

# Constants
MODEL_NAME = "gpt-3.5-turbo"

# Create a timestamp-based filename
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_filename = f"results_{timestamp}.txt"

def print_and_log(message):
    """Prints the message to the console and appends it to the output file."""
    with open(output_filename, 'a') as f:
        f.write(message + "\n")
    print(message)

def find_swift_files_in_directory(directory):
    print(f"Searching for .swift files in {directory}...")
    swift_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".swift"):
                swift_files.append(os.path.join(root, file))
    print(f"Found {len(swift_files)} .swift files.")
    return swift_files

def extract_objects_from_file(file_path):
    print(f"Extracting object definitions from {file_path}...")
    with open(file_path, 'r') as file:
        content = file.read()
    objects = re.findall(r'\b(class|struct|enum|protocol)\s+(\w+)', content)
    print(f"Extracted {len(objects)} object(s) from {file_path}.")
    return [obj[1] for obj in objects]

def ask_gpt_dependencies(defined_object, code_content):
    print_and_log(f"Asking {MODEL_NAME} about dependencies of {defined_object}...")
    message_input = [
        {"role": "system", "content": "You can only respond with a Python list of strings."},
        {"role": "user", "content": f"In the following Swift code, identify which objects the {defined_object} is using. Please give the answer as a Python list with no formatting.\n {code_content}"}
    ]
    
    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=message_input
    )
    print_and_log(f"{MODEL_NAME} response: {response}")
    
    dependencies = set()
    if response and 'choices' in response and response['choices']:
        raw_response = response['choices'][0]['message']['content'].strip().strip('[').strip(']')
        # Split the response by commas and strip whitespace to get each object name
        matches = [item.strip().strip('\'').strip('\"') for item in raw_response.split(',')]
        dependencies.update(matches)
        print_and_log(f"{MODEL_NAME} identified {len(dependencies)} dependencies for {defined_object}.")
    return dependencies

def create_directed_graph(directory):
    print("Starting to create the directed graph...")
    swift_files = find_swift_files_in_directory(directory)
    
    # Get all objects from all files
    all_objects = set()
    for file in swift_files:
        all_objects.update(extract_objects_from_file(file))
    print(f"Found {len(all_objects)} objects in {len(swift_files)} files.")
    
    graph = {}
    for file in swift_files:
        objects_in_file = extract_objects_from_file(file)
        with open(file, 'r') as f:
            code_content = f.read()
        for obj in objects_in_file:
            dependencies = ask_gpt_dependencies(obj, code_content)
            if obj not in graph:
                graph[obj] = set()
            # Only add dependencies that are also in the list of objects
            dependencies = [dep for dep in dependencies if dep in all_objects]
            graph[obj].update(dependencies)
    return graph

def main(directory):
    graph = create_directed_graph(directory)
    print_and_log("\nDirected Graph (Object Definitions -> Usages):")
    print_and_log("="*60)
    for origin, endpoints in graph.items():
        for endpoint in endpoints:
            print_and_log(f"{origin} -> {endpoint}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a directed graph of object definitions and usages from Swift files.")
    parser.add_argument("directory", help="Path to the directory containing Swift files.")
    args = parser.parse_args()

    main(args.directory)
