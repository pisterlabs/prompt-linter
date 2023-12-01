import os
import openai
import datetime

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_file_tree(directory, exclude_dir):
    file_tree = {}

    for root, dirs, files in os.walk(directory):
        # Exclude the script's directory
        if os.path.commonpath([root, exclude_dir]) == exclude_dir:
            continue

        for name in files:
            file_path = os.path.join(root, name)
            relative_path = os.path.relpath(file_path, directory)
            file_tree[relative_path] = 'file'

    return file_tree

def send_to_openai(query):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "user",
                    "content": "Analyze and structure the following content concisely: " + query
                }
            ]
        )
        return response.choices[0].message['content']
    except Exception as e:
        print(f"Error with OpenAI: {e}")
        return str(e)

def get_code_structure(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            code = file.read()
        return send_to_openai(code)
    except UnicodeDecodeError:
        error_message = f"Error reading {filepath} due to encoding issues."
        print(error_message)
        return error_message

def main():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    file_tree = get_file_tree(os.getcwd(), script_directory)  # Changed to scan from the current working directory

    important_files = [file for file in file_tree.keys() if file.endswith('.py')]

    print(f"Files to Analyze: {important_files}")  # Debugging

    project_structure = {}
    for file in important_files:
        print(f"Analyzing: {file}...")
        project_structure[file] = get_code_structure(file)

    # Create output folder with timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_folder = os.path.join(script_directory, f'architecture_output_{timestamp}')
    os.makedirs(output_folder)

    with open(os.path.join(output_folder, 'project_structure.txt'), 'w') as out_file:
        for file, structure in project_structure.items():
            out_file.write(f"{file}:\n{structure}\n\n")

    print("Analysis complete! Check the architecture output folder for results.")

if __name__ == '__main__':
    main()
