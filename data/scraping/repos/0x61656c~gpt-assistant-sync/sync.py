import os
from openai import OpenAI
from datetime import datetime

###############################################
# Configuration
###############################################

# Enter the directory you want synced 
directory_path = os.path.dirname(os.path.realpath(__file__))

# Enter your OpenAI API key
api_key = "" 

# Enter the ID for your Assistant
assistant_id = ""

###############################################
###############################################
###############################################

client = OpenAI(api_key=api_key)

# Supported file formats
supported_formats = {'c', 'cpp', 'csv', 'docx', 'html', 'java', 'json', 'md', 
                     'pdf', 'php', 'pptx', 'py', 'rb', 'tex', 'txt', 'css', 
                     'jpeg', 'jpg', 'js', 'gif', 'png', 'tar', 'ts', 'xlsx', 
                     'xml', 'zip'}

def compile_files(directory_path, output_file):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        if os.path.isdir(file_path):
            # It's a directory, recursively compile its contents
            compile_files(file_path, output_file)
        else:
            file_extension = os.path.splitext(filename)[1][1:].lower()  # Extract file extension

            if file_extension in supported_formats:
                print("Compiling file: " + file_path)
                with open(file_path, 'r') as file:
                    try:
                        content = file.read()
                        output_file.write(f"\n---{filename}---\n")
                        output_file.write(content)
                    except Exception as e:
                        print(f"Error reading {filename}: {e}")

def upload_compiled_file(file_path):
    try:
        with open(file_path, 'rb') as file:
            # Upload the file to OpenAI
            response = client.files.create(
                file=file,
                purpose='assistants'
            )
            file_id = response.id

            # Attach the file to an assistant
            client.beta.assistants.files.create(
                assistant_id=assistant_id,
                file_id=file_id
            )

            print(f"Uploaded compiled file: {response}")
    except Exception as e:
        print(f"Error uploading compiled file: {e}")

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    compiled_file_path = os.path.join(directory_path, "sync-" + timestamp + ".txt")

    with open(compiled_file_path, 'w') as output_file:
        compile_files(directory_path, output_file)

    upload_compiled_file(compiled_file_path)
