import os
import openai

openai.api_key = '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'


def generate_code(prompt, existing_code):
    response = openai.Completion.create(
        engine="gpt-3.5-turbo",
        prompt=f'{existing_code}\n# {prompt}',
        temperature=0.5,
        max_tokens=8000
    )

    return response.choices[0].text.strip()


def select_python_file(directory):
    print("Available Python files:")
    files = [f for f in os.listdir(directory) if f.endswith('.py')]
    for i, file in enumerate(files):
        print(f"{i + 1}. {file}")

    selection = int(input("Select the number of the Python file: ")) - 1
    return files[selection]


directory = input("Enter the directory path: ")

while True:
    prompt = input('Enter a description of the code you want to generate: ')
    if prompt.lower() == 'quit':
        break

    selected_file = select_python_file(directory)
    file_path = os.path.join(directory, selected_file)
    with open(file_path, 'r') as file:
        existing_code = file.read()

    modified_code = generate_code(prompt, existing_code)
    print(f'Modified Code: \n{modified_code}')
