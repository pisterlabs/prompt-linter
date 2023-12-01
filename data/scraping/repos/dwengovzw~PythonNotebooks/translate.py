import json
from openai import OpenAI
import os
import re

def translate_text(text, target_language):
    # Return original text if it only includes html tags, punctuation or whitespace
    if not containsText(text):
        print(f"'{text}' does not contain translatable text.")
        return text
  
    # Set up your OpenAI API key
    client = OpenAI()
    # Call the OpenAI API for translation
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"Translate the following text from dutch to {target_language}, keep html and markdown syntax and only return the translation. If no {target_language} text is found in the input, return the input. Keep Python syntax but translate comments.\n{text}"}
        ])
    
    translated_text = response.choices[0].message.content
    print(f"Translated '{text}' to '{translated_text}'")
    
    return translated_text

# Use a regex to filter out text that only contains html tags, whitespace or punctuation.
def containsText(text):
    return not re.match(r'^((<[^>]*>)*[\s]*[\.,;\'"!?]*)+$', text)


def translate_json(json_data, target_language):
    for cell in json_data['cells']:
        if 'source' in cell:
            # Translate each line in the 'source' list
            translated_lines = [translate_text(line, target_language) for line in cell['source']]
            cell['source'] = translated_lines

    return json_data

def ask_settings():
    # Ask the user for the target language
    target_language = input('What language would you like to translate to? ').lower()
    source_directory = os.path.normpath(input('What is the name of the source directory?\n(All .ipynb files in this directory will be translated.)\n'))
    target_directory = os.path.normpath(input('What is the name of the target directory?\n(All .ipynb files will be saved to this directory.)\n'))
    recursive = input('Would you like to translate all subdirectories as well? (y/n) ')
    if recursive.lower() == 'y' or recursive.lower() == 'yes':
        recursive = True
    else:
        recursive = False
    
    return {
        'target_language': target_language,
        'source_directory': source_directory,
        'target_directory': target_directory,
        'recursive': recursive
        }
    
    
def translate_file(file_source_path, file_destination_path, target_language):
    with open(file_source_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
        translated_json_data = translate_json(json_data, target_language)
        os.makedirs(os.path.dirname(file_destination_path), exist_ok=True)
        with open(file_destination_path, 'w', encoding='utf-8') as output_file:
            json.dump(translated_json_data, output_file, indent=2)
            
            
def iterate_ipynb_files(source_directory, recursive):
    ipynb_files = []
    if recursive:
        for root, dirs, files in os.walk(source_directory):
            for file in files:
                if file.endswith(".ipynb"):
                    ipynb_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(source_directory):
            if file.endswith(".ipynb"):
                ipynb_files.append(os.path.join(source_directory, file))
    
    return ipynb_files

def main():
    
    settings = ask_settings()
    ipynb_files = iterate_ipynb_files(settings['source_directory'], settings['recursive'])
    for file in ipynb_files:
        print(file)
        translate_file(file, file.replace(settings['source_directory'], settings['target_directory']), settings['target_language'])
        
if __name__ == "__main__":
    main()