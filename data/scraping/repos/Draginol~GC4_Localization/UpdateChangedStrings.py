import csv
import os
import xml.etree.ElementTree as ET
import tkinter as tk
from tkinter import filedialog
import openai

openai_api_key = os.getenv('OPENAI_API_KEY')

import os

openai_api_key = os.getenv('OPENAI_API_KEY')

if openai_api_key is None:
    print("OPENAI_API_KEY is not set.")
else:
    print("OpenAI API Key found.")
    # You can now use openai_api_key in your OpenAI API calls


def select_file():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])

def translate_text(label_name, text, target_language):
    prompt = f"In the context of a Sci-Fi video game, given the string table entry label '{label_name}' as context, translate the following text into {target_language}. Respect all formatting codes and do not include the label. Add spaces without breaking meaning if a phrase is long to ensure word wrapping is not broken. Text to translate: {text}"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            n=1,
            stop=None,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in getting translation feedback: {e}")
        return None

def update_xml_files(csv_file_path, root_directory):
    languages = ["Chinese", "French", "German", "Greek", "Italian", "Japanese", "Korean", "Polish", "Portuguese", "Russian", "Spanish"]

    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            label_to_find = row['Label']
            original_text = row['New Text']

            for language in languages:
                xml_directory = os.path.join(root_directory, language, 'Text')

                if not os.path.isdir(xml_directory):
                    print(f"Directory not found: {xml_directory}")
                    continue

                for xml_file in os.listdir(xml_directory):
                    if xml_file.endswith('.xml'):
                        xml_file_path = os.path.join(xml_directory, xml_file)
                        tree = ET.parse(xml_file_path)
                        root = tree.getroot()

                        for string_table in root.findall('.//StringTable'):
                            if string_table.find('Label').text == label_to_find:
                                translated_text = translate_text(label_to_find, original_text, language)
                                if translated_text:
                                    string_table.find('String').text = translated_text
                                    tree.write(xml_file_path, encoding='utf-8', xml_declaration=True)
                                    print(f"Updated {label_to_find} in {language}/{xml_file}")
                                else:
                                    print(f"Failed to translate for {label_to_find} in {language}")
                                break

csv_file = select_file()
root_dir = os.path.dirname(csv_file)
update_xml_files(csv_file, root_dir)


