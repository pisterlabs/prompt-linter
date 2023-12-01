import asyncio
import os
import openai
from bs4 import BeautifulSoup
from doctran import Doctran
from dotenv import load_dotenv
from tqdm import tqdm
import re

load_dotenv()

env_vars = ['OPENAI_API_KEY', ]
os.environ.update({key: os.getenv(key) for key in env_vars})
openai.api_key = os.getenv('OPENAI_API_KEY')

doctran = Doctran(openai_api_key=openai.api_key)


def add_space_after_tags(html_doc):
    html_doc = html_doc.replace("</strong>", "</strong> ")
    html_doc = html_doc.replace("</a>", "</a> ")
    html_doc = html_doc.replace("</b>", "</b> ")
    return html_doc


def translate_html_content(html_content):
    soup = BeautifulSoup(html_content, features="html.parser")

    async def translate_text(text_element):
        # Check if the next sibling of the text element is an HTML tag
        if text_element.next_sibling and text_element.next_sibling.name in ["a", "b", "strong"]:
            # If it is, add a space at the end of the translated text
            document = doctran.parse(content=text_element)
            translated = await document.translate(language="french").execute()
            return translated.transformed_content + " "
        else:
            document = doctran.parse(content=text_element)
            translated = await document.translate(language="french").execute()
            return translated.transformed_content
        
    for text_element in tqdm(soup.body.find_all(string=lambda text: text.strip()), desc="Translating", ncols=100):
        new_text = asyncio.run(translate_text(text_element))
        text_element.replace_with(new_text)

    html_doc = str(soup.body)
    
    html_doc = add_space_after_tags(html_doc)
    
    # Remove phrases
    html_doc = re.sub(re.compile("Bonjour, comment ça va?", re.IGNORECASE), "", html_doc)
    html_doc = re.sub(re.compile("Bonjour, comment puis-je vous aider aujourd'hui?", re.IGNORECASE), "", html_doc)
    html_doc = re.sub(re.compile("Ceci est un document traduit en français.", re.IGNORECASE), "", html_doc)
    
    # Replace "Mon Livre" with "Ledger"
    html_doc = re.sub(re.compile("Mon Livre", re.IGNORECASE), "Ledger", html_doc)
    html_doc = re.sub(re.compile("pièces", re.IGNORECASE), "cryptos", html_doc)
    html_doc = re.sub(re.compile("pièce", re.IGNORECASE), "crypto", html_doc)
    
    print(html_doc)
    return html_doc
    

article_ids_to_translate = ['13791830911389']  # Replace with your actual numeric article IDs
#locales = ['en-us']

def translate_html_files(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in tqdm(os.listdir(input_dir), desc="Processing files", ncols=100):
        match = re.search(r'zd_(\d+)_en-us\.html', filename)
        if match and match.group(1):  # Check if a non-empty ID is extracted
            file_id = match.group(1)
            if file_id in article_ids_to_translate:
                input_file = os.path.join(input_dir, filename)
                output_file = os.path.join(output_dir, filename)

                with open(input_file, "r", encoding="utf-8") as f:
                    html_content = f.read()

                translated_html_content = translate_html_content(html_content)

                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(translated_html_content)

translate_html_files("./input_files/articles", "./translated_files")
