import requests
import re
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter

def split_text_by_punctuation(text, max_chars=280):
    try:
        split_text = []
        def closest_split(s, ch):
            index = max_chars
            while index > 0:
                if s[index] == ch:
                    return index
                index -= 1
            return -1
        def split_chunk(chunk, split_char):
            while len(chunk) > max_chars:
                split_index = closest_split(chunk, split_char)
                if split_index == -1: 
                    break
                split_text.append(chunk[:split_index+1].strip())
                chunk = chunk[split_index+1:].strip()
            return chunk
        # Step 1: Split by double new lines
        chunks = text.split('\n\n')
        for chunk in chunks:
            if len(chunk) > max_chars:
                # Step 2: Split by new lines if chunk is still too long
                chunk = split_chunk(chunk, '\n')
            if len(chunk) > max_chars:
                # Step 3: Split by full stops if line is still too long
                chunk = split_chunk(chunk, '.')
            if len(chunk) > max_chars:
                # Step 4: Split by spaces if sentence is still too long
                words = chunk.split(' ')
                new_sentence = ''
                for word in words:
                    if len(new_sentence + ' ' + word) <= max_chars:
                        new_sentence += ' ' + word
                    else:
                        split_text.append(new_sentence.strip())
                        new_sentence = word
                if new_sentence:
                    split_text.append(new_sentence.strip())
            else:
                split_text.append(chunk)
        return split_text
    except Exception as e:
        print(e)
        return [text]

def get_helius_links():
    solana_links = []
    r = requests.get('https://docs.solana.com/getstarted/')
    soup = BeautifulSoup(r.text, 'html.parser')
    # Find all 'a' tags within divs with class 'css-175oi2r'
    links = soup.findAll('a', class_='menu__link')
    for link in links:
        href = link.get('href')
        if href and href.startswith('/'):
            solana_links.append(href)
    # Remove duplicates and sort
    helius_links = sorted(list(set(solana_links)))
    return helius_links



def get_solana_info(extension):
    url = 'https://docs.solana.com' + extension
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    # Get the div with the data-testid
    contents = soup.find('div', class_='theme-doc-markdown markdown')
    # Try to get the text
    try:
        contents = re.sub('(?<=[.!?])(?=[^\s])','\n', contents.text)
        contents = split_text_by_punctuation(contents, max_chars=1500)
        for text in contents:
            print(text)
            print()
    except Exception as e:
        print('Error: '+ str(e))
        contents = ''
    return contents

links = get_helius_links()


final_json = []

for link in links:
    for content in get_solana_info(link):
        base_dict = {
            'link':     'https://docs.solana.com' + link, 
            'title':    link.split('/')[-1], 
            'content':  content, 
            'type':     'content'
            } 
        final_json.append(base_dict)



# # Save the final_json to a json file
import json
with open('src/data_ingestion/solana/solana-docs.json', 'w') as f:
    json.dump(final_json, f, indent=4)



