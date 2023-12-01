import os
import re
import tiktoken
import openai
from bs4 import BeautifulSoup
enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
API_KEY = os.environ.get("OPENAI_API_KEY")
openai.api_key = API_KEY

def truncate_encoded_text(encoded_text, max_length=2000):
    return encoded_text[:max_length]

def read_html_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content


def parse_html(html_content):
    # Remove DOCTYPE declaration
    html_content = re.sub(r'<!DOCTYPE.*?>', '', html_content, flags=re.IGNORECASE)
    
    # Parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()

    # Replace <br> elements with an empty string
    for br in soup.find_all("br"):
        br.replace_with("")

    result = []

    # Extract the title text
    title_element = soup.find('h1', class_='COVER-TITLE')
    title_tag = soup.find('title') if not title_element else None

    if title_element:
        title = title_element.get_text().strip()
    elif title_tag:
        title = title_tag.get_text().strip()
    else:
        title = ""

    new_title = re.sub(r'[\s\d_::]+', '', title) if title else ""

    # Find the first table element and extract the text before it
    first_table = soup.find('table')
    # Extract the text before the first table using the 'next_siblings' property
    text_before_table = ""
    if first_table is not None:
        if first_table:
            for sibling in reversed(list(first_table.previous_siblings)):
                if sibling.name is None:
                    text_before_table += str(sibling)
                elif sibling.name == 'p':
                    text_before_table += sibling.get_text()

        # Extract and process table rows
        rows = soup.find_all('tr')
        result = [new_title, text_before_table]
        for row in rows:
            cell_texts = [cell.get_text().strip() for cell in row.find_all(['td', 'th'])]
            line = '\t'.join(cell_texts).strip()
            if line:
                result.append(line)
            
        # Join the processed lines
        parsed_text = '\n'.join(result)
    else :
        text = soup.get_text()
        return text

    return parsed_text

def remove_html_files(folder_path):
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if file.endswith('.html'):
            os.remove(file_path)
            print(f"Removed HTML file: {file_path}")    

def main():
    parent_folder = "train_dataset"
    for folder, subfolders, files in os.walk(parent_folder):
        for file in files:
            if file.endswith(".xls"):
                new_file_name = file.replace(".xls", ".html")
                os.rename(os.path.join(folder, file), os.path.join(folder, new_file_name))

    for item in os.listdir(parent_folder):
        item_path = os.path.join(parent_folder, item)
        if os.path.isdir(item_path):
            n = 0
            print(f"Processing folder: {item_path}")
            for file in os.listdir(item_path):
                if not file.endswith('.html'):
                    continue

                parsed_text = parse_html(read_html_file(file_path=os.path.join(item_path, file)))
                encoded_text = enc.encode(parsed_text)
                truncated_encoded_text = truncate_encoded_text(encoded_text)
                decoded_text = enc.decode(truncated_encoded_text)
                n += 1

                # write to txt file
                with open(os.path.join(item_path, f"{item}_{n}.txt"), "w") as f:
                    f.write(decoded_text)
                    print(f"Writing to {item}_{n}.txt")

            remove_html_files(item_path)

if __name__ == "__main__":
  main()
