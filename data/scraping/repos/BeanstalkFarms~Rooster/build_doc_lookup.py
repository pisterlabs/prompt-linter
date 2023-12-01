import markdown
from bs4 import BeautifulSoup
import openai
import os

openai.api_key = os.environ.get("OPENAI_API_KEY")


def md_to_text(md):
    html = markdown.markdown(md)
    soup = BeautifulSoup(html, features='html.parser')
    return soup.get_text()


def get_gpt_answer(prompt, max_tokens):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=0.0,
    )
    return response.choices[0].text


def get_all_files_recursively(dir):
    file_paths = []
    for root, directories, files in os.walk(dir):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)
    return [f for f in file_paths if f.endswith('.md')]


def get_summary_for_doc(doc_path):
    with open(doc_path, 'r') as f:
        doc = f.read()
    doc_text = md_to_text(doc)
    prompt = f"Give me a list of few word topics from this text, comma separated:\n\n{doc_text}\n\nTopics:"
    answer = get_gpt_answer(prompt, 150)
    return answer


def get_doc_title_from_path(doc_path):
    doc_title = doc_path.replace('./training-data/docs/Farmers-Almanac/', '').replace('.md', '')
    doc_title = doc_title.split('/')
    for i, token in enumerate(doc_title):
        doc_title[i] = token.replace('-', ' ').title()

    return ' - '.join(doc_title)


def generate_lookup_doc():
    """
    Create lookup.txt. This file is consumed by the server to pick the correct document to answer a user question.
    """
    all_docs = get_all_files_recursively("./training-data/docs/Farmers-Almanac")
    lookup_doc = ""

    for i, doc in enumerate(all_docs):
        doc_title = get_doc_title_from_path(doc)
        print(f'Generating summary for {doc_title} ({i+1}/{len(all_docs)})')
        doc_summary = get_summary_for_doc(doc)
        lookup_doc += f"{i+1}: {doc_title}: {doc_summary}\n"

    # write to file
    with open('./lookup.txt', 'w') as f:
        f.write(lookup_doc)


def get_doc_num_to_path():
    """
    Create lookup-paths.txt to map lookup.txt line numbers to respective doc in the Farmers' Almanac
    """
    all_docs = get_all_files_recursively("./training-data/docs/Farmers-Almanac")
    lookup_paths = ""

    for i, doc in enumerate(all_docs):
        lookup_paths += f"{doc}\n"

    # write to file
    with open('./lookup-paths.txt', 'w') as f:
        f.write(lookup_paths)
