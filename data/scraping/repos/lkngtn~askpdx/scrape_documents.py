import argparse
import pickle
import requests
import xmltodict

from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter
from tqdm import tqdm 

def extract_text_from(url):
    try:
        html = requests.get(url).text
        soup = BeautifulSoup(html, features="html.parser")
        main_content = soup.find('main')
        text = main_content.get_text()

        lines = (line.strip() for line in text.splitlines())
        return '\n'.join(line for line in lines if line)
    except Exception as e:
        print(e)
        return ''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Embedding website content')
    parser.add_argument('-s', '--sitemap', type=str, required=False,
            help='PATH to your sitemap.xml', default='sitemap.xml')
    parser.add_argument('-f', '--filter', type=str, required=False,
            help='Text which needs to be included in all URLs which should be considered',
            default='https://www.portland.gov/')
    args = parser.parse_args()

    with open(args.sitemap) as sitemap:
        raw = xmltodict.parse(sitemap.read())

    pages = []
    pbar = tqdm(raw['items']['item'])
    for url in pbar:
        url = url['url']
        # print(f'processing: ', url)
        pbar.set_description(f'Processing: {url}')
        if args.filter in url:
            pages.append({'text': extract_text_from(url), 'source': url})
     

    text_splitter = CharacterTextSplitter(chunk_size=1000, separator="\n")

    docs, metadatas = [], []
    for page in pages:
        splits = text_splitter.split_text(page['text'])
        docs.extend(splits)
        metadatas.extend([{"source": page['source']}] * len(splits))
        print(f"Split {page['source']} into {len(splits)} chunks")

# Store docs, metadatas locally
    with open("docs.pkl", "wb") as f:
        pickle.dump(docs, f)

    with open("metadatas.pkl", "wb") as f:
        pickle.dump(metadatas, f)
