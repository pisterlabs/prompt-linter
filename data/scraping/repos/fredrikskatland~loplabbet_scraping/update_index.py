import requests
from bs4 import BeautifulSoup
from langchain.indexes import SQLRecordManager
from langchain.indexes import index as LangchainIndex
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import os
import pinecone

def get_windows_env(var_name):
    try:
        result = os.popen(f"powershell.exe -c 'echo $env:{var_name}'").read().strip()
        return result
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

pinecone.init(
                api_key=get_windows_env("PINECONE_API_FINN"),  # find at app.pinecone.io
                environment = get_windows_env("PINECONE_ENV_FINN")  # next to api key in console
            )
embeddings = OpenAIEmbeddings(openai_api_key=get_windows_env("OPENAI_API_KEY"))
collection_name = "loplabbet-produkter"
index_name = "loplabbet-produkter"
pinecone_index = pinecone.Index(index_name)
#index_stats_response = pinecone_index.describe_index_stats()
#index_name = "loplabbet-produkter"
vectorstore = Pinecone.from_existing_index(index_name = index_name, embedding=embeddings)

    # Setting up a record manager
namespace = f"pinecone/{collection_name}"
record_manager = SQLRecordManager(
    namespace, db_url="sqlite:///record_manager_cache.sql"
)
record_manager.create_schema()




def extract_text_from(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    try:
        # Headline
        headline = soup.select_one('html > body > div:nth-of-type(1) > main > div > section:nth-of-type(2) > div > div:nth-of-type(2) > div:nth-of-type(1) > div > h1').text

        # Headline description
        headline_description = soup.select_one('html > body > div:nth-of-type(1) > main > div > section:nth-of-type(2) > div > div:nth-of-type(2) > div:nth-of-type(1) > div > div:nth-of-type(3) > p').text

        # Beskrivelse
        beskrivelse = soup.select_one('html > body > div:nth-of-type(1) > main > div > section:nth-of-type(2) > div > div:nth-of-type(1) > div > div:nth-of-type(2) > div > div:nth-of-type(2) > div')
        if beskrivelse is not None:
            beskrivelse_tekst = '\n'.join(child.get_text().strip() for child in beskrivelse.children if child.name)
        else:
            beskrivelse_tekst = ''

        # Ratings
        rating_sections = soup.select('.ll-product-review__rating')

        ratings_dict = {}

        for section in rating_sections:
            # Extract the rating category name
            category_name = section.select_one('.ll-product-rating__label').get_text().strip()
            
            # Extract the star rating
            star_container = section.select_one('.ll-product-rating__stars')
            #star_rating = str(int(star_container['class'][-1].split('-')[-1])) + " av 6"
            star_rating = int(star_container['class'][-1].split('-')[-1])
            
            ratings_dict[category_name] = star_rating

        if len(ratings_dict) == 0:
            ratings_dict = {'Stabilitet':'', 'Støtdemping':'', 'Løpsfølelse':''}


        # Details
        detail_sections = soup.select('.ll-product-detail--container')

        details_dict = {}

        for section in detail_sections:
            # Extract the detail description
            detail_description_raw = section.select_one('div > div:not(.ll-product-detail--bold)').get_text().strip()
            
            # Split the description at the non-breaking space and take the first portion
            detail_description = detail_description_raw.split('\xa0')[0]

            # Extract the detail value and remove any non-numeric characters like spaces
            detail_value_raw = section.select_one('.ll-product-detail--bold').get_text().strip()
            
            # Check if the value is numeric (like '8') or a string (like '30/22')
            if '/' in detail_value_raw:
                detail_value = detail_value_raw
            else:
                detail_value = int(detail_value_raw)

            details_dict[detail_description] = detail_value

        output = {
            'headline': headline,
            'headline_description': headline_description,
            'beskrivelse_tekst': beskrivelse_tekst,
            'url': url,
            'ratings': ratings_dict,
            'details': details_dict,

        }
    except:
        output = {
            'headline': 'Parse error',
            'headline_description': 'Parse error',
            'beskrivelse_tekst': 'Parse error',
            'url': url,
            'ratings': {'Stabilitet':'', 'Støtdemping':'', 'Løpsfølelse':''},
            'details': {},
        }
    return output

def format_to_markdown(data):
    details_with_linebreaks = data['beskrivelse_tekst'].replace('\n', '  \n') # Ensuring line breaks are respected in markdown
    markdown_text = f"## {data['headline']}\n"
    markdown_text += f"\n**Beskrivelse:** {data['headline_description']}\n"
    markdown_text += f"\n### Detaljer:\n"
    markdown_text += f"\n{details_with_linebreaks}\n"
    for key, value in data['details'].items():
        markdown_text += f"\n- **{key}**: {value}"
    markdown_text += f"\n### Vurderinger:\n"
    for key, value in data['ratings'].items():
        markdown_text += f"\n- **{key}**: {value}"
    markdown_text += f"\n[Kilde]({data['url']})"
    return markdown_text

def update_index(df, indexing_type="incremental"):
    product_urls = df['links'].tolist()
    sex = df['gender'].tolist()
    categories = df['category'].tolist()
    ranks = range(1, len(product_urls)+1)

    # Extract texts
    texts = []
    numberOfUrls = len(product_urls)
    counter = 0

    # Loop through dataframe
    for index, row in df.iterrows():
        texts.append(extract_text_from(row['links']))
        counter += 1
        print(f"Extracted text from {row['links']}. {counter} of {numberOfUrls} done.")

    
    # Format to markdown
    markdown_texts = [format_to_markdown(text) for text in texts]

    # Set url and headline as metadata
    metadatas = [{'url': text['url']
                    , 'produktnavn': text['headline']
                    , 'ratings_Stabilitet': text['ratings'].get('Stabilitet', '')
                    , 'ratings_Demping': text['ratings'].get('Støtdemping', '')
                    , 'ratings_Løpsfølelse': text['ratings'].get('Løpsfølelse', '')
                    #, 'details': text['details']
                    , 'sex': sex
                    , 'category': category
                    , 'rank': rank} for text, sex, category, rank in zip(texts, sex, categories, ranks)]

    documents = [Document(page_content=string, metadata=meta) for string, meta in zip(markdown_texts, metadatas)]
    print(LangchainIndex(
        documents,
        record_manager,
        vectorstore,
        cleanup=indexing_type,
        source_id_key="url"
    ))

