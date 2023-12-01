from sqlalchemy import Column, Integer, String, create_engine, Text
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv
import pythonbible as bible
from bs4 import BeautifulSoup
import os, requests, llama_index
import pandas as pd

bsb = pd.read_csv('app/bsb.tsv', sep='\t')

load_dotenv()

import google_connector as gc

# create engine
pool = gc.connect_with_connector('new_verses')
Base = declarative_base()

# if temp folder doesn't exist create it
if not os.path.exists('temp'):
    os.makedirs('temp')

# create table
class NewVerse(Base):
    __tablename__ = 'new_verses'
    verse_id = Column(Integer, primary_key=True)
    bible_hub_url = Column(String)
    verse_text = Column(Text)
    commentary = Column(Text)

    def __repr__(self):
        return f"<NewVerse(verse_id='{self.verse_id}', bible_hub_url='{self.bible_hub_url}', verse_text='{self.verse_text}', commentary='{self.commentary}')>"

# create the table in the database
Base.metadata.create_all(pool)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

from llama_index import VectorStoreIndex, SimpleDirectoryReader
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from llama_index import ServiceContext

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo-1106",
    temperature=0
)

llm_embeddings = OpenAIEmbeddings()

service_context = ServiceContext.from_defaults(
    llm=llm, embed_model=llm_embeddings
)

llama_index.set_global_service_context(service_context)

def get_commentary_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    maintable2 = soup.find('div', {'class': 'maintable2'})
    jump_div = maintable2.find('div', {'id': 'jump'})
    jump_div.extract()
    return maintable2.get_text()

def get_bsb_text(verse):
    return bsb.loc[bsb['Verse'] == verse, 'Berean Standard Bible'].values[0]

def add_verse(references):
    # create session
    Session = sessionmaker(bind=pool)

    # create a new verse
    for i in references:
        verse_id = bible.convert_reference_to_verse_ids(i)
        book = str(i.book).lower().split('.')[1]
        book = book.split('_')[1] + '_' + book.split('_')[0] if '_' in book else book
        for j in verse_id:
            session = Session()
            reference = bible.convert_verse_ids_to_references([j])
            bible_hub_url = f"https://biblehub.com/commentaries/{book}/{reference[0].start_chapter}-{reference[0].start_verse}.htm"
            commentary_text = get_commentary_text(bible_hub_url)

            new_verse = NewVerse(
                verse_id=j, 
                bible_hub_url=bible_hub_url, 
                verse_text=bible.get_verse_text(j),
                commentary=commentary_text
            )

            # add the new verse to the session
            session.add(new_verse)

            # commit the transaction
            session.commit()

            # close the session
            session.close()

def check_if_verse_exists(verse_id):
    # create session
    Session = sessionmaker(bind=pool)
    session = Session()

    # query the new_verses table
    verse = session.query(NewVerse).filter(NewVerse.verse_id == verse_id).first()

    # close the session
    session.close()

    if verse is not None:
        return verse.commentary
    else:
        new_ref = bible.convert_verse_ids_to_references([verse_id])
        add_verse(new_ref)
        return check_if_verse_exists(verse_id)

def get_commentary_from_db(references):
    output = ''
    for i in references:
        verse_id = bible.convert_reference_to_verse_ids(i)
        temp = bible.convert_verse_ids_to_references([verse for verse in verse_id])
        reference_out = bible.format_scripture_references(temp)
        output += f'\n{reference_out}'
        text_out = ''
        for j in verse_id:
            ref = bible.convert_verse_ids_to_references([j])
            temp_ref = bible.format_scripture_references(ref)
            ref = ref[0]
            output += f'\n{ref.start_chapter}.{ref.start_verse} - {check_if_verse_exists(j)}'
            try:
               text_out += f'{get_bsb_text(temp_ref)}\n'
               version = 'BSB'
            except:
                text_out += f'{bible.get_verse_text(j)}\n'
                version = 'ASV'
    return output, reference_out, text_out, version

def check_input(input):
    references = bible.get_references(input)
    if len(references) == 0:
        return None
    else:
        text_, reference_out, text_out, version = get_commentary_from_db(references)
        # write text_ to file
        with open('temp/temp.txt', 'w', encoding="utf-8") as f:
            f.write(text_)

        return f'  \n{text_out} - {reference_out} ({version})'
    
def generate_query_index():
    print('Generating query index...')
    index = VectorStoreIndex([])
    node_parser = index.service_context.node_parser
    documents = SimpleDirectoryReader('temp').load_data()
    for doc in documents:
        index.insert_nodes(node_parser.get_nodes_from_documents([doc]))

    return index.as_query_engine()