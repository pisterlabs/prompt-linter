
import os
import pandas as pd
import PyPDF2
from paperscraper.pdf import save_pdf
from paperscraper.get_dumps import biorxiv

from VectorDatabase import Lantern, Fragment, Publication
import openai
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
import PyPDF2

keywords_groups = {
    'CX-MS': ['cross-link', 'crosslink', 'XL-MS', 'CX-MS', 'CL-MS', 'XLMS', 'CXMS', 'CLMS', "chemical crosslinking mass spectrometry", 'photo-crosslinking', 'crosslinking restraints', 'crosslinking-derived restraints', 'chemical crosslinking', 'in vivo crosslinking', 'crosslinking data'],
    'HDX': ['Hydrogen–deuterium exchange mass spectrometry', 'Hydrogen/deuterium exchange mass spectrometry' 'HDX', 'HDXMS', 'HDX-MS'],
    'EPR': ['electron paramagnetic resonance spectroscopy', 'EPR', 'DEER', "Double electron electron resonance spectroscopy"],
    'FRET': ['FRET',  "forster resonance energy transfer", "fluorescence resonance energy transfer"],
    'AFM': ['AFM',  "atomic force microscopy" ],
    'SAS': ['SAS', 'SAXS', 'SANS', "Small angle solution scattering", "solution scattering", "SEC-SAXS", "SEC-SAS", "SASBDB", "Small angle X-ray scattering", "Small angle neutron scattering"],
    '3DGENOME': ['HiC', 'Hi-C', "chromosome conformation capture"],
    'Y2H': ['Y2H', "yeast two-hybrid"],
    'DNA_FOOTPRINTING': ["DNA Footprinting", "hydroxyl radical footprinting"],
    'XRAY_TOMOGRAPHY': ["soft x-ray tomography"],
    'FTIR': ["FTIR", "Infrared spectroscopy", "Fourier-transform infrared spectroscopy"],
    'FLUORESCENCE': ["Fluorescence imaging", "fluorescence microscopy", "TIRF"],
    'EVOLUTION': ['coevolution', "evolutionary covariance"],
    'PREDICTED': ["predicted contacts"],
    'INTEGRATIVE': ["integrative structure", "hybrid structure", "integrative modeling", "hybrid modeling"],
    'SHAPE': ['Hydroxyl Acylation analyzed by Primer Extension']
}

import re

class LlmHandler:

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(separators = ["\n\n", "\n", ".", ","], chunk_size=300, chunk_overlap=100)
        self.llm=ChatOpenAI(
                openai_api_key=openai_api_key,
                temperature=0, model_name="gpt-4", max_tokens=300, request_timeout = 30, max_retries=3
            )
        
        
    def evaluate_queries(self, embedding, queries):
        chatbot = RetrievalQA.from_chain_type(
            llm=self.llm, 
            chain_type="stuff", 
            retriever=embedding.as_retriever(search_type="similarity", search_kwargs={"k":3})
        )
        
        template = """ {query}? """
        response = []
        for q in queries:
            prompt = PromptTemplate(
                input_variables=["query"],
                template=template,
            )

            response.append(chatbot.run(
                prompt.format(query=q)
            ))
        return response


llm = LlmHandler()

methods_string = ''
for i, (k, v) in enumerate(keywords_groups.items()):
    if i > 0:
        methods_string += ' or '
    methods_string += f'{k} ({", ".join(v)})'


def get_embeddings(fname):
    """
    """
    loader = TextLoader(fname)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(separators = ["\n\n", "\n", ".", ","],chunk_size = 300, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    emb = OpenAIEmbeddings()
    input_texts = [d.page_content for d in docs]

    input_embeddings = emb.embed_documents(input_texts)
    text_embeddings = list(zip(input_texts, input_embeddings))
    return text_embeddings, emb

def retreiveTextFromPdf(inp_file):


    json = pd.read_json(path_or_buf=inp_file, lines=True)
    lantern = Lantern()

    for n, doi in enumerate(json['doi']):
        #print(n, doi)


        ##NOTE: This is for example purpose only
        if n > 0:
            break

        if lantern.publicationExists(doi):
            continue

        paper_data = {'doi': doi}
        doi = doi.replace("/", "-")
        pdf_dir = './papers/'
        if not os.path.exists(pdf_dir):
            os.mkdir(pdf_dir)

        pdfsavefile='./papers/' + doi +'.pdf'
        save_pdf(paper_data, filepath=pdfsavefile)

        # creating a pdf reader object
        reader = PyPDF2.PdfReader(pdfsavefile)
        save_txt_path = 'scrapped_txts/'
        if not os.path.exists(save_txt_path):
            os.mkdir(save_txt_path)
        extract_text = ''
        for page in reader.pages:
            extract_text+=page.extract_text()

        txt_file = str('{}.txt'.format(doi))
        with open(save_txt_path+txt_file, 'w') as file:
            file.write(extract_text)


        txt_embs, emb = get_embeddings(save_txt_path+txt_file)

        fragments = []
        for txt, embs in txt_embs:
            fragment = Fragment(doi, 'methods', txt, embs)
            fragments.append(fragment)

        title = ""
        pmc = ""
        pubmed = ""

        publication = Publication(doi, title, pmc, pubmed, doi)

        lantern.insertEmbeddings(fragments)
        lantern.insertPublication(publication)

        os.remove(pdfsavefile)


def add_publication_by_doi(doi):
  lantern = Lantern()
  if lantern.publicationExists(doi):
    return

  paper_data = {'doi': doi}
  doi = doi.replace("/", "-")
  pdf_dir = './papers/'
  if not os.path.exists(pdf_dir):
      os.mkdir(pdf_dir)

  pdfsavefile='./papers/' + doi +'.pdf'
  save_pdf(paper_data, filepath=pdfsavefile)

  # creating a pdf reader object
  reader = PyPDF2.PdfReader(pdfsavefile)
  save_txt_path = 'scrapped_txts/'
  if not os.path.exists(save_txt_path):
      os.mkdir(save_txt_path)
  extract_text = ''
  for page in reader.pages:
      extract_text+=page.extract_text()

  txt_file = str('{}.txt'.format(doi))
  with open(save_txt_path+txt_file, 'w') as file:
      file.write(extract_text)


  txt_embs, emb = get_embeddings(save_txt_path+txt_file)

  fragments = []
  for txt, embs in txt_embs:
      fragment = Fragment(doi, 'methods', txt, embs)
      fragments.append(fragment)

  title = ""
  pmc = ""
  pubmed = ""

  publication = Publication(doi, title, pmc, pubmed, doi)

  lantern.insertEmbeddings(fragments)
  lantern.insertPublication(publication)
  #print(fragments)
  os.remove(pdfsavefile)


def process_result(result):
  if result == None:
    return (False, None)
  for response in result:
    if "cryo" in response.lower():
        return (False, None)
    return (response.lower().startswith('yes'), response)

lantern = Lantern()
def get_embeddings_for_pub(id):
  input_texts = []
  input_embeddings = []
  if lantern.publicationExists(id):
    fragments = lantern.getAllFragmentsOfPublication(id)
    for fragment in fragments:
      input_texts.append(fragment.content)
      input_embeddings.append(fragment.vector)
    text_embeddings = list(zip(input_texts, input_embeddings))
    return text_embeddings

def main():
  open_ai_emb = OpenAIEmbeddings()
  #add_publication_by_doi('10.1101/2023.10.31.564925')
  #add_publication_by_doi('10.1101/2023.03.03.531047')
  query = [f"You are reading a materials and methods section of a scientific paper. Here is the list of structural biology methods {methods_string}.\n\n Did the authors use any methods from the list? \n\n Answer with Yes or No followed by the names of the methods."]
  lantern = Lantern()
  publications = lantern.getUnreadPublication()

  all_results = []
  rows = []
  hits = 0
  for pub in publications[5:]:
    text_embeddings = get_embeddings_for_pub(pub.id)
    flag = False
    for text, _ in text_embeddings:
      if re.search("cryo-?em", text, re.IGNORECASE):
        flag = True
        break
    if flag:
      faissIndex = FAISS.from_embeddings(text_embeddings=text_embeddings, embedding=open_ai_emb)
      result = llm.evaluate_queries(faissIndex, query)
      classification, response = process_result(result)
      hits += classification
    else:
      classification, response = process_result(None)
      #print('paper not about cryo-em')
    rows.append([pub.doi, pub.title, "11-2-2023", "11-5-2023", "", int(classification), response, ""])

  from google_sheets import SpreadsheetUpdater
  gs = SpreadsheetUpdater()
  print(rows)
  gs.append_rows(rows)
  msg = f"""
    This batch of paper analysis has concluded. 
    {len(rows)} papers were analyzed in total over the date range 11/2 - 11/3
    {hits} {"were" if ((hits>0) or (hits == 0)) else was} classified as having multi-method structural data
"""
  print(msg)
  gs.email(message=msg)


main()
    

