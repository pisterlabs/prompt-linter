import argparse
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.html import partition_html
from unstructured.documents.elements import Element
from research_copilot.db import graph
from langchain.graphs.graph_document import GraphDocument, Node, Relationship
from langchain.schema import Document
from hashlib import md5
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from unstructured.cleaners.core import clean
from sentence_transformers import SentenceTransformer
import pandas as pd
import os
from pathlib import Path
import logging
from tqdm import tqdm

log = logging.getLogger(__name__)

embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
# default dim: 1024
# neo4j can handle up to 2048

def ingest_data(filepath: str):
  if not os.path.exists(filepath):
    raise Exception(f"File {filepath} does not exist.")
  
  file_suffix = os.path.splitext(filepath)[1]
  
  if file_suffix == '.pdf':
    return ingest_pdf(filepath)
  elif file_suffix == '.csv':
    return ingest_courtlistener_csv(filepath)
  else:
    raise Exception(f"File type {file_suffix} is not supported.")
 
def ingest_pdf(filepath: str):
  # partition
  log.info('Partitioning PDF')
  document, elements = parse_pdf_to_document(filepath)
  partitioned_graph_document = convert_unstructured_elements_to_graph_document(document, elements)

  # chunk and clean
  log.info('Cleaning and chunking partition text')
  chunked_graph_document = chunk_partitioned_graph_document(partitioned_graph_document, add_text_embeddings=True) 

  # store in db
  log.info('Storing in db')
  graph.add_graph_documents([partitioned_graph_document, chunked_graph_document])
  
  log.info('Done')
  return

## TODO TEST ME
def ingest_courtlistener_csv(filepath: str):
  log.info('Ingesting CSV - assuming this is exported courtlistener data')
  
  # extract html and metadata
  log.info('Extracting html and metadata')
  df = pd.read_csv(filepath)

  def get_html_content(row)->str|None:
    opinion_htmls = [row.html_with_citations, row.html, row.html_lawbox, row.html_columbia, row.html_anon_2020]
    for html_content in opinion_htmls:
      if html_content is not None and len(html_content) > 0:
        return html_content
    if row.xml_harvard is not None and len(row.xml_harvard) > 0:
      #  TODO convert to html
      return None
    return None

  for index, row in tqdm(df.iterrows()):
    html_content = get_html_content(row)
    if html_content is None:
      log.warn(f'No html content found for row {index}')
      continue
    metadata = row[df.columns not in ['html_with_citations', 'html', 'html_lawbox', 'html_columbia', 'html_anon_2020']]
    metadata['id'] = row.search_opinion_id
    ingest_html_content(html_content, metadata=metadata)
  log.info('Done')

def ingest_html_content(html_content: str, metadata=None):
  # partition
  document, elements = parse_html_to_document(html_content, metadata=metadata)
  partitioned_graph_document = convert_unstructured_elements_to_graph_document(document, elements)
  # chunk and clean
  chunked_graph_document = chunk_partitioned_graph_document(partitioned_graph_document, add_text_embeddings=True)
  # store in db
  graph.add_graph_documents([partitioned_graph_document, chunked_graph_document])
  return

def parse_html_to_document(html_content: str, metadata=None):
  elements = partition_html(text=html_content)
  document = Document(
    page_content='NO_DATA', # TODO replace?
    metadata={
    'source_type': NODE_TYPE.HTML,
    **(metadata or {})
  })
  return document, elements


def default_properties():
  return {
    'created_at': datetime.now().isoformat(),
  }

def chunk_partitioned_graph_document(graph_document: GraphDocument, add_text_embeddings: bool = False):
  '''
  Chunk the underlying text from ap partitioned graph document and return a new graph document with the chunks.
  '''
  chunked_graph_document = GraphDocument(nodes=[], relationships=[], source=graph_document.source)
  for node in graph_document.nodes:
    chunks = chunk_text(clean(node.properties['text'], extra_whitespace=True))
    embeddings = embedding_model.encode(chunks) if add_text_embeddings else [None] * len(chunks)
    for chunk, embedding in zip(chunks, embeddings):
      chunk_id = md5(chunk.encode()).hexdigest()
      chunk_node = Node(id=chunk_id, type=NODE_TYPE.TEXT_CHUNK, properties={'text': chunk, 'text_embedding': embedding, 'text_len': len(chunk), **default_properties()})
      chunk_relationship = Relationship(source=chunk_node, target=node, type=RELATIONSHIP_TYPE.CHUNK_OF, properties={**default_properties()})

      chunked_graph_document.nodes.append(chunk_node)
      chunked_graph_document.relationships.append(chunk_relationship)
      # TODO add relationships between nodes to denote order in document / hierarchy
  return chunked_graph_document
  
def chunk_text(text: str):
  '''
  Chunk text into smaller strings.
  '''
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size = 1000, # TODO tune this
      chunk_overlap  = 20,
      length_function = len,
      is_separator_regex = False,
  )
  chunks = text_splitter.split_text(text)
  # TODO track metadata about where in the source text the chunks came from (coordinates)
  return chunks

class NODE_TYPE: # applied as node label in neo4j
  UNSTRUCTURED_ELEMENT = 'UNSTRUCTURED_ELEMENT'
  PDF = 'PDF'
  TEXT_CHUNK = 'TEXT_CHUNK'
  HTML = 'HTML'

class RELATIONSHIP_TYPE:
  PARTITION_OF = 'PARTITION_OF'
  CHUNK_OF = 'CHUNK_OF'

def parse_pdf_to_document(filepath: str, metadata=None):
  '''
  Parse a PDF into a list of elements and a root document
  '''
  elements = partition_pdf(filepath, url=None, strategy='hi_res')
  document = Document(
    page_content='NO_DATA', # TODO replace?
    metadata={
    'filepath': filepath,
    'id': md5(filepath.encode()).hexdigest(),
    'source_type': NODE_TYPE.PDF,
    **(metadata or {})
  })
  return document, elements

# TODO rename data `source`, confusing with neo4j `source` node in relationships. probably want to just leave langchain abstractions
def convert_unstructured_elements_to_graph_document(source_document: Document, elements: list[Element]):
  '''
  Convert a list of unstructured elements into a list of graph documents.
  '''

  """
  element example:

  {'type': 'Title',
    'element_id': '015301d4f56aa4b20ec10ac889d2343f',
    'metadata': {'coordinates': {'points': ((157.62199999999999,
        114.23496279999995),
        (157.62199999999999, 146.5141628),
        (457.7358962799999, 146.5141628),
        (457.7358962799999, 114.23496279999995)),
      'system': 'PixelSpace',
      'layout_width': 612,
      'layout_height': 792},
      'filename': 'layout-parser-paper-fast.pdf',
      'file_directory': '../data',
      'last_modified': '2023-09-19T12:38:49',
      'filetype': 'application/pdf',
      'page_number': 1},
    'text': 'LayoutParser: A Uniﬁed Toolkit for Deep Learning Based Document Image Analysis'}
  """
  """
  graph document example:
  node:
  id='http://www.wikidata.org/entity/Q55979202' type='Person' properties={'name': 'Leila Stahl Buffett'}
  relationship:  
  source=Node(id='http://www.wikidata.org/entity/Q55979202', type='Person', properties={}) target=Node(id='http://www.wikidata.org/entity/Q723488', type='Person', properties={}) type='ROMANTIC_RELATIONSHIP' properties={'evidence': 'He married Leila Stahl on December 27, 1925.', 'isNotCurrent': 'true', 'startTime': '1925-12-27'}
  source:
  page_content='...' metadata={'title': 'Howard Buffett', 'summary': 'Howard Homan Buffett (August 13, 1903 – April 30, 1964) was an American businessman, investor, and politician. He was a four-term Republican United States Representative for the state of Nebraska. He was the father of Warren Buffett, the American billionaire businessman and investor.\n\n', 'source': 'https://en.wikipedia.org/wiki/Howard_Buffett'}
  """
  graph_document = GraphDocument(nodes=list(), relationships=list(), source=source_document)
  # create source node
  source_node = Node(id=source_document.metadata['id'], type=source_document.metadata['source_type'], properties=source_document.metadata) # TODO pass in metadata to properties if necessary
  for element in elements:
    element_dict = element.to_dict()
    # don't include metadata in node properties, breaks neo4j on load (can't handle non-primitive values)
    node = Node(id=str(element.id), type=NODE_TYPE.UNSTRUCTURED_ELEMENT, properties={
      'element_type': element_dict['type'],
      'element_id': element_dict['element_id'],
      'text': element_dict['text'],
      'text_len': len(element_dict['text']), 
      **default_properties()})
    relationship = Relationship(source=node, target=Node(id=source_node.id), type=RELATIONSHIP_TYPE.PARTITION_OF, properties={**default_properties()})
    graph_document.nodes.append(node)
    graph_document.relationships.append(relationship)
    # TODO add relationships between nodes to denote order in document / hierarchy
  return graph_document



def main():
  parser = argparse.ArgumentParser(description='Ingest data into the research copilot database.')
  parser.add_argument('filepath', type=str, help='path to file to ingest')
  args = parser.parse_args()
  ingest_data(args.filepath)


if __name__ == '__main__':
  # main()
  datapath = Path(__file__).parent.parent.parent.resolve().joinpath('data')
  relpath = 'complaint.pdf'
  ingest_data(str(datapath.joinpath(relpath).resolve()))

