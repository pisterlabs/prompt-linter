# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Query Data Sources
"""
import os
from typing import List, Tuple
from pathlib import Path
from common.utils.logging_handler import Logger
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import CSVLoader
from pypdf import PdfReader
from utils.errors import NoDocumentsIndexedException

# pylint: disable=broad-exception-caught

# text chunk size for embedding data
CHUNK_SIZE = 1000

class DataSource:
  """
  Class for query data sources
  """

  def __init__(self, storage_client):
    self.storage_client = storage_client
    self.docs_not_processed = []

  def download_documents(self, doc_url: str, temp_dir: Path) -> \
        List[Tuple[str, str, str]]:
    """
    Download files from doc_url source to a local tmp directory

    Args:
        doc_url: url pointing to container of documents to be indexed
        temp_dir: Path to temporary directory to download files to

    Returns:
        list of tuples (doc name, document url, local file path)
    """
    doc_filepaths = []
    bucket_name = doc_url.split("gs://")[1].split("/")[0]
    Logger.info(f"downloading {doc_url} from bucket {bucket_name}")

    for blob in self.storage_client.list_blobs(bucket_name):
      # Download the file to the tmp folder flattening all directories
      file_name = Path(blob.name).name
      file_path = os.path.join(temp_dir, file_name)
      blob.download_to_filename(file_path)
      doc_filepaths.append((blob.name, blob.path, file_path))

    if len(doc_filepaths) == 0:
      raise NoDocumentsIndexedException(
          f"No documents can be indexed at url {doc_url}")

    return doc_filepaths

  def chunk_document(self, doc_name: str, doc_url: str, doc_filepath: str) -> \
                      List[str]:
    """
    Process doc into chunks for embeddings

    Args:
       doc_name: file name of document
       doc_url: remote url of document
       doc_filepath: local file path of document
    Returns:
       list of text chunks or None if the document could not be processed
    """

    text_chunks = None

    # use langchain text splitter
    text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE,
                                          chunk_overlap=0)

    Logger.info(f"generating index data for {doc_name}")

    # read doc data and split into text chunks
    # skip any file that can't be read or generates an error
    doc_text_list = None
    try:
      doc_text_list = self.read_doc(doc_name, doc_filepath)
      if doc_text_list is None:
        Logger.error(f"no content read from {doc_name}")
        self.docs_not_processed.append(doc_url)
    except Exception as e:
      Logger.error(f"error reading doc {doc_name}: {e}")
      self.docs_not_processed.append(doc_url)

    if doc_text_list is not None:
      # split text into chunks
      text_chunks = []
      for text in doc_text_list:
        text_chunks.extend(text_splitter.split_text(text))

      if all(element == "" for element in text_chunks):
        Logger.warning(f"All extracted pages from {doc_name} are empty.")
        self.docs_not_processed.append(doc_url)

    return text_chunks

  def read_doc(self, doc_name: str, doc_filepath: str) -> List[str]:
    """
    Read document and return content as a list of strings

    Args:
      doc_name: name of document
      doc_filepath: local file path
    Returns:
      doc content as a list of strings
    """
    doc_extension = doc_name.split(".")[-1]
    doc_extension = doc_extension.lower()
    doc_text_list = None
    loader = None

    if doc_extension == "txt":
      with open(doc_filepath, "r", encoding="utf-8") as f:
        doc_text = f.read()
      doc_text_list = [doc_text]
    elif doc_extension == "csv":
      loader = CSVLoader(file_path=doc_filepath)
    elif doc_extension == "pdf":
      # read PDF into array of pages
      doc_text_list = []
      with open(doc_filepath, "rb") as f:
        reader = PdfReader(f)
        num_pages = len(reader.pages)
        Logger.info(f"Reading pdf file {doc_name} with {num_pages} pages")
        for page in range(num_pages):
          doc_text_list.append(reader.pages[page].extract_text())
        Logger.info(f"Finished reading pdf file {doc_name}")
    else:
      # return None if doc type not supported
      Logger.error(
          f"Cannot read {doc_name}: unsupported extension {doc_extension}")
      pass

    if loader is not None:
      langchain_document = loader.load()
      doc_text_list = [section.content for section in langchain_document]

    return doc_text_list
