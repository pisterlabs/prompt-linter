import requests
import json
import os.path
import openai
from gpt_index import SimpleDirectoryReader
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from cloud_storage import *
import uuid

def make_post_api_request(url, headers, data):
    response = requests.post(url, headers=headers, data=json.dumps(data))
    response.raise_for_status()
    return response.json()

def make_get_api_request(url, headers, data):
    response = requests.get(url, headers=headers, data=json.dumps(data))
    response.raise_for_status()
    return response.json()


def get_all_identifiers(response):
    identifiers = []
    for result in response["result"]["content"]:
        identifiers.append(result["identifier"])
    return identifiers

def find_children_with_pdf_mime_type(content):
    coontentMetdata = []
    for child in content["children"]:
        if child["mimeType"] in ["application/pdf", "video/mp4"]:
            coontentMetdata.append({ 
                "name": child["name"],
                "previewUrl": child["previewUrl"],
                "artifactUrl": child["artifactUrl"],
                # "streamingUrl": child["streamingUrl"],
                "downloadUrl": child["downloadUrl"],
                "mimeType": child["mimeType"],
                "identifier" : child["identifier"],
                "contentType": child["contentType"]
            })
        elif child["mimeType"] == "application/vnd.ekstep.content-collection":
            coontentMetdata.extend(find_children_with_pdf_mime_type(child))
    return coontentMetdata

def get_metadata_of_children(identifiers):
    contents = []
    for identifier in identifiers:
        url = "https://sunbirdsaas.com/action/content/v3/hierarchy/{}".format(identifier)
        response = make_get_api_request(url, None, None)
        childrens = find_children_with_pdf_mime_type(response["result"]["content"])
        contents = contents + childrens
    return contents

def extract_filename_from_url(url):
  """Extracts the file name from the given URL.

  Args:
    url: The URL to extract the file name from.

  Returns:
    The file name, or None if the URL does not contain a file name.
  """

  url_parts = url.split("/")
  filename = url_parts[-1]
  if filename == "":
    return None
  return filename

def download_pdf(url, save_path):
    """Downloads a big PDF file from the given URL and saves it to the given filename.

    Args:
        url: The URL of the PDF file.
        filename: The filename to save the PDF file to.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(save_path, 'wb') as pdf_file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    pdf_file.write(chunk)
        print("Content downloaded and saved successfully. ===>" , save_path)
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        print("Content downloaded and saved failed. ===>" , save_path)

def get_all_collection(): 
    url = "https://sunbirdsaas.com/api/content/v1/search"
    headers = {"Content-Type": "application/json"}
    data = {
        "request": {
            "filters": {
                "channel": "013812745304276992183",
                "contentType": ["Collection"],
                "keywords": ["djp_category_toys", "djp_category_games", "djp_category_stories", "djp_category_flashc", "djp_category_activitys", "djp_category_manuals"]
            }
        }
    }
    response = make_post_api_request(url, headers, data)
    return response

def get_list_of_documents(contents):
    source_chunks = []
    indexed_content = []
    for index, data in enumerate(contents):
        if not data["identifier"] in indexed_content:
            sources = SimpleDirectoryReader(input_files=[data["filepath"]] ,recursive=True).load_data()
            splitter = RecursiveCharacterTextSplitter(chunk_size=4 * 1024, chunk_overlap=200)
            counter = 0
            for index, source in enumerate(sources):
                for chunk in splitter.split_text(source.text):
                    # new_metadata = {"source": str(counter), "doc_id":  source.doc_id}.update(data)
                    source_chunks.append(Document(page_content=chunk, metadata=data))
                    counter += 1
            indexed_content.append(data["identifier"])
    print("Total indexed content ::", len(indexed_content))
    return source_chunks

def langchain_indexing(uuid_number, documents):
    load_dotenv()
    try:
        search_index = FAISS.from_documents(documents, OpenAIEmbeddings())
        search_index.save_local("")
        error_message = None
        status_code = 200
    except openai.error.RateLimitError as e:
        error_message = f"OpenAI API request exceeded rate limit: {e}"
        status_code = 500
    except (openai.error.APIError, openai.error.ServiceUnavailableError):
        error_message = "Server is overloaded or unable to answer your request at the moment. Please try again later"
        status_code = 503
    except Exception as e:
        error_message = str(e.__context__) + " and " + e.__str__()
        status_code = 500
    return error_message, status_code

def main():
    
     # Make the first API request to search for collections
    collections = get_all_collection()

    # Get all the identifiers from the response
    identifiers = get_all_identifiers(collections)
    print("Total collections ::", len(identifiers))

    # Get only the content which has "mimeType": "application/pdf"
    contents = get_metadata_of_children(identifiers)
    print("Total PDf contents ::", len(contents))

    # Create output directory if not exist
    output_dir_path = 'data/'
    os.makedirs(output_dir_path, exist_ok=True)

   # Download the big PDF file and save it to the given filename.
    for index, data in enumerate(contents):
        filename = extract_filename_from_url(data["artifactUrl"])
        # filesplit = os.path.splitext(filename)
        # filename = "data/content_{}.{}".format(index, filesplit[1])
        data["filepath"] = "data/" + filename
        download_pdf(data["artifactUrl"], data["filepath"])

    print("Download process sucessfully completed!")

    uuid_number = str(uuid.uuid1())
    print("uuid_number =====>", uuid_number)
    # os.makedirs(uuid_number, exist_ok=True)

    documents = get_list_of_documents(contents)
    langchain_indexing(uuid_number, documents)

    index_files = ["index.faiss", "index.pkl"]
    for index_file in index_files:
        upload_file(uuid_number, index_file)
        os.remove(index_file)

    print("Index files uploaded to cloud")

    print("============ DONE =============")

if __name__ == "__main__":
    main()