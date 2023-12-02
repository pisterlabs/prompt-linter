import os
import argparse
import glob
import html
import io
import re
import time
from pypdf import PdfReader, PdfWriter
from azure.identity import AzureDeveloperCliCredential
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import *
from azure.search.documents import SearchClient
from azure.ai.formrecognizer import DocumentAnalysisClient
import base64
import openai
from langchain.document_loaders import UnstructuredFileLoader
import fitz
import json

parser = argparse.ArgumentParser(
    description="Prepare documents by extracting content from PDFs, splitting content into sections, uploading to blob storage, and indexing in a search index.",
    epilog="Example: index-pdf-with-image.py '..\data\*' --storageaccount myaccount --container mycontainer --searchservice mysearch --index myindex -v"
    )
parser.add_argument("files", help="Files to be processed")
parser.add_argument("--category", help="Value for the category field in the search index for all sections indexed in this run")
parser.add_argument("--skipblobs", action="store_true", help="Skip uploading individual pages to Azure Blob Storage")
parser.add_argument("--storageaccount", help="Azure Blob Storage account name")
parser.add_argument("--container", help="Azure Blob Storage container name")
parser.add_argument("--storagekey", required=False, help="Optional. Use this Azure Blob Storage account key instead of the current user identity to login (use az login to set current user for Azure)")
parser.add_argument("--tenantid", required=False, help="Optional. Use this to define the Azure directory where to authenticate)")
parser.add_argument("--searchservice", help="Name of the Azure Cognitive Search service where content should be indexed (must exist already)")
parser.add_argument("--index", help="Name of the Azure Cognitive Search index where content should be indexed (will be created if it doesn't exist)")
parser.add_argument("--searchkey", required=False, help="Optional. Use this Azure Cognitive Search account key instead of the current user identity to login (use az login to set current user for Azure)")
parser.add_argument("--remove", action="store_true", help="Remove references to this document from blob storage and the search index")
parser.add_argument("--removeall", action="store_true", help="Remove all blobs from blob storage and documents from the search index")
parser.add_argument("--localpdfparser", action="store_true", help="Use PyPdf local PDF parser (supports only digital PDFs) instead of Azure Form Recognizer service to extract text, tables and layout from the documents")
parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
#Added by Alfred
parser.add_argument("--folder", required=False, help="Azure Blob contatiner folder")
args = parser.parse_args()

azd_credential = AzureDeveloperCliCredential() if args.tenantid == None else AzureDeveloperCliCredential(tenant_id=args.tenantid)
default_creds = azd_credential if args.searchkey == None or args.storagekey == None else None
search_creds = default_creds if args.searchkey == None else AzureKeyCredential(args.searchkey)
if not args.skipblobs:
    storage_creds = default_creds if args.storagekey == None else args.storagekey

def blob_name_from_file_page(filename, page = 0):
    if os.path.splitext(filename)[1].lower() == ".pdf":
        return os.path.splitext(os.path.basename(filename))[0] + f"-{page}" + ".pdf"
    else:
        return os.path.basename(filename)

def upload_blobs(filename):
    blob_service = BlobServiceClient(account_url=f"https://{args.storageaccount}.blob.core.windows.net", credential=storage_creds)
    blob_container = blob_service.get_container_client(args.container)
    if not blob_container.exists():
        blob_container.create_container()

    # if file is PDF split into pages and upload each page as a separate blob
    if os.path.splitext(filename)[1].lower() == ".pdf":
        reader = PdfReader(filename)
        pages = reader.pages
        for i in range(len(pages)):
            blob_name = blob_name_from_file_page(filename, i)
            if args.verbose: print(f"\tUploading blob for page {i} -> {blob_name}")
            f = io.BytesIO()
            writer = PdfWriter()
            writer.add_page(pages[i])
            writer.write(f)
            f.seek(0)
            if args.folder:
                remote_file_path = os.path.join(args.folder+'/', blob_name)
                blob_container.upload_blob(remote_file_path, f, overwrite=True)
            else:
                blob_container.upload_blob(blob_name, f, overwrite=True)
    else:
        blob_name = blob_name_from_file_page(filename)
        if args.verbose: print(f"\tUploading blob -> {blob_name}")
        if args.folder:
            remote_file_path = os.path.join(args.folder+'/', blob_name)
        with open(filename,"rb") as data:
            if args.folder:
                remote_file_path = os.path.join(args.folder+'/', blob_name)
                blob_container.upload_blob(blob_name, data, overwrite=True)
            else:
                blob_container.upload_blob(blob_name, data, overwrite=True)

def remove_blobs(filename):
    if args.verbose: print(f"Removing blobs for '{filename or '<all>'}'")
    blob_service = BlobServiceClient(account_url=f"https://{args.storageaccount}.blob.core.windows.net", credential=storage_creds)
    blob_container = blob_service.get_container_client(args.container)
    if blob_container.exists():
        if filename == None:
            blobs = blob_container.list_blob_names()
        else:
            prefix = os.path.splitext(os.path.basename(filename))[0]
            blobs = filter(lambda b: re.match(f"{prefix}-\d+\.pdf", b), blob_container.list_blob_names(name_starts_with=os.path.splitext(os.path.basename(prefix))[0]))
        for b in blobs:
            if args.verbose: print(f"\tRemoving blob {b}")
            blob_container.delete_blob(b)

def upload_base64_to_blob(image_base64_str: str, doc_name, page, number):
    image_bytes = base64.b64decode(image_base64_str)
    blob_service = BlobServiceClient(account_url=f"https://{args.storageaccount}.blob.core.windows.net", credential=storage_creds)
    file_name = f"{doc_name}-{page}-{number}.png"
    blob_client = blob_service.get_blob_client(container=args.container, blob=file_name)
    blob_client.upload_blob(image_bytes, blob_type="BlockBlob", overwrite=True)
    url = f"https://{args.storageaccount}.blob.core.windows.net/{args.container}/{args.folder}/{blob_client.get_blob_properties().name}"
    return url

def get_document_text_with_images(filename):
    doc = fitz.open(filename)
    page_context = []
    for page_number, page in enumerate(doc):
        page_json_str = page.get_text('json')
        json_obj = json.loads(page_json_str)
        page_text = ""
        image_index_on_page = 0
        for b in json_obj['blocks']:
            type = b['type']
            if type == 0:
                for s in b['lines']:
                    for w in s['spans']:
                        page_text += w['text']
            elif type == 1:
                image_base64 = b['image']
                image_url=upload_base64_to_blob(image_base64, os.path.basename(filename), page_number, image_index_on_page)
                page_text += f"<img src='{image_url}'/>"
                image_index_on_page += 1
        page_context.append((page_number,page_text))
        return page_context

def create_search_index():
    if args.verbose: print(f"Ensuring search index {args.index} exists")
    index_client = SearchIndexClient(endpoint=f"https://{args.searchservice}.search.windows.net/",
                                     credential=search_creds)
    if args.index not in index_client.list_index_names():
        index = SearchIndex(
            name=args.index,
            fields=[
                SimpleField(name="id", type="Edm.String", key=True),
                SearchableField(name="content", type="Edm.String", analyzer_name="zh-Hans.microsoft"),
                SimpleField(name="category", type="Edm.String", filterable=True, facetable=True),
                SimpleField(name="sourcepage", type="Edm.String", filterable=True, facetable=True),
                SimpleField(name="sourcefile", type="Edm.String", filterable=True, facetable=True),
                SimpleField(name="metadata_storage_name", type="Edm.String", filterable=True, sortable=False, facetable=False),
                SimpleField(name="metadata_storage_path", type="Edm.String", filterable=False, sortable=False, facetable=False)
            ],
            semantic_settings=SemanticSettings(
                configurations=[SemanticConfiguration(
                    name='default',
                    prioritized_fields=PrioritizedFields(
                        title_field=None, prioritized_content_fields=[SemanticField(field_name='content')]))])
        )
        if args.verbose: print(f"Creating {args.index} search index")
        index_client.create_index(index)
    else:
        if args.verbose: print(f"Search index {args.index} already exists")

def create_sections(filename, page_context):
    for i, (pagenum, pagetext) in enumerate(page_context):
        encoded_id = base64.urlsafe_b64encode(f"{filename}-{pagenum}-section-{i}".encode('utf-8'))
        encoded_id_str = str(encoded_id.decode('utf-8'))
        yield {
            # "id": re.sub("[^0-9a-zA-Z_-]","_",f"{filename}-{i}"),
            "id": encoded_id_str,
            "content": pagetext,
            "category": '',
            "sourcepage": f"{filename}-{pagenum}.pdf",
            "sourcefile": filename,
            "metadata_storage_name": os.path.basename(filename),
            "metadata_storage_path": f"https://{args.storageaccount}.blob.core.windows.net/{args.container}/{args.folder}/{os.path.basename(filename)}"
        }

def index_sections(filename, sections):
    if args.verbose: print(f"Indexing sections from '{filename}' into search index '{args.index}'")
    search_client = SearchClient(endpoint=f"https://{args.searchservice}.search.windows.net/",
                                    index_name=args.index,
                                    credential=search_creds)
    i = 0
    batch = []
    for s in sections:
        batch.append(s)
        i += 1
        if i % 1000 == 0:
            results = search_client.upload_documents(documents=batch)
            succeeded = sum([1 for r in results if r.succeeded])
            if args.verbose: print(f"\tIndexed {len(results)} sections, {succeeded} succeeded")
            batch = []

    if len(batch) > 0:
        results = search_client.upload_documents(documents=batch)
        succeeded = sum([1 for r in results if r.succeeded])
        if args.verbose: print(f"\tIndexed {len(results)} sections, {succeeded} succeeded")

def remove_from_index(filename):
    if args.verbose: print(f"Removing sections from '{filename or '<all>'}' from search index '{args.index}'")
    search_client = SearchClient(endpoint=f"https://{args.searchservice}.search.windows.net/",
                                    index_name=args.index,
                                    credential=search_creds)
    while True:
        filter = None if filename == None else f"sourcefile eq '{os.path.basename(filename)}'"
        r = search_client.search("", filter=filter, top=1000, include_total_count=True)
        if r.get_count() == 0:
            break
        r = search_client.delete_documents(documents=[{ "id": d["id"] } for d in r])
        if args.verbose: print(f"\tRemoved {len(r)} sections from index")
        # It can take a few seconds for search results to reflect changes, so wait a bit
        time.sleep(2)

if not args.remove:
        create_search_index()
    
print(f"Processing files...")
for filename in glob.glob(args.files):
    if args.verbose: print(f"Processing '{filename}'")
    if args.remove:
        remove_blobs(filename)
        remove_from_index(filename)
    elif args.removeall:
        remove_blobs(None)
        remove_from_index(None)
    else:
        if not args.skipblobs:
            upload_blobs(filename)
        page_map = get_document_text_with_images(filename)
        sections = create_sections(os.path.basename(filename), page_map)
        index_sections(os.path.basename(filename), sections)