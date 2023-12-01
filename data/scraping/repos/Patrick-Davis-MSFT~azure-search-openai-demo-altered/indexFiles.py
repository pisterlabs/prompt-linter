import os
import io
import openai
import time
import queue
import base64
import sys
import re
import html
from io import BytesIO
from core.modelhelper import get_token_limit
from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes.models import (
    HnswParameters,
    PrioritizedFields,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticSettings,
    SimpleField,
    VectorSearch,
    VectorSearchAlgorithmConfiguration,
)

from pypdf import PdfReader, PdfWriter
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

class indexFiles():
    def __init__(self, storageAcct:str, stagingContainer: str, indexContainer: str, formAnalyzer: str,formAnalyzerKey: str, cognitiveSearch: str, embDeployment: str, aoaiService: str):
        self.storageAcct = storageAcct
        self.stagingContainer = stagingContainer
        self.indexContainer = indexContainer
        self.formAnalyzer = formAnalyzer
        self.formAnalyzerKey = formAnalyzerKey
        self.cognitiveSearch = cognitiveSearch
        self.embDeployment = embDeployment
        self.aoaiService = aoaiService
        self.chatgpt_token_limit = 5000 #get_token_limit(chatgpt_model)
        
        self.open_ai_token_cache = {}
        self.CACHE_KEY_TOKEN_CRED = 'openai_token_cred'
        self.CACHE_KEY_CREATED_TIME = 'created_time'
        self.CACHE_KEY_TOKEN_TYPE = 'token_type'
        self.KB_FIELDS_CONTENT = os.environ.get("KB_FIELDS_CONTENT") or "content"
        self.KB_FIELDS_CATEGORY = os.environ.get("KB_FIELDS_CATEGORY") or "category"
        self.KB_FIELDS_SOURCEPAGE = os.environ.get("KB_FIELDS_SOURCEPAGE") or "sourcepage"

        
    def run(self, cognitiveSearchIndex, openAIAuth, azure_credential, printAPI: queue.Queue): 
        self.cognitiveSearchIndex = cognitiveSearchIndex 
        self.printAPI = printAPI
        print("running indexFiles\n")
        printAPI.put("Processing Files...")
        blob_service_client = BlobServiceClient(account_url=f"https://" + self.storageAcct + ".blob.core.windows.net", credential=azure_credential)
        container_client = blob_service_client.get_container_client(self.stagingContainer)
        upload_client = blob_service_client.get_container_client(self.indexContainer)

        self.printAPI.put("Generating Search Keys..." )
        openai.api_key = azure_credential.get_token("https://cognitiveservices.azure.com/.default").token
        openai.api_type = "azure_ad"

        self.open_ai_token_cache[self.CACHE_KEY_CREATED_TIME] = time.time()
        self.open_ai_token_cache[self.CACHE_KEY_TOKEN_CRED] = azure_credential
        self.open_ai_token_cache[self.CACHE_KEY_TOKEN_TYPE] = "azure_ad"
        
        self.printAPI.put("Stage 1 Chunking...")
        # Split PDF files into pages
        self.splitPDF(container_client, upload_client)

        
        self.printAPI.put("Stage 2 Check Index...")
        #check if index exists
        self.create_index(azure_credential)

        self.printAPI.put("Stage 3 Indexing...")
        # Index PDF files then delete them from staging
        blobs = container_client.list_blobs()
        for file in blobs:
            sys.stdout.write("Processing file \n" + str(file.name))
            sys.stdout.flush()
            self.printAPI.put("Indexing file: " + str(file.name))   
            page_map = self.get_document_text(file.name, blob_service_client, container_client)
            use_vectors = True
            sections = self.create_sections(file.name, page_map, use_vectors)
            self.index_sections(file.name, sections, azure_credential)
            #Remove File after indexing
            container_client.delete_blob(file.name)


        self.printAPI.put("Stage Completed Indexing...")
        return {"status": "Index Completed"}
    
    def create_index(self, azure_credential):
        key = AzureKeyCredential(os.environ.get("AZURE_SEARCH_SERVICE_KEY") or "key")
        print(f"Checking for search index '{self.cognitiveSearchIndex}' in search service '{self.cognitiveSearch}'")
        search_client = SearchIndexClient(endpoint=f"https://{self.cognitiveSearch}.search.windows.net/", credential=key)
        #search_client = SearchIndexClient(endpoint=f"https://{self.cognitiveSearch}.search.windows.net/", credential=azure_credential)
        
        #check index exists
        try:    
            if search_client.get_index(self.cognitiveSearchIndex):
                print(f"\tIndex '{self.cognitiveSearchIndex}' already exists, skipping creation")
                self.printAPI.put(f"Index '{self.cognitiveSearchIndex}' already exists, skipping creation")
                return
        except: 
            print(f"\tIndex '{self.cognitiveSearchIndex}' does not exist on '{self.cognitiveSearch}', creating...")
            self.printAPI.put(f"Index '{self.cognitiveSearchIndex}' does not exist on '{self.cognitiveSearch}', creating...")
            index = SearchIndex(
                name=self.cognitiveSearchIndex,
                fields=[
                    SimpleField(name="id", type="Edm.String", key=True, sortable=True),
                    SearchableField(name="content", type="Edm.String", analyzer_name="en.microsoft"),
                    SearchField(name="embedding", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                                hidden=False, searchable=True, filterable=False, sortable=False, facetable=False,
                                vector_search_dimensions=1536, vector_search_configuration="default"),
                    SimpleField(name="category", type="Edm.String", filterable=True, facetable=True, sortable=True),
                    SearchableField(name="sourcepage", type="Edm.String", filterable=True, facetable=True, analyzer_name="en.microsoft", sortable=True),
                    SearchableField(name="sourcefile", type="Edm.String", filterable=True, facetable=True, analyzer_name="en.microsoft", sortable=True)
                ],
                semantic_settings=SemanticSettings(
                    configurations=[SemanticConfiguration(
                        name='default',
                        prioritized_fields=PrioritizedFields(
                            title_field=None, prioritized_content_fields=[SemanticField(field_name='content')]))]),
                    vector_search=VectorSearch(
                        algorithm_configurations=[
                            VectorSearchAlgorithmConfiguration(
                                name="default",
                                kind="hnsw",
                                hnsw_parameters=HnswParameters(metric="cosine")
                            )
                        ]
                    )
                )
            try:
                search_client.create_index(index)
            except Exception as ex:
                print(str(ex))
                self.printAPI.put(f"Error Creating Index: '{str(ex)}'")
                raise (ex)
            finally:
                i=0
        finally:
            return

    def index_sections(self, filename, sections, azure_credential):
        print(f"Indexing sections from '{filename}' into search index '{self.cognitiveSearch}'")
        self.printAPI.put(f"Indexing sections from '{filename}' into search index '{self.cognitiveSearch}'")
    

        search_client = SearchClient(endpoint=f"https://{self.cognitiveSearch}.search.windows.net/",
                                        index_name=self.cognitiveSearchIndex,
                                        credential=azure_credential)
        i = 0
        batch = []
        for s in sections:
            batch.append(s)
            i += 1
            if i % 1000 == 0:
                results = search_client.upload_documents(documents=batch)
                succeeded = sum([1 for r in results if r.succeeded])
                print(f"\tIndexed {len(results)} sections, {succeeded} succeeded")
                self.printAPI.put(f"\tIndexed {len(results)} sections, {succeeded} succeeded")
                batch = []

        if len(batch) > 0:
            results = search_client.upload_documents(documents=batch)
            succeeded = sum([1 for r in results if r.succeeded])
            print(f"\tIndexed {len(results)} sections, {succeeded} succeeded")
            self.printAPI.put(f"\tIndexed {len(results)} sections, {succeeded} succeeded")
    
    def blob_name_from_file_page(self, filename, page = 0):
        if os.path.splitext(filename)[1].lower() == ".pdf":
            return os.path.splitext(os.path.basename(filename))[0] + f"-{page}" + ".pdf"
        else:
            return os.path.basename(filename)

    def filename_to_id(self, filename):
        filename_ascii = re.sub("[^0-9a-zA-Z_-]", "_", filename)
        filename_hash = base64.b16encode(filename.encode('utf-8')).decode('ascii')
        return f"file-{filename_ascii}-{filename_hash}"
    
    def split_text(self, page_map, filename):
        MAX_SECTION_LENGTH = 1000
        SENTENCE_SEARCH_LIMIT = 100
        SECTION_OVERLAP = 100
        SENTENCE_ENDINGS = [".", "!", "?"]
        WORDS_BREAKS = [",", ";", ":", " ", "(", ")", "[", "]", "{", "}", "\t", "\n"]
        print(f"Splitting '{filename}' into sections")
        self.printAPI.put(f"Splitting '{filename}' into sections")

        def find_page(offset):
            num_pages = len(page_map)
            for i in range(num_pages - 1):
                if offset >= page_map[i][1] and offset < page_map[i + 1][1]:
                    return i
            return num_pages - 1

        all_text = "".join(p[2] for p in page_map)
        length = len(all_text)
        start = 0
        end = length
        while start + SECTION_OVERLAP < length:
            last_word = -1
            end = start + MAX_SECTION_LENGTH

            if end > length:
                end = length
            else:
                # Try to find the end of the sentence
                while end < length and (end - start - MAX_SECTION_LENGTH) < SENTENCE_SEARCH_LIMIT and all_text[end] not in SENTENCE_ENDINGS:
                    if all_text[end] in WORDS_BREAKS:
                        last_word = end
                    end += 1
                if end < length and all_text[end] not in SENTENCE_ENDINGS and last_word > 0:
                    end = last_word # Fall back to at least keeping a whole word
            if end < length:
                end += 1

            # Try to find the start of the sentence or at least a whole word boundary
            last_word = -1
            while start > 0 and start > end - MAX_SECTION_LENGTH - 2 * SENTENCE_SEARCH_LIMIT and all_text[start] not in SENTENCE_ENDINGS:
                if all_text[start] in WORDS_BREAKS:
                    last_word = start
                start -= 1
            if all_text[start] not in SENTENCE_ENDINGS and last_word > 0:
                start = last_word
            if start > 0:
                start += 1

            section_text = all_text[start:end]
            yield (section_text, find_page(start))

            last_table_start = section_text.rfind("<table")
            if (last_table_start > 2 * SENTENCE_SEARCH_LIMIT and last_table_start > section_text.rfind("</table")):
                # If the section ends with an unclosed table, we need to start the next section with the table.
                # If table starts inside SENTENCE_SEARCH_LIMIT, we ignore it, as that will cause an infinite loop for tables longer than MAX_SECTION_LENGTH
                # If last table starts inside SECTION_OVERLAP, keep overlapping
                print(f"Section ends with unclosed table, starting next section with the table at page {find_page(start)} offset {start} table start {last_table_start}")
                self.printAPI.put(f"Section ends with unclosed table, starting next section with the table at page {find_page(start)} offset {start} table start {last_table_start}")
                start = min(end - SECTION_OVERLAP, start + last_table_start)
            else:
                start = end - SECTION_OVERLAP

        if start + SECTION_OVERLAP < end:
            yield (all_text[start:end], find_page(start))


    def create_sections(self, filename, page_map, use_vectors):
        file_id = self.filename_to_id(filename)
        for i, (content, pagenum) in enumerate(self.split_text(page_map, filename)):
            section = {
                "id": f"{file_id}-page-{i}",
                "content": content,
                "category": self.KB_FIELDS_CATEGORY,
                "sourcepage": self.blob_name_from_file_page(filename, pagenum),
                "sourcefile": filename
            }
            if use_vectors:
                section["embedding"] = self.compute_embedding(content)
            yield section
    
    def refresh_openai_token(self):
        if self.open_ai_token_cache[self.CACHE_KEY_TOKEN_TYPE] == 'azure_ad' and self.open_ai_token_cache[self.CACHE_KEY_CREATED_TIME] + 300 < time.time():
            token_cred = self.open_ai_token_cache[self.CACHE_KEY_TOKEN_CRED]
            openai.api_key = token_cred.get_token("https://cognitiveservices.azure.com/.default").token
            self.open_ai_token_cache[self.CACHE_KEY_CREATED_TIME] = time.time()

    def before_retry_sleep(retry_state):
        print("Rate limited on the OpenAI embeddings API, sleeping before retrying...") 
    
    @retry(
        retry=retry_if_exception_type(openai.error.RateLimitError),
        wait=wait_random_exponential(min=10, max=60),
        stop=stop_after_attempt(15),
        before_sleep=before_retry_sleep
    )
    def compute_embedding(self, text):
        self.refresh_openai_token()
        print(f"Computing embedding for text of length {len(text)}")
        self.printAPI.put(f"Computing embedding for text of length {len(text)}, This will sleep where needed.")
        return openai.Embedding.create(engine=self.embDeployment, model=self.embDeployment, input=text)["data"][0]["embedding"]

    def get_document_text(self, filename: str, blob_service_client: BlobServiceClient, container_client: BlobServiceClient) -> dict:
        offset = 0
        page_map = []

        
        #form_recognizer_client = DocumentAnalysisClient(endpoint=f"https://{AZURE_FORMRECOGNIZER_SERVICE}.cognitiveservices.azure.com/", credential=azure_credential, headers={"x-ms-useragent": "azure-search-chat-demo/1.0.0"})
        #form_recognizer_client = DocumentAnalysisClient(endpoint=f"https://{AZURE_FORMRECOGNIZER_SERVICE}.cognitiveservices.azure.com/", credential=azure_credential)
        form_recognizer_client = DocumentAnalysisClient(endpoint=f"https://{self.formAnalyzer}.cognitiveservices.azure.com/", credential=AzureKeyCredential(self.formAnalyzerKey))
        b = container_client.download_blob(filename).readall()
        poller = form_recognizer_client.begin_analyze_document('prebuilt-document', document = b)
        form_recognizer_results = poller.result()

        for page_num, page in enumerate(form_recognizer_results.pages):
            tables_on_page = [
                table
                for table in form_recognizer_results.tables
                if table.bounding_regions[0].page_number == page_num + 1
            ]

            # mark all positions of the table spans in the page
            page_offset = page.spans[0].offset
            page_length = page.spans[0].length
            table_chars = [-1] * page_length
            for table_id, table in enumerate(tables_on_page):
                for span in table.spans:
                    # replace all table spans with "table_id" in table_chars array
                    for i in range(span.length):
                        idx = span.offset - page_offset + i
                        if idx >= 0 and idx < page_length:
                            table_chars[idx] = table_id

            # build page text by replacing characters in table spans with table html
            page_text = ""
            added_tables = set()
            for idx, table_id in enumerate(table_chars):
                if table_id == -1:
                    page_text += form_recognizer_results.content[page_offset + idx]
                elif table_id not in added_tables:
                    page_text += self.table_to_html(tables_on_page[table_id])
                    added_tables.add(table_id)

            page_text += " "
            page_map.append((page_num, offset, page_text))
            offset += len(page_text)

        return page_map
    
    def table_to_html(self, table):
        table_html = "<table>"
        rows = [sorted([cell for cell in table.cells if cell.row_index == i], key=lambda cell: cell.column_index) for i in range(table.row_count)]
        for row_cells in rows:
            table_html += "<tr>"
            for cell in row_cells:
                tag = "th" if (cell.kind == "columnHeader" or cell.kind == "rowHeader") else "td"
                cell_spans = ""
                if cell.column_span > 1: cell_spans += f" colSpan={cell.column_span}"
                if cell.row_span > 1: cell_spans += f" rowSpan={cell.row_span}"
                table_html += f"<{tag}{cell_spans}>{html.escape(cell.content)}</{tag}>"
            table_html +="</tr>"
        table_html += "</table>"
        return table_html

    def splitPDF(self, staging_cc, search_cc) -> None:
        blobs = staging_cc.list_blobs()
        files = []
        
        print("before loop\n")
        for blob in blobs:
        # get the blob to bytes
            print("Chunking File... " + blob.name + "\n")
            self.printAPI.put("Chunking File: " + blob.name)
            blob_client = staging_cc.get_blob_client(blob.name)
            downloaded_blob = blob_client.download_blob()
            pdf_bytes = downloaded_blob.content_as_bytes()
            if blob.name.lower().endswith('.pdf'):
                # convert to pdf and split pages for pdf files
                pdf = PdfReader(BytesIO(pdf_bytes))
                pages = pdf.pages
                for i in range(len(pages)):
                    blob_name = blob.name[:-len(".pdf")] + "-" + str(i) + ".pdf"
                    f = io.BytesIO()
                    writer = PdfWriter()
                    writer.add_page(pages[i])
                    writer.write(f)
                    f.seek(0)
                    search_cc.upload_blob(blob_name, f, overwrite=True)
                    files.append(blob_name)
            else:
                search_cc.upload_blob(blob.name, blob, overwrite=True)
                files.append(blob.name)
        