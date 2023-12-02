# Copyright 2023 Google LLC
# Copyright 2023 Qarik Group, LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Initial configuration and setup for VertexAI Vector Search and Index Engine.

This code is based on the following Google Cloud Platform example:
https://github.com/GoogleCloudPlatform/generative-ai/blob/main/language/use-cases/document-qa/question_answering_documents_langchain_matching_engine.ipynb?short_path=29b9638
"""


import langchain
import vertexai
from common import solution
from common.log import Logger
from google.cloud import aiplatform
from langchain.document_loaders import GCSDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from query_engine.matching_engine import ME_DIMENSIONS, CustomVertexAIEmbeddings, MatchingEngine
from query_engine.matching_engine_tools import MatchingEngineUtils

logger = Logger(__name__).get_logger()
logger.info('Initializing...')
logger.info(f'Vertex AI SDK version: {aiplatform.__version__}')
logger.info(f'LangChain version: {langchain.__version__}')

PROJECT_ID: str = solution.PROJECT_ID
REGION: str = solution.REGION
ME_REGION = solution.REGION
ME_INDEX_NAME: str = f'{PROJECT_ID}-me-index'
ME_EMBEDDING_DIR: str = solution.getenv('ME_EMBEDDING_BUCKET')
GCS_BUCKET_DOCS = solution.getenv('RESUME_BUCKET_NAME')

# Initialize Vertex AI SDK
vertexai.init(project=PROJECT_ID, location=REGION)


# Embeddings API integrated with langChain
EMBEDDING_QPM = 100
embeddings = CustomVertexAIEmbeddings(
    requests_per_minute=EMBEDDING_QPM,
)

"""
As part of the environment setup, create an index on Vertex AI Matching Engine and deploy the index to an Endpoint. Index Endpoint can be [public](https://cloud.google.com/vertex-ai/docs/matching-engine/deploy-index-public) or [private](https://cloud.google.com/vertex-ai/docs/matching-engine/deploy-index-vpc). This notebook uses a **Public endpoint**.
Refer to the [Matching Engine documentation](https://cloud.google.com/vertex-ai/docs/matching-engine/overview) for details.

NOTE: Please note creating an Index on Matching Engine and deploying the Index to an Index Endpoint can take up to 1 hour.</b>

- Configure parameters to create Matching Engine index
    - `ME_REGION`: Region where Matching Engine Index and Index Endpoint are deployed
    - `ME_INDEX_NAME`: Matching Engine index display name
    - `ME_EMBEDDING_DIR`: Cloud Storage path to allow inserting, updating or deleting the contents of the Index
    - `ME_DIMENSIONS`: The number of dimensions of the input vectors. Vertex AI Embedding API generates 768 dimensional vector embeddings.

    You can [create index](https://cloud.google.com/vertex-ai/docs/matching-engine/create-manage-index#create-index) on Vertex AI Matching Engine for batch updates or streaming updates.

This creates Matching Engine Index:
- With [streaming updates](https://cloud.google.com/vertex-ai/docs/matching-engine/create-manage-index#create-stream)
- With default configuration - e.g. small shard size

You can [update the index configuration](https://cloud.google.com/vertex-ai/docs/matching-engine/configuring-indexes) in the Matching Engine utilities script.

While the index is being created and deployed, you can read more about Matching Engine's ANN service which uses a new type of vector quantization developed by Google Research: [Accelerating Large-Scale Inference with Anisotropic Vector Quantization](https://arxiv.org/abs/1908.10396).

For more information about how this works, see [Announcing ScaNN: Efficient
Vector Similarity Search](https://ai.googleblog.com/2020/07/announcing-scann-efficient-vector.html).
"""

mengine = MatchingEngineUtils(PROJECT_ID, ME_REGION, ME_INDEX_NAME)

logger.info('Started Index creation...')
index = mengine.create_index(
    embedding_gcs_uri=f'gs://{ME_EMBEDDING_DIR}/init_index',
    dimensions=ME_DIMENSIONS,
    index_update_method='streaming',
    index_algorithm='tree-ah',
)

if index:
    logger.info(index.name)
else:
    logger.info('Index creation still in progress...')

"""
Deploy index to Index Endpoint on Matching Engine. This [deploys the index to a public endpoint](https://cloud.google.com/vertex-ai/docs/matching-engine/deploy-index-public). The deployment operation creates a  public endpoint that will be used for querying the index for approximate nearest neighbors.

For deploying index to a Private Endpoint, refer to the [documentation](https://cloud.google.com/vertex-ai/docs/matching-engine/deploy-index-vpc) to set up pre-requisites.
"""

index_endpoint = mengine.deploy_index()
if index_endpoint:
    logger.info(f'Index endpoint resource name: {index_endpoint.name}')
    logger.info(f'Index endpoint public domain name: {index_endpoint.public_endpoint_domain_name}')
    logger.info('Deployed indexes on the index endpoint:')
    for d in index_endpoint.deployed_indexes:
        logger.info(f'    {d.id}')

"""
Add Document Embeddings to Matching Engine - Vector Store

This step ingests and parse PDF documents, split them, generate embeddings and add the embeddings to the vector store. The document corpus used as dataset is a sample of Google published research papers across different domains - large models, traffic simulation, productivity etc.

### Ingest PDF files

The document corpus is hosted on Cloud Storage bucket (at `gs://github-repo/documents/google-research-pdfs/`) and LangChain provides a convenient document loader [`GCSDirectoryLoader`](https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/google_cloud_storage_directory.html) to load documents from a Cloud Storage bucket. The loader uses `Unstructured` package to load files of many types including pdfs, images, html and more.

Ingest PDF files
"""

logger.info(f'Processing documents from {GCS_BUCKET_DOCS}')
loader = GCSDirectoryLoader(
    project_name=PROJECT_ID, bucket=GCS_BUCKET_DOCS, prefix=''
)
documents = loader.load()

# Add document name and source to the metadata
for document in documents:
    doc_md = document.metadata
    document_name = doc_md['source'].split('/')[-1]
    # derive doc source from Document loader
    doc_source_prefix = '/'.join(GCS_BUCKET_DOCS.split('/')[:3])
    doc_source_suffix = '/'.join(doc_md['source'].split('/')[4:-1])
    source = f'{doc_source_prefix}/{doc_source_suffix}'
    document.metadata = {'source': source, 'document_name': document_name}

logger.info(f'# of documents loaded (pre-chunking) = {len(documents)}')

# Verify document metadata
documents[0].metadata

"""
Chunk documents
Split the documents to smaller chunks. When splitting the document, ensure a few chunks can fit within the context length of LLM.
"""
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50,
    separators=['\n\n', '\n', '.', '!', '?', ',', ' ', ''],
)
doc_splits = text_splitter.split_documents(documents)

# Add chunk number to metadata
for idx, split in enumerate(doc_splits):
    split.metadata['chunk'] = idx

logger.info(f'# of documents = {len(doc_splits)}')

doc_splits[0].metadata

# Configure Matching Engine as Vector Store. Get Matching Engine Index id and Endpoint id

ME_INDEX_ID, ME_INDEX_ENDPOINT_ID = mengine.get_index_and_endpoint()
logger.info(f'ME_INDEX_ID={ME_INDEX_ID}')
logger.info(f'ME_INDEX_ENDPOINT_ID={ME_INDEX_ENDPOINT_ID}')

# Initialize Matching Engine vector store with text embeddings model
me = MatchingEngine.from_components(
    project_id=PROJECT_ID,
    region=ME_REGION,
    gcs_bucket_name=f'gs://{ME_EMBEDDING_DIR}'.split('/')[2],
    embedding=embeddings,
    index_id=ME_INDEX_ID,
    endpoint_id=ME_INDEX_ENDPOINT_ID,
)

"""
Add documents as embeddings in Matching Engine as index

The document chunks are transformed as embeddings (vectors) using Vertex AI Embeddings API and added to the index with **[streaming index update](https://cloud.google.com/vertex-ai/docs/matching-engine/create-manage-index#create-index)**. With Streaming Updates, you can update and query your index within a few seconds.

The original document text is stored on Cloud Storage bucket had referenced by id.

Prepare text and metadata to be added to the vectors
"""

# Store docs as embeddings in Matching Engine index
# It may take a while since API is rate limited
texts = [doc.page_content for doc in doc_splits]
metadatas = [
    [
        {'namespace': 'source', 'allow_list': [doc.metadata['source']]},
        {'namespace': 'document_name', 'allow_list': [doc.metadata['document_name']]},
        {'namespace': 'chunk', 'allow_list': [str(doc.metadata['chunk'])]},
    ]
    for doc in doc_splits
]

# Add embeddings to the vector store
# Depending on the volume and size of documents, this step may take time.

doc_ids = me.add_texts(texts=texts, metadatas=metadatas)

# Validate semantic search with Matching Engine is working
me.similarity_search('List all people with Java skills?', k=2)
me.similarity_search('Who is the CTO of Qarik Group?', k=2)
