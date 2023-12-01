from typing import List
from llama_index.llms import OpenAI
from llama_index.schema import Document, TextNode
from llama_index.langchain_helpers.text_splitter import SentenceSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from llama_index import ServiceContext
from llama_index import LangchainEmbedding
from llama_index.node_parser import SimpleNodeParser
from llama_index import VectorStoreIndex
from llama_index.node_parser.extractors import (
    MetadataExtractor,
    SummaryExtractor,
    KeywordExtractor,
)

from clean_sample_ho3 import clean_sample_ho3_pages


def identify_original_documents(documents: List[Document], nodes: List[TextNode]) -> List[TextNode]:
    """
    This function identifies page numbers from a LlamaIndex Document list and adds them back
        to parsed TextNode's (aka chunks) metadata.
    We are splitting this way so that:
        * Chunks ignore page breaks, but respect sentence boundry
        * We maximize token by reducing unnessesary whitespace and line breaks
        * Normalizing chunk size 
    Args:
    documents: A list of LlamaIndex Document objects.
    chunks: A list of LlamaIndex TextNode objects parsed from the 'documents'.
    Returns:
    chunks: The provided chunks with appended page info.
    """
    # Combine all documents into a single string and track their ranges.
    long_string = ""
    pages_dict = {}
    for i, doc in enumerate(documents):
        start = len(long_string)
        long_string += doc.text
        end = len(long_string)
        pages_dict[i] = (start, end)

    # Identify the original document page numbers.
    for i, node in enumerate(nodes):
        chunk_start = long_string.find(node.text)
        chunk_end = chunk_start + len(node.text)
        pages = []
        for doc_number, (doc_start, doc_end) in pages_dict.items():
            if (chunk_start >= doc_start and chunk_start < doc_end) or (chunk_end > doc_start and chunk_end <= doc_end):
                pages.append(doc_number + 1)
        # Update existing metadata dictionary
        node.metadata.update({'pages': '-'.join(map(str, pages))})

    return nodes


def build_ho3_sample_policy_index(sample_ho3_policy_docs: List[Document], llm_kw_extract: bool):
    # Apply cleaning function to each Document, i.e., page
    for i, _ in enumerate(sample_ho3_policy_docs):
        sample_ho3_policy_docs[i].text = clean_sample_ho3_pages(sample_ho3_policy_docs[i].text)
        
    # Normalize the text by converting pdf pages into a singe string
    docs = [Document(text="".join(sample_ho3_policy_docs[i].text for i in range(len(sample_ho3_policy_docs))),
                        id_="HO3_sample.pdf",
                        metadata={"source": "https://www.iii.org/sites/default/files/docs/pdf/HO3_sample.pdf",
                                "Document Name": "HO3 Sample Policy", 
                                "Category": "Homeowner's Insurance Policy"},
                        excluded_llm_metadata_keys=['source', 'Document Name'],
                        excluded_embed_metadata_keys=['source'],
                        )]
    
    # Split priority: "\n\n\n" -> "\n" -> " " 
    text_splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=0,
    )
    
    if llm_kw_extract:
        metadata_extractor = MetadataExtractor(
            extractors=[
            KeywordExtractor(keywords=5),
            ],
        )
        node_parser = SimpleNodeParser(
            text_splitter=text_splitter,
            metadata_extractor=metadata_extractor,
        )
    else:
        node_parser = SimpleNodeParser(
        text_splitter=text_splitter,
        )
    
    # Parse text from the prepped Document
    ho3_nodes = node_parser.get_nodes_from_documents(docs)
    
    # Add page number(s) to metadata, e.g., pages 2-3
    nodes_with_pages = identify_original_documents(sample_ho3_policy_docs, ho3_nodes)
    
    # Add human readable id
    new_nodes = []
    for node in nodes_with_pages:
        node.id_ = f"{node.metadata['Document Name']} | Pages: {node.metadata['pages']}"
        new_nodes.append(node)
    
    # Build vector store index
    llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
    embed_model = LangchainEmbedding(OpenAIEmbeddings())
        
    index = VectorStoreIndex(
        new_nodes, 
        service_context=ServiceContext.from_defaults(
            llm=llm,
            node_parser=node_parser,
            embed_model=embed_model
            )
        )
    
    return index




