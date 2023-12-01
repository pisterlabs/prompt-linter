import fitz

import torch
from transformers import pipeline
from transformers import GenerationConfig

from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema.document import Document

from .llama2_prompts import llama2_template, llama2_prompt_ending_words


# hard code the model names for the demo application.
DEFAULT_EMBED_MODEL = 'sentence-transformers/all-mpnet-base-v2'
DEFAULT_MODEL = 'meta-llama/Llama-2-7b-chat-hf'


def load_llm() -> (pipeline, PromptTemplate, str):
    """
    load the model using the transformers pipeline.
    """
    gen_config = GenerationConfig.from_pretrained(DEFAULT_MODEL)
    gen_config.max_new_tokens = 4096
    gen_config.temperature = 1e-5
    
    llm = pipeline(
        task="text-generation",
        model=DEFAULT_MODEL,
        torch_dtype=torch.float16,
        generation_config=gen_config,
        device_map='auto',
    )
    
    # load the corresponding prompt template
    llama2_prompt = PromptTemplate.from_template(llama2_template)
    
    return llm, llama2_prompt, llama2_prompt_ending_words


def load_emb() -> HuggingFaceEmbeddings:
    """
    load the embedding model.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return HuggingFaceEmbeddings(model_name=DEFAULT_EMBED_MODEL,
                                 model_kwargs={'device': device}
                                )


def split_pdf(pdf_bytes: bytes, filename: str = None, min_chunk_length: int = 1000) -> list[Document]:
    """Split a PDF file into chunks. Each chunk consists of one or more PDF blocks. Chunk's meta data
    contain the starting page number and each block's page number and bounding box.

    Parameters:
        pdf_bytes: the pdf file in bytes.
        filename: the name of the uploaded file.
        min_chunk_length: the min length of a chunk, default to 1000 chars.

    Return:
        a list of `langchain.schema.document.Document`.
    """
    pdf_doc = fitz.open("pdf", pdf_bytes)
    documents = []

    chunk_buffer = {}
    for i, page in enumerate(pdf_doc):
        blocks = page.get_text('blocks')

        for block in blocks:
            page_content = block[4]
            metadata = {
                'source': filename if filename else 'dummy name', # not used in the current impl.
                'page': i,
                'bbox': ','.join(map(str, [i] + list(block[:4]))), # langChain only allows str, int or float for meta data values
            }

            if chunk_buffer.get('page_content'):
                chunk_buffer['page_content'] = '\n\n'.join([chunk_buffer['page_content'], page_content])
                chunk_buffer['metadata']['bbox'] = '\n'.join(
                    [chunk_buffer['metadata']['bbox'], metadata['bbox']]
                )
            else:
                chunk_buffer['page_content'] = page_content
                chunk_buffer['metadata'] = metadata

            # create a document once the chunk is longer enough.
            if len(chunk_buffer['page_content']) >= min_chunk_length:
                document = Document(page_content=chunk_buffer['page_content'],
                                    metadata=chunk_buffer['metadata'])
                documents.append(document)
                chunk_buffer = {}

        # convert the remaining in the chunk buffer to a document
        if chunk_buffer.get('page_content'):
            document = Document(page_content=chunk_buffer['page_content'], metadata=chunk_buffer['metadata'])
            documents.append(document)

    return documents


class PDFChatBot:
    def __init__(self) -> None:
        self.embedding = None
        self.llm = None
        self.prompt =None
        self.prompt_ending_words = None
        self.vectordb = None


    def load_vectordb(self, docs: list[Document]) -> None:
        """create a vector db from input documents

        Parameters:
            docs: a list of LangChain documents.
        """
        self._docs = docs

        # use Chroma as vector db.
        # no need to persist the database
        self.vectordb = Chroma.from_documents(
            documents=self._docs,
            embedding=self.embedding,
            persist_directory=None
        )


    def search(self, query: str, k: int = 5) -> (str, list[Document]):
        """This method searches the vector database and concats search results as context. The
        context and query are sent to LLM to get answer.

        Parameters:
            query: input question.
            k: the number of returned documents (default to 5).

        Return:
            a tuple of summary, found relevant documents, summaries for each document.
        """
        # retrieve relevant chunks from the vector database
        src_docs = self.vectordb.similarity_search(query, k=k)

        # concatenate chunks together to be used as context
        texts = [doc.page_content for doc in src_docs]
        ctx = '\n\n'.join(texts)
        
        seqs = self.llm(
            self.prompt.format(context=ctx, question=query),
            do_sample=True,        
            num_return_sequences=1,
        )
        answer = seqs[0]['generated_text']

        # need to remove the prompt from the generated text
        idx_start = answer.index(self.prompt_ending_words) + len(self.prompt_ending_words)
        answer = answer[idx_start:].strip()
        
        return answer, src_docs
