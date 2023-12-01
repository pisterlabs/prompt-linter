from typing import Dict, List
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from tools import *
from paper_class import Paper
from document_source import DocumentSource


class PaperSource:
    def __init__(self, 
                 papers: Dict[str, Paper], 
                 openai_api_key: str,
                 ignore_references: bool = True):
        """
        Initializes a PaperSource object with a dictionary of papers and an OpenAI API key.

        Args:
            papers (Dict[str, Paper]): A dictionary containing paper titles as keys and object of class Paper as values.
            openai_api_key (str): The OpenAI API key for text embeddings.
            ignore_references (bool): Whether to ignore the chunks containing references.
        """
        if len(papers) == 0:
            raise ValueError("No papers was provided.")

        self.ignore_references_ = ignore_references
        self.papers_: Dict[str, Paper] = papers
        self.document_source_ = DocumentSource(
            openai_api_key=openai_api_key,
        )
        for title, paper in papers.items():
            docs = self._process_pdf(paper)  # Extract the PDF into chunks and append them to the doc_list.
            self.document_source_.add_documents(docs)

    def papers(self) -> Dict[str, Paper]:
        """
        Returns the dictionary of papers.

        Returns:
            Dict[str, Paper]: A dictionary containing paper titles as keys and Paper objects as values.
        """
        return self.papers_

    def _process_pdf(self, paper: Paper) -> List[Document]:
        """
        Download a PDF, extract its content, and split it into text chunks.

        Args:
            paper (Paper): A Paper object representing the paper to be processed.

        Returns:
            List[Document]: A list of Document objects, each containing a text chunk with metadata.
        """
        # Download the PDF and obtain the file path.
        pdf_path = paper.download()
        print(f"Loading PDF: {pdf_path}")
        
        # Load the PDF content.
        loader = PyPDFLoader(pdf_path)
        pdf = loader.load()
        print(f"Extracting & splitting text from paper: {paper.title}")
        
        # Initialize a text splitter.
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        
        # Split the PDF into text chunks (list of Document objects).
        docs = text_splitter.split_documents(pdf)
        
        # Assign the same title to each chunk.
        doc_list = []
        for doc in docs:
            # Filter out reference sections if choose to ignore them.
            if self.ignore_references_ and contains_arxiv_reference(doc.page_content):
                print('The reference section is skipped.')
                continue
            doc.metadata['source'] = paper.title
            doc_list.append(doc)
        
        return doc_list

    def retrieve(self, **kwargs) -> List[Document]:
        """
        Search for papers related to a query using text embeddings and cosine distance.

        Args:
            query (str): The query string to search for related papers.

        Returns:
            List[Document]: A list of Document objects representing the related papers found.
        """
        return self.document_source_.retrieve(**kwargs)
    

if __name__ == '__main__':
    from test_utils import get_test_papers
    import os
    
    paper_source = PaperSource(
        papers=get_test_papers(),
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        ignore_references=True,
    )
    
    print(paper_source.retrieve(
        query='test',
        num_retrieval=5,
        score_threshold=0.4,
    ))
