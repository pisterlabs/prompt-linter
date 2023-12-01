import arxiv
import os

from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator


class ResearchPaper:
    def __init__(self, paper_id, paper_directory='research-papers'):
        self.paper_id = paper_id
        self.paper_directory = paper_directory
        self.pdf_file_path = f'{paper_id}.pdf'
        self.txt_file_path = f'{paper_id}.txt'
        self.index = None

        if not os.path.exists(self.paper_directory):
            os.makedirs(self.paper_directory)

        # ensures that the folder does not contain more than 5 files (since we don't have a db)
        self.clear_folder_if_more_than_five_files()
        if not self.is_txt_file_exists():
            self.download()
            self.convert_pdf_to_txt()
        self.prepare_index()

    def download(self):
        search = arxiv.Search(id_list=[self.paper_id])
        paper = next(search.results())
        paper.download_pdf(dirpath=self.paper_directory,
                           filename=self.pdf_file_path)

    def convert_pdf_to_txt(self) -> str:
        loader = PyPDFLoader(os.path.join(
            self.paper_directory, self.pdf_file_path))
        pages = loader.load_and_split()
        text = ""

        for page in pages:
            text += page.page_content

        with open(os.path.join(self.paper_directory, self.txt_file_path), 'w') as f:
            f.write(text)
        return text

    def prepare_index(self):
        loader = TextLoader(os.path.join(
            self.paper_directory, self.txt_file_path))
        self.index = VectorstoreIndexCreator().from_loaders([loader])

    def query(self, q) -> str:
        response = self.index.query(q)
        return response

    def has_more_than_five_files(self) -> bool:
        files = os.listdir(self.paper_directory)
        return len(files) > 5

    def clear_folder_if_more_than_five_files(self):
        if self.has_more_than_five_files():
            files = os.listdir(self.paper_directory)
            for file in files:
                os.remove(os.path.join(self.paper_directory, file))

    def is_txt_file_exists(self) -> bool:
        return os.path.exists(os.path.join(self.paper_directory, self.txt_file_path))
