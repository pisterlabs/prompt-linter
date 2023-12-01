from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


class TextProcessor():
    def __init__(self, ck_size, ck_overlap_window):
        self.ck_size = ck_size
        self.ck_overlap_window = ck_overlap_window

    def load_data(self, fpath):
        with open(fpath) as f:
            return f.readlines()

    def split_data(self, document_list):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.ck_size,
            chunk_overlap=self.ck_overlap_window,
        )
        return text_splitter.create_documents(document_list)

    def display_info(self, data):
        print(f'There are {len(data)} documents in your data. ')
        print(
            f'There are {len(data[0].page_content)} characters in your first document. ')

    def create_docs(self, datalist, fpath):
        return [Document(page_content=text, metadata={"source": f'{fpath}_{i}'}) for i, text in enumerate(datalist)]


if __name__ == "__main__":
    test_fpath = '../dataset/bohrium_qa_demo.txt'
    processor = TextProcessor(1000, 10)
    datalist = processor.load_data(test_fpath)
    res_data = processor.create_docs(datalist, test_fpath)
    print(res_data[1])
