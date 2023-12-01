from langchain.document_loaders import Docx2txtLoader


class WordParser():
    def __init__(self):
        self.module = "WP"

    def load_docs(self):
        loader = Docx2txtLoader("../data/FRD.docx")
        data = loader.load()

        return data


if __name__ == "__main__":
    wp = WordParser()
    print(wp.load_docs())
