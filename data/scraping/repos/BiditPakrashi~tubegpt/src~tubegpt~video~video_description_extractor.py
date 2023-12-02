from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class VideoDescriptionExtractor():

    def __init__(self) -> None:
        pass

    def load_from_file(self,file_path):
        loader = TextLoader(file_path)
        docs_vision_description = loader.load()
        text_splitter_vision_description = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
        texts_vision_description = text_splitter_vision_description.split_documents(docs_vision_description)
        return texts_vision_description


