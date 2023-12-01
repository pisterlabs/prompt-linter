from langchain.document_loaders import HuggingFaceDatasetLoader
dataset_name = "imdb"
page_content_column = "text"


loader = HuggingFaceDatasetLoader(dataset_name, page_content_column)

data = loader.load()

print(data[:15])