from langchain.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(file_path='./example.csv', source_column='Timecode')

data = loader.load()

print(len(data))