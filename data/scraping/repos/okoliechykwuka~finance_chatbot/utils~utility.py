import os
import pandas as pd

class ExcelLoader():
    def __init__(self, file):
        import pandas as pd
        self.status = False
        self.name =  'ExcelLoader'
        self.file = file
        self.loader = pd.ExcelFile
        self.ext = ['xlsx']
    
    def load(self):
        from langchain.document_loaders.csv_loader import CSVLoader

        ssheet = self.loader(self.file)
        try:
            os.mkdir('temp')

        except FileExistsError:
            pass
        docs = []
        for i,sheet in enumerate(ssheet.sheet_names):
            df = ssheet.parse(sheet)
            temp_path = f'./temp/{sheet}.csv'
            docs.append(temp_path)
            df.to_csv(temp_path, index=False)
        return docs

def process_csv_file(file):
    file_paths = []
    if file.split('.')[-1] == 'csv':
        file_paths.append(file)
    elif file.split('.')[-1] == 'xlsx':
        loader = ExcelLoader(file)
        paths = loader.load()
        file_paths.extend(paths)
    if len(file_paths) == 1:
        return file_paths[0]
    return file_paths


# def get_loader(file_type):
#     import tempfile

#     if file_type == "text/plain":
#         Loader = TextLoader
#     elif file_type == "application/pdf":
#         Loader = PyPDFLoader
#     elif file_type == "text/csv":
#         Loader = CSVLoader
#         csv_files.append(file)

#     elif file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
#         Loader = ExcelLoader

#     else:
#         raise ValueError('File type is not supported')