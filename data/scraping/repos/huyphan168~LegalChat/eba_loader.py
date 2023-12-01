import pickle
from langchain.document_loaders.base import BaseLoader
from langchain.docstore.document import Document
from pathlib import Path
from typing import List
import pandas as pd
import re

class EBALoader(BaseLoader):
    """Loader that loads EBAs documentation"""

    def __init__(self, path: str):
        """Initialize path."""
        self.file_path = path

    def load(self) -> List[Document]:
        """Load documents."""
        def _clean_data(data: str) -> str:
            text = re.sub(r"^\d+\. |\n", "", data)
            return text.strip()
        
        def _clean_topic(data: str) -> str:
            text = re.sub(r"^\d+ |\n", "", data)
            return text.strip()

        def _parsing_xlsx(file_excel_path: str):
            docs = []
            database = pd.read_excel(file_excel_path)
            database.columns = database.columns.str.lower()
            database['requirement_text'] = database['requirement_text'].apply(_clean_data)
            df_grouped = database.groupby(['level_1_text'])['requirement_text'].apply(lambda x: "%s" % ' '.join(x))
            df_grouped = df_grouped.reset_index()
            source_name = database['source'].unique()[0]
            for i in range(len(df_grouped)):
                document = df_grouped.iloc[i]
                text = document['requirement_text']
                level_1_text = document['level_1_text']
                meta_data = {'source': source_name, 'topic': _clean_topic(level_1_text)}
                docs.append(Document(page_content=text, metadata=meta_data))
            return docs
        
        def _parsing_python(file_python_path: str):
            docs = []
            with open(file_python_path, "r") as f:
                text = f.read()
            ##Seperating the code into functions
            function_regex = r"(def [a-zA-Z_]+\([^\)]*\):(?:\n|.)*?(?=def [a-zA-Z_]+\()|\Z)"
            functions = re.findall(function_regex, text, re.DOTALL)
            meta_data = {'source': file_python_path}
            for function in functions:
                text = function
                Document(page_content=text, metadata=meta_data)
                docs.append(Document(page_content=text, metadata=meta_data))
            return docs

        docs = []
        for p in Path(self.file_path).rglob("*"):
            if p.is_dir():
                continue
            if p.suffix == ".xlsx":
                docs_per_file = _parsing_xlsx(p)
            # if p.suffix == ".pdf":
            #     continue
            # if p.suffix == ".py":
            #     docs_per_file = _parsing_python(p)
            docs.extend(docs_per_file)
        return docs

if __name__ == "__main__":
    loader = EBALoader("eba_documents/")
    docs = loader.load()
    print(docs[0])