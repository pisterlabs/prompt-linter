"""
https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/code_splitter

切分代码
"""

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language,
)

for e in Language:
    print(e)

# You can also see the separators used for a given language
splits = RecursiveCharacterTextSplitter.get_separators_for_language(Language.PYTHON)
print("python splits,", splits)

"""
python
"""

PYTHON_CODE = """
def hello_world():
    print("Hello, World!")

# Call the function
hello_world()
"""
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=50, chunk_overlap=0
)
python_docs = python_splitter.create_documents([PYTHON_CODE])
print("python_docs\n", python_docs)

"""
余下看官网
"""