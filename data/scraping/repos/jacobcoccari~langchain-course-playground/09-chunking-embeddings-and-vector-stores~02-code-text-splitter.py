from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language,
)

# You can also see the separators used for a given language
# separators = RecursiveCharacterTextSplitter.get_separators_for_language(Language.PYTHON)
# print(separators)

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
print(python_docs)
