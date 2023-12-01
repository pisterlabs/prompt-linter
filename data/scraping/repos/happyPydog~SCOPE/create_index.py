"""Create the index for project."""
from langchain import FAISS
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.schema import Document
from rich import print
from rich.progress import track

from ai import summary_code, tag_code
from file import get_file_paths
from python_parser import analyze_code

PROJET_DIR = "./MetaGPT/metagpt"
SUPPORTED_FILE_TYPES = [".py"]
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    file_paths = get_file_paths(PROJET_DIR, SUPPORTED_FILE_TYPES)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    documents = []
    for path in track(file_paths):
        with open(path, "r") as f:
            source = f.read()

        result = analyze_code(source)

        if result["modules"]:
            print(result["modules"])

        if result["functions_called"]:
            print(result["functions_called"])

        if result["global_vars"]:
            print(result["global_vars"])

        if result["function_codes"]:
            print(result["function_codes"])

        if result["class_codes"]:
            print(result["class_codes"])
            for class_name, code in result["class_codes"].items():
                summary = summary_code(code)
                print(f"💬 {summary = }")
                description = tag_code(code)
                print(f"🏷️  {description = }")

                # create document
                metadata = {
                    "file_path": path,
                    "class_name": class_name,
                    "summary": summary,
                    "code": code,
                }
                document = Document(page_content=description, metadata=metadata)
                documents.append(document)

    # create index
    db = FAISS.from_documents(documents, embeddings)
    db.save_local("vector_store")


if __name__ == "__main__":
    main()
