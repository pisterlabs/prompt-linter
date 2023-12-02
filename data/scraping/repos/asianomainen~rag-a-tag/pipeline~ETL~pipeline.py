import os
import glob
import pypdf
import weaviate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

WEAVIATE_DB_URL = os.getenv("WEAVIATE_DB_URL")
WEAVIATE_APIKEY = os.getenv("WEAVIATE_APIKEY")
OPEN_AI_APIKEY = os.getenv("OPEN_AI_APIKEY")
PDF_FILES_PATH = "pipeline/ETL/pdf_files/*"

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=120)
client = weaviate.Client(
    url=WEAVIATE_DB_URL,
    auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_APIKEY),
    additional_headers={"X-OpenAI-Api-Key": OPEN_AI_APIKEY}
)


def process_pdf(file_path):
    data = []

    with open(file_path, "rb") as file:
        reader = pypdf.PdfReader(file)
        for i, page in enumerate(reader.pages):
            texts = text_splitter.split_text(page.extract_text())
            for j, chunk in enumerate(texts):
                data.append({
                    "title": os.path.basename(file_path),
                    "content": chunk.replace("-\n", "").replace(".\n", ". ").replace("\n", " "),
                    "page": i,
                    "chunk": j
                })

    return data


def import_to_weaviate(data_objects):
    class_obj = {
        "class": "ResearchPaper",
        "vectorizer": "text2vec-openai",
        "moduleConfig": {
            "text2vec-openai": {},
            "generative-openai": {}
        }
    }

    client.schema.delete_class("ResearchPaper")  # to delete the class
    client.schema.create_class(class_obj)  # use this if "ResearchPaper" class is not created yet

    client.batch.configure(batch_size=100)
    with client.batch as batch:  # Initialize a batch process
        for i, data_object in enumerate(data_objects):  # Batch import data
            print(f"Importing chunk {i}")
            properties = {
                "title": data_object["title"],
                "content": data_object["content"],
                "chunk_index": f'{data_object["title"]}, page {data_object["page"]}, chunk {data_object["chunk"]}',
                "page": data_object["page"]
            }
            batch.add_data_object(
                data_object=properties,
                class_name="ResearchPaper"
            )


def main():
    pdf_files = glob.glob(PDF_FILES_PATH)
    all_data = []

    for pdf_file in pdf_files:
        all_data.extend(process_pdf(pdf_file))

    # UNCOMMENT LINE BELOW TO RE-UPLOAD PDFs TO WEAVIATE. WARNING: OVERWRITES ALL EXISTING FILES.
    # import_to_weaviate(all_data)


if __name__ == "__main__":
    main()
