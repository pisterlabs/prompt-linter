import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter

from main import weaviate_client

SCHEMA_PATH = "./schema/medical_docs.json"


class PDFHandler:
    def __init__(self, pdf_file, is_recreate=True):
        self.pdf_file = pdf_file

        with fitz.open(stream=self.pdf_file.read(), filetype="pdf") as doc:
            self.text = ""
            for page in doc:
                self.text += page.get_text()

        if is_recreate:
            weaviate_client.schema.delete_all()
        try:
            weaviate_client.schema.create_class(SCHEMA_PATH)
        except Exception as e:
            print(e)

    def split_text(self):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = splitter.split_text(text=self.text)

        return chunks

    def upload_to_weaviate(self):
        chunks = self.split_text()

        with weaviate_client.batch(batch_size=100, timeout_retries=3) as batch:
            for i, d in enumerate(chunks):
                print(f"importing entry: {i}")
                properties = {
                    "text": d,
                }

                batch.add_data_object(
                    properties,
                    "MedicalDocs",
                )

        print("Import completed!")
