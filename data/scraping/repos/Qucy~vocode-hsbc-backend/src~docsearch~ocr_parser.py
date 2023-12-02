from azure.ai.formrecognizer import DocumentAnalysisClient
from langchain.docstore.document import Document


def extract_text_from_img(
    file: str | bytes, document_analysis_client: DocumentAnalysisClient
) -> list[Document]:
    """Extract text from file using Azure Form Recogniser.
    :param file: File path (str) or BytesIO object
    :returns: List of Document objects
    """

    # if string; then extract bytes from file first
    if isinstance(file, str):
        with open(file, "rb") as f:
            file = bytearray(f.read())

    # submit file to Azure Form Recogniser
    poller = document_analysis_client.begin_analyze_document(
        "prebuilt-document", document=file
    )

    result = poller.result()

    # Extract text from results; one Document per page
    output = []
    for page in result.pages:
        text = ""
        for line in page.lines:
            text += line.content
            text += "\n"
        output.append(Document(page_content=text))

    return output
