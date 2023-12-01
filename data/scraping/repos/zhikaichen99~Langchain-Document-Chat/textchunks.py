from langchain.text_splitter import CharacterTextSplitter

def create_text_chunks(text):
    """
    This function takes the extracted text from the documents and splits it into smaller text chunks
    The resulting text chunks are returned

    Inputs:
        text - the text extracted from documents
    Outputs:
        chunks - a list containing the chunks of texts
    """
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )

    chunks = text_splitter.split_text(text)
    return chunks
