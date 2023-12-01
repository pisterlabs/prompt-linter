from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from io import BytesIO
from src.utils import fix_pdf_text


def ask_pdf_simple(question: str, pdf: BytesIO) -> str:
    if isinstance(pdf, BytesIO):
        # load doc
        doc = PdfReader(pdf)
        pages = [fix_pdf_text(page.extract_text()) for page in doc.pages]
        text = " ".join(pages)

        # split doc in chunks
        splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = splitter.split_text(text)

        # vectorize chunks
        embedding = OpenAIEmbeddings()
        vectordb = FAISS.from_texts(chunks, embedding)

        # perform semantic search
        top_k_chunks = vectordb.similarity_search(question, k=5)
        top_k_chunks = [chunk.page_content for chunk in top_k_chunks]
        context = "\n\n".join(top_k_chunks)

        # create prompt with top k context
        prompt = f"""
Your task is to answer a question given some context.

The context is given here, delimited by triple backticks:

```
{context}
```

This is the question: "{question}".

If you don't know the answer, just say that you don't know, don't try to make up an answer.
"""
        return prompt

    else:
        return ""

_title = "Ask a question to a PDF file"
_description = """It may fail, since I used an extremely simple RAG technique.

**Do not use the follow-up chat to ask more questions.**
Instead, re-write them in the `Question` Field and use `Launch Prompt` again."""