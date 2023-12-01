import dotenv
import PyPDF2
import requests

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import OnlinePDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

dotenv.load_dotenv()


def download_and_extract_pdf(pdf_url):
    downloaded_pdf_path = "downloaded_pdf.pdf"
    if os.isfile(pdf_url):
        downloaded_pdf_path = pdf_url
    else:
        response = requests.get(pdf_url)
        response.raise_for_status()  # Check for HTTP errors
        with open(downloaded_pdf_path, "wb") as pdf_file:
            pdf_file.write(response.content)
    try:
        pdf_text = ""
        with open(downloaded_pdf_path, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                pdf_text += page.extract_text()

        return pdf_text
    except requests.exceptions.RequestException as e:
        print(f"Error downloading PDF: {e}")
        return None
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return None


def main():
    # Initialize the ChatOpenAI model
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        verbose=True,
    )

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

    # Define the QA prompt template outside the loop
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    # Initialize the QA chain outside the loop
    qa_chain = None

    while True:
        # Prompt the user for the PDF URL
        url = input("Enter the URL of the PDF (or 'q' to quit): ")
        if url.lower() == "q":
            break

        # Download and extract text from the PDF
        # pdf_text = download_and_extract_pdf(pdf_url)
        # if pdf_text is None:
        #    continue

        # Split the PDF text into chunks

        loader = OnlinePDFLoader(url)
        data = loader.load()
        all_splits = text_splitter.split_documents(data)
        # Create embeddings and vectorstore
        vectorstore = Chroma.from_documents(
            documents=all_splits, embedding=OpenAIEmbeddings()
        )

        # Initialize the QA chain if not already initialized
        if qa_chain is None:
            qa_chain = RetrievalQA.from_chain_type(
                llm,
                retriever=vectorstore.as_retriever(),
                chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
            )

        while True:
            # Prompt the user for a question
            question = input("Ask a question (or 'q' to change PDF): ")
            if question.lower() == "q":
                break

            # Perform question-answering
            answer = qa_chain(dict(query=question))

            # Display the answer
            print("Answer:", answer)


if __name__ == "__main__":
    main()
