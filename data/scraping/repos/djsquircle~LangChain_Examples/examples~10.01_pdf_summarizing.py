import os
from dotenv import load_dotenv
from langchain import OpenAI, PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from typing import Optional


def summarize_pdf(file_path: str, model_name: Optional[str] = "text-davinci-003", temperature: Optional[float] = 0) -> str:
    """
    Summarize the content of a PDF file.

    Args:
        file_path: The path to the PDF file to be summarized.
        model_name: The name of the OpenAI model to use for summarization.
        temperature: The temperature parameter controlling the randomness of the output.

    Returns:
        A summary of the PDF content as a string.
    """

    # Load environment variables if a .env file exists
    if os.path.exists(".env"):
        load_dotenv()

    # Initialize language model
    llm = OpenAI(model_name=model_name, temperature=temperature)

    # Load the summarization chain
    summarize_chain = load_summarize_chain(llm)

    # Load the document using PyPDFLoader
    document_loader = PyPDFLoader(file_path=file_path)
    document = document_loader.load()

    # Summarize the document
    summary = summarize_chain(document)

    # Return the summary text
    return summary['output_text']


# Example usage if this script is executed as the main program
if __name__ == "__main__":
    file_path = "./DAOGEN_PDF/LunaVega.pdf"
    summary = summarize_pdf(file_path)
    print(summary)

#from your_module import summarize_pdf
#summary = summarize_pdf('./path_to_pdf/document.pdf')
#print(summary)
#Make sure to replace your_module with the actual name of the Python script or module where the function is defined (without the .py extension).

