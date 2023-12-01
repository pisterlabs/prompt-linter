"""
Extract all components from resume
using pdfplumber and gpt
"""

# required imports
import openai
import pdfplumber

# import constants
from config import (
    RESUME_PROMPT, 
    OPENAI_MODELS
)


class ResumeExtraction:

    """
    using a hyperparameter tuned model,
    extract contents from a PDF file
    """

    def __init__(self, resume_file):
        self.pdf = pdfplumber.open(resume_file)

    def process_resume(self):

        """
        read contents of a pdf file
        and perform extraction using 
        gpt model

        return: dictionary of results
        """

        pdf_content = ""
        extraction_result = {}

        try:
            # read contents of pdf file
            for page in self.pdf.pages:
                pdf_content += page.extract_text(layout=True)
            pdf_content = pdf_content.strip()
            prompt = RESUME_PROMPT + pdf_content
            
            # get response from gpt
            gpt_response = openai.Completion.create(
                model = OPENAI_MODELS.get("extraction_model"),
                prompt = prompt,
                temperature = 0.2,
                max_tokens = 100
            )
            extraction_result = gpt_response["choices"][0]["text"]
        except Exception as error:
            print(f"process_resume :: Exception :: {str(error)}")
        
        return extraction_result
