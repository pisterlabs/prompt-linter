import pdftotext
import openai
import re
import logging
import json
import os

OPEN_API_KEY = ""

class ResumeParser():
    def __init__(self, OPENAI_API_KEY):
        # set GPT-3 API key from the environment vairable
        openai.api_key = OPENAI_API_KEY
        # GPT-3 completion questions
        self.prompt_questions = \
"""Resume le texte suivant en format JSON avec exactement la structure suivante {info_basique: {nom_complet, email, linkedin_url, université, niveau_education (Licence, Master, Doctorat), annee_de_diplomation, spécialité}, experience_professionnelle: [{titre_du_poste, entreprise, durée, résumé_de_lexperience}]}}]}
"""
       # set up this parser's logger
        logging.basicConfig(filename='logs/parser.log', level=logging.DEBUG)
        self.logger = logging.getLogger()

    def pdf2string(self: object, pdf_path: str) -> str:
        """
        Extract the content of a pdf file to string.
        :param pdf_path: Path to the PDF file.
        :return: PDF content string.
        """
        with open(pdf_path, "rb") as f:
            pdf = pdftotext.PDF(f)
        pdf_str = "\n\n".join(pdf)
        pdf_str = re.sub('\s[,.]', ',', pdf_str)
        pdf_str = re.sub('[\n]+', '\n', pdf_str)
        pdf_str = re.sub('[\s]+', ' ', pdf_str)
        pdf_str = re.sub('http[s]?(://)?', '', pdf_str)
        return pdf_str

    def query_completion(self: object,
                        prompt: str,
                        engine: str = 'text-curie-001',
                        temperature: float = 0.0,
                        max_tokens: int = 100,
                        top_p: int = 1,
                        frequency_penalty: int = 0,
                        presence_penalty: int = 0) -> object:
        """
        Base function for querying GPT-3. 
        Send a request to GPT-3 with the passed-in function parameters and return the response object.
        :param prompt: GPT-3 completion prompt.
        :param engine: The engine, or model, to generate completion.
        :param temperature: Controls the randomnesss. Lower means more deterministic.
        :param max_tokens: Maximum number of tokens to be used for prompt and completion combined.
        :param top_p: Controls diversity via nucleus sampling.
        :param frequency_penalty: How much to penalize new tokens based on their existence in text so far.
        :param presence_penalty: How much to penalize new tokens based on whether they appear in text so far.
        :return: GPT-3 response object
        """
        self.logger.info(f'query_completion: using {engine}')
        estimated_prompt_tokens = int(len(prompt.split()) * 1.6)
        self.logger.info(f'estimated prompt tokens: {estimated_prompt_tokens}')
        estimated_answer_tokens = 2049 - estimated_prompt_tokens
        if estimated_answer_tokens < max_tokens:
            self.logger.warning('estimated_answer_tokens lower than max_tokens, changing max_tokens to', estimated_answer_tokens)
        response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        temperature=temperature,
        max_tokens=min(4096-estimated_prompt_tokens, max_tokens),
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
        )
        return response
    
    def query_resume(self: object, pdf_path: str) -> dict:
        """
        Query GPT-3 for the work experience and / or basic information from the resume at the PDF file path.
        :param pdf_path: Path to the PDF file.
        :return dictionary of resume with keys (basic_info, work_experience).
        """
        resume = {}
        pdf_str = self.pdf2string(pdf_path)
        print(pdf_path)
        prompt = self.prompt_questions + '\n' + pdf_str
        max_tokens = 1500
        engine = 'text-davinci-003'
        response = self.query_completion(prompt,engine=engine,max_tokens=max_tokens)
        response_text = response['choices'][0]['text'].strip()
        return response_text


parser = ResumeParser(OPEN_API_KEY)

# Create a directory to store parsed CVs if it doesn't exist
if not os.path.exists('CV_parsed'):
    os.makedirs('CV_parsed')

import glob

# Get all the PDF files in the CV_to_study directory
cv_files = glob.glob("CV_to_study/*.PDF")

for cv_file in cv_files:
    # Get the filename without the extension
    file_name = os.path.basename(cv_file).split('.')[0]

    # Parse the resume
    parsed_resume = parser.query_resume(cv_file)

    # Save the parsed resume in the CV_parsed directory with the same name but with a ".txt" extension
    with open(f"CV_parsed/{file_name}.txt", "w") as output_file:
        output_file.write(parsed_resume)