from PyPDF2 import PdfReader

# creating a pdf file object
# pdfObject = open('./data/obama-worlds-matter.pdf', 'rb')

# # creating a pdf reader object
# pdfReader = PdfFileReader(pdfObject)

# # Extract and concatenate each page's content
# text=''
# for i in range(0,pdfReader.numPages):
#     # creating a page object
#     pageObject = pdfReader.getPage(i)
#     # extracting text from page
#     text += pageObject.extractText()
# print(text)
import openai
import re
import logging
import json

class resume_to_structured() :
    def __init__(self, OPENAI_API_KEY) :
        openai.api_key = OPENAI_API_KEY
        self.prompt_questions = \
"""Summarize the text below into a JSON with exactly the following structure {basic_info: {first_name, last_name, full_name, email, phone_number, location, portfolio_website_url, linkedin_url, github_main_page_url, university, education_level (BS, MS, or PhD), graduation_year, graduation_month, majors, GPA}, work_experience: [{job_title, company, location, duration, job_summary}], project_experience:[{project_name, project_discription}]}
"""

        logging.basicConfig(filename = 'logs/parser.log', level = logging.DEBUG)
        self.logger = logging.getLogger()

    # def pdf2txt(self: object, pdf_path: str) -> str :
    #     # creating a pdf file object
    #     pdfObject = open(pdf_path, 'rb')

    #     # creating a pdf reader object
    #     pdfReader = PdfReader(pdfObject)

    #     # Extract and concatenate each page's content
    #     text=''
    #     for i in range(0, len(pdfReader.pages)):
    #         # creating a page object
    #         pageObject = pdfReader.pages[i]
    #         # extracting text from page
    #         text += pageObject.extract_text()
    #     print(len(text))
    #     print(type(text))
    #     info = (text[:10000] + '..') if len(text) > 75 else text
    #     # Get PDF and return string of it.

    #     # with open(pdf_path, "rb") as f:
    #     #     pdf = PdfFileReader(f)
    #     # pdf_str = "\n\n".join(pdf)
    #     pdf_str = re.sub('\s[,.]', ',', info)
    #     pdf_str = re.sub('[\n]+', '\n', pdf_str)
    #     pdf_str = re.sub('[\s]+', ' ', pdf_str)
    #     pdf_str = re.sub('http[s]?(://)?', '', pdf_str)
    #     return info

    def convertion(self: object,
                        prompt: str,
                        engine: str = 'text-davinci-003',
                        temperature: float = 0.0,
                        max_tokens: int = 100,
                        top_p: int = 1,
                        frequency_penalty: int = 0,
                        presence_penalty: int = 0) -> object :

        self.logger.info(f'convertion: using {engine}')
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
    
    def ending_process(self: object, JD) -> dict :

        # Get PDF resume and return JSON FILE for resume

        resume = {}
        # str_resume = self.pdf2txt(resume_path)
        prompt = self.prompt_questions + '\n' + JD

        max_tokens = 1500
        engine = 'text-davinci-002'
        response = self.convertion(prompt, engine = engine, max_tokens = max_tokens)
        response_text = response['choices'][0]['text'].strip()
        print("============================")
        print(response)
        resume = json.loads(response_text)

        return resume
