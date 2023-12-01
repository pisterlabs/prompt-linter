import os
import openai
from WebScraping import file_uploader
from interface import Routine
from prompting import build_prompt


openai.api_key = os.getenv("OPENAI_KEY")

class Medical(Routine):
    '''
    Medical question handling routine.
    '''

    @staticmethod
    def process(inp, history):
        '''
        Processes the question as defined in Routine assuming it's medical.
        '''
        response = openai.Answer.create(
            search_model="ada",
            model="curie",
            question= build_prompt(inp, history),
            file=file_uploader.fetch_id(),
            examples_context="Question: Who is most at risk of severe illness from COVID-19? Answer: People aged 60 years and over, and those with underlying medical problems like high blood pressure, heart and lung problems, diabetes, obesity or cancer, are at higher risk of developing serious illness. Hepatitis A is an inflammation of the liver caused by the hepatitis A virus (HAV). The virus is primarily spread when an uninfected (and unvaccinated) person ingests food or water that is contaminated with the faeces of an infected person. Epilepsy is not contagious. Although many underlying disease mechanisms can lead to epilepsy, the cause of the disease is still unknown in about 50% of cases globally.",
            examples=[["Friend: What has the most risk from getting COVID?",
                       "People aged 60 years and over, and those with underlying medical problems like high blood pressure, heart and lung problems, diabetes, obesity or cancer."],
                      ["Friend: What is hepatitis?",
                       "Hepatitis A is an inflammation of the liver caused by the hepatitis A virus (HAV). The virus is primarily spread when an uninfected (and unvaccinated) person ingests food or water that is contaminated with the faeces of an infected person."],
                      ["Friend: YOOOO CAN I SPREAD EPILEPSY TO OTHERS??????",
                       "No, you cannot spread epilepsy to others as epilepsy is not contagious. Although many underlying disease mechanisms can lead to epilepsy, the cause of the disease is still unknown in about half of the cases globally."],
                      ['Friend: HEEEEY CAN I DO SOMETHING TO AVOID GETTING A BURULI ULCER? PLEASEEE',
                       'There are currently no primary preventive measures for Buruli ulcer. The mode of transmission is not known.']],
            max_rerank=10,
            max_tokens=200,
            stop=["\n", "<|endoftext|>"]
        )

        return response["answers"][0]
