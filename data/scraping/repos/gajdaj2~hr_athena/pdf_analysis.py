import PyPDF2
import openai
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader

from dotenv import load_dotenv

load_dotenv()


def summary_template():
    return """
    given the information {information} about person  from I want you to create:
        1. a short summary of the person
        2. 5 Technology what he know
        3. Two interesting facts about the person
        4. Two interview questions base on his experience
    """


def skills_technology_template():
    return """
    given the information {information} about person  from I want you to create:
        1. 5 Technology what he know
    """


def short_summary_template():
    return """
    given the information {information} about person  from I want you to create:
        1. a short summary of the person
    """


def interesting_facts_template():
    return """
    given the information {information} about person  from I want you to create:
        1. Two interesting facts about the person
    """


def interview_questions_template():
    return """
    given the linkedin information {information} about person  from I want you to create:
        1. Four interview questions with answers base on his experience
    """


def free_question_template(question):
    return """
     given the information {information} about person  from I want you to create:
        1. """ + question + ""
    """"""


def get_text_from_wav(path) -> str:
    audio_file = path
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript['text']


def get_text_from_pdf(path: str) -> str:
    text = ""
    reader = PyPDF2.PdfReader(path)
    for x in range(len(reader.pages)):
        filePage = reader.pages[x]
        text += filePage.extract_text(0)
    return text


def pdf_cv_analysis(file_text: str, template, question=None) -> str:
    text = file_text
    if question is None:
        summary_prompt_template = PromptTemplate(input_variables=["information"], template=template())
    else:
        summary_prompt_template = PromptTemplate(input_variables=["information"], template=template(question))
    llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')
    chain = LLMChain(llm=llm, prompt=summary_prompt_template)
    profile = chain.run(information=text)
    return profile
