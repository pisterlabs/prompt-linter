import os
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from constant import openai_key

os.environ['OPENAI_API_KEY'] = openai_key



def gpt_qs(sk_exp,job_applying_for):
    ''' The function takes skill,experience and job_applying_for as params & generates questions '''

    llm = OpenAI(temperature=0.8)

    questions = PromptTemplate(
        input_variables = ['sk_exp','job_applying_for'] ,
        template = '''Write 10 difficult questions for an interviewee about these skills with respective experiences:
        "{sk_exp}" ,
        who is applying for "{job_applying_for}". You must return questions and nothing else. Make sure to not ask question starting from "how have you used...." , just ask technical questions. Make sure to not include anything other than questions.''')

    question = LLMChain(llm=llm , prompt=questions,verbose=True) 
    response = question.run(sk_exp = sk_exp,job_applying_for = job_applying_for)
    with open('questions.txt', 'a') as file:
        file.write(response)

