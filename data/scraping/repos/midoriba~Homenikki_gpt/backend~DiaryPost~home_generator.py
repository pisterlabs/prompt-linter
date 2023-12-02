from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain import LLMChain
import json

class HomeGenerator:
    __llm = OpenAI(model_name='text-davinci-003')
    with open('DiaryPost/home_template.json', encoding='utf-8', mode='r') as f:
        __templates = json.load(f)

    def __init__(self, personality=None):
        self.__template = self.__templates.get(personality, self.__templates['standard'])
        self.__prompt = PromptTemplate(
            input_variables=['diary_text'],
            template=self.__template
        )
        self.__chain = LLMChain(llm=self.__llm, prompt=self.__prompt, verbose=True)

    def generate(self, text):
        return self.__chain(text)['text']
    
if(__name__ == '__main__'):
    hg = HomeGenerator('wildman')
    print(hg.generate(input()))