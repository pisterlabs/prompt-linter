from django.shortcuts import render

# Create your views here.
from rest_framework.decorators import api_view
from rest_framework.response import Response
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import json
from fastapi.encoders import jsonable_encoder
import re

from prompts import *
import os
import sys
import openai
import backoff
from functools import partial
import re
import numpy as np
# import file



api_key = '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'
embeddings = OpenAIEmbeddings(openai_api_key=api_key)


    
@api_view(['POST'])
def process_file(request):
    if request.method == 'POST' and request.FILES['file']:
        file = request.FILES['file']
        global contents
        if file.name.endswith('.pdf'):
           pdf_file = PdfReader(file)
           extracted_text = ""
           for page_num in range(len(pdf_file.pages)):
                page = pdf_file.pages[page_num]
                text = page.extract_text()
                
                extracted_text += text
           contents = extracted_text
            
        elif file.name.endswith('.txt'):
            contents = file.read().decode('utf-8')
            
        elif file.name.endswith('.docx'):
            contents = "word format not supported"
            
        else:
            contents = "format not supported"

        
        contents=remove_non_alphanumeric(contents[:1000])
        global documents
 # text_splitter = CharacterTextSplitter(separator = " ", chunk_size=200, chunk_overlap=0)
        splitter = CharacterTextSplitter(separator=" ", chunk_size=200, chunk_overlap=2)

        with open('/contents.txt', 'w', encoding='utf-8') as file:
            file.write(contents)


        documents = splitter.create_documents([contents])
        print("global documents    ",documents)
        # embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        # docsearch = Chroma.from_documents(documents, embeddings)
       

        pgc={}
        for i in range(1,len(documents)):
            srn="source"+str(i)
            pgc[srn]=documents[i].page_content
            

        with open('D:/Grad/Internship/QABOT/fileapi/fileapi_app/documents.txt', 'w', encoding='utf-8') as file:
            file.write(str(pgc))



      
        

        return Response({'result': contents})
    else:
        return Response({'error': 'error'})



@api_view(['POST'])
def chat(request):
    if request.method == 'POST':
        
        
        input_string = request.data.get('input_string', '')

        print("input_string  ",input_string)
        strjson=json.dumps( jsonable_encoder(documents))
        
        with open('/contents.txt', 'r') as file:
            contents = file.read()

        print("contents   ",contents[10])

        # question="question: What serves as an impermeable barrier to aggressive metabolites?"
        question="question: "+input_string
        print("question   ",question)
        promptstr="""

{
	'Source1': Hexagonal Boron Nitride The Thinnest Insulating Barrier to Microbial Corrosion Govinda ChilkoorSushma Priyanka KaranamShane StarNamita ShresthaRajesh K Sani Venkata KK UpadhyayulaDebjit GhoshalNikhil A KoratkarM Meyyappan and Venkataramana Gadhamshetty Civil and Environmental EngineeringMaterials and Metallurgical EngineeringChemical and Biological Engineering and Surface Engineering Research Center South Dakota School of Mines and Technology 501 E Saint Joseph Boulevard Rapid City South Dakota 57701 United States Green Technologies and Environmental Economics Platform Chemistry Department Umea University Umea Sweden 90187 Department of Chemical and Biological EngineeringDepartment of .
	'Source2': Mechanical Aerospace and Nuclear Engineering and Department of Materials Science and Engineering Rensselaer Polytechnic Institute 110 Eighth Street Troy New York 121803590 United States Center for Nanotechnology NASA Ames Research Center Mo ett Field Mountain View California 94035 United States SSupporting Information ABSTRACT We report the use of a single layer of two dimensional hexagonal boron nitride SLhBN as thethinnest insulating barrier to microbial corrosion inducedby the sulfatereducing bacteria Desulfovibrio alaskensis G20 .
	'Source3': We used electrochemical methods to assess the corrosionresistance of SLhBN on copper against the e ects of both the planktonic and sessile forms of the sulfatereducingbacteria Cyclic voltammetry results show that SLhBNCuis eective in suppressing corrosion e ects of the planktonic cells at potentials as high as 02 V vsAg AgCl The peak anodic current for the SLhBN coatings is36 times lower than that of bare Cu Linear polarization resistance tests con rm that the SLhBN coatings serve as a barrier against corrosive e ects of the G20 bio lm when compared to bare Cu The SLhBN serves as an impermeable barrier to aggressive metabolites and o ers91 corrosion inhibition e ciency which is comparable to much thicker commercial coatings such as polyaniline In addition to impermeability the insulating nature of SLhBN suppresses galvanic e ects and improves its ability to combat microbial .
	'Source4': corrosion KEYWORDS 2D coatings hexagonal boron nitride microbial corrosion sulfatereducing bacteria Microbially induced corrosion MIC results in an unanticipated attack on metals in seemingly benignenvironments and threatens a range of multibillion dollar industries including aviation surface transportation and water infrastructure1MIC accounts for 20 40 of the annual corrosion costs including direct and indirect impacts whichhave been estimated to reach as high as 1 trillion2MIC poses a signi cantnancial burden in the US alone in the form of total direct costs 30 50 billionyear3biocide require ments 12 billionyear4and direct costs in oil and gas industries 2 billionyear5The US annually spends nearly 6 billion to .
	'Source5': combat MIC e ects of sulfatereducing bacteria SRB alone6The SRBs secrete exopolymers on metal surfaces and form a bio lm to induce uniform corrosion or localized pitting attack using one or more of the following mechanisms7 i disruption of passivating metaloxide lms ii alteringredox conditions at the metal solution interface iii regeneration of the electron acceptors iv production of aggressive metabolites eg suldes and v depolarizing the cathodic reactions Major corrosion mitigation practices including protective coatings and impressed current cathodic protection tend to fail under MIC conditions For example commercially availablepolymer coatings eg epoxy liners are prone to biodegrada tion and they exhibit poor adhesion toward metals underaqueous conditions 811The thickness of commercial coatings 501000 m can also disrupt the functionality eg Received August 31 2017 Accepted .
}
Hexagonal Boron Nitride The Thinnest Insulating Barrier to Microbial Corrosion Govinda ChilkoorSushma Priyanka KaranamShane StarNamita ShresthaRajesh K Sani Venkata KK UpadhyayulaDebjit GhoshalNikhil A KoratkarM Meyyappan and Venkataramana Gadhamshetty Civil and Environmental EngineeringMaterials and Metallurgical EngineeringChemical and Biological Engineering and Surface Engineering Research Center South Dakota School of Mines and Technology 501 E Saint Joseph Boulevard Rapid City South Dakota 57701 United States Green Technologies and Environmental Economics Platform Chemistry Department Umea University Umea Sweden 90187 Department of Chemical and Biological EngineeringDepartment of Mechanical Aerospace and Nuclear Engineering and Department of Materials Science and Engineering Rensselaer Polytechnic Institute 110 Eighth Street Troy New York 121803590 United States Center for Nanotechnology NASA Ames Research Center Mo ett Field Mountain View California 94035 United States SSupporting Information ABSTRACT We report the use of a single layer of two dimensional hexagonal boron nitride SLhBN as thethinnest insulating barrier to microbial corrosion inducedby the sulfatereducing bacteria Desulfovibrio alaskensis G20 We used electrochemical methods to assess the corrosionresistance of SLhBN on copper against the e ects of both the planktonic and sessile forms of the sulfatereducingbacteria Cyclic voltammetry results show that SLhBNCuis eective in suppressing corrosion e ects of the planktonic cells at potentials as high as 02 V vsAg AgCl The peak anodic current for the SLhBN coatings is36 times lower than that of bare Cu Linear polarization resistance tests con rm that the SLhBN coatings serve as a barrier against corrosive e ects of the G20 bio lm when compared to bare Cu The SLhBN serves as an impermeable barrier to aggressive metabolites and o ers91 corrosion inhibition e ciency which is comparable to much thicker commercial coatings such as polyaniline In addition to impermeability the insulating nature of SLhBN suppresses galvanic e ects and improves its ability to combat microbial corrosion KEYWORDS 2D coatings hexagonal boron nitride microbial corrosion sulfatereducing bacteria Microbially induced corrosion MIC results in an unanticipated attack on metals in seemingly benignenvironments and threatens a range of multibillion dollar industries including aviation surface transportation and water infrastructure1MIC accounts for 20 40 of the annual corrosion costs including direct and indirect impacts whichhave been estimated to reach as high as 1 trillion2MIC poses a signi cantnancial burden in the US alone in the form of total direct costs 30 50 billionyear3biocide require ments 12 billionyear4and direct costs in oil and gas industries 2 billionyear5The US annually spends nearly 6 billion to combat MIC e ects of sulfatereducing bacteria SRB alone6The SRBs secrete exopolymers on metal surfaces and form a bio lm to induce uniform corrosion or localized pitting attack using one or more of the following mechanisms7 i disruption of passivating metaloxide lms ii alteringredox conditions at the metal solution interface iii regeneration of the electron acceptors iv production of aggressive metabolites eg suldes and v depolarizing the cathodic reactions Major corrosion mitigation practices including protective coatings and impressed current cathodic protection tend to fail under MIC conditions For example commercially availablepolymer coatings eg epoxy liners are prone to biodegrada tion and they exhibit poor adhesion toward metals underaqueous conditions 811The thickness of commercial coatings 501000 m can also disrupt the functionality eg Received August 31 2017 Accepted"""
        # x = {'question':question, 'sources':promptstr}
        x = {'question':question, 'sources':contents}
        print("brfore run       ")


        responsegen=""
        try:
            responsegen=run(x)
        except:
            print("error")
        print("after run")
        


        return Response({'answer': responsegen})


def remove_non_alphanumeric(text):
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return cleaned_text

steps = 1

completion_tokens = prompt_tokens = 0

# api_key = os.getenv("j1hzz2w0QO7TZhiyzrDLT3BlbkFJJJ0CrYeYpJEuRAxAzIRb", "")
# if api_key != "":
#     openai.api_key = "j1hzz2w0QO7TZhiyzrDLT3BlbkFJJJ0CrYeYpJEuRAxAzIRb"
# else:
#     print("Warning: OPENAI_API_KEY is not set")
openai.api_key = "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"
api_base = os.getenv("OPENAI_API_BASE", "")
if api_base != "":
    print("Warning: OPENAI_API_BASE is set to {}".format(api_base))
    openai.api_base = api_base

# @backoff.on_exception(backoff.expo, openai.error.OpenAIError)
def completions_with_backoff(**kwargs):
    print("in completions_with_backoff")
    global res
    try:
        res=openai.ChatCompletion.create(**kwargs)
    except Exception as error:
        print(error)
    return res

def gpt(prompt, model="gpt-3.5-turbo", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
    
def chatgpt(messages, model="gpt-3.5-turbo", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    print("in chatgpt")
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        
        res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop)
        
        print("out  completions_with_backoff")
        outputs.extend([choice["message"]["content"] for choice in res["choices"]])
        # log completion tokens
        completion_tokens += res["usage"]["completion_tokens"]
        prompt_tokens += res["usage"]["prompt_tokens"]
    return outputs
    
def gpt_usage(backend="gpt-3.5-turbo"):
    global completion_tokens, prompt_tokens
    if backend == "gpt-4":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    elif backend == "gpt-3.5-turbo":
        cost = (completion_tokens + prompt_tokens) / 1000 * 0.0002
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}


class PromptTree:
        
    def __init__(self):
        self.y = ''
        self.steps = 2
        self.stops = []
    
    def generate_thoughts(self, state, question, sources, stop=None):
        # print("STATE", state)
        # print("QUESTION", question)
        # print("SOURCES", sources)
        prompt = cot_prompt.format(question=question, sources=sources) + state
        print("in generate_thoughts")
        thoughts = gpt(prompt, n=3, stop=stop)
        return [state + _ for _ in thoughts]

    def generate_vote(self, thoughts):
            prompt = vote_prompt
            for i, y in enumerate(thoughts, 1):
                # y = y.replace('Plan:\n', '')
                # TODO: truncate the plan part?
                prompt += f'Choice {i}:\n{y}\n'

            vote_outputs = gpt(prompt, n=3, stop=None)

            vote_results = [0] * len(thoughts)
            for vote_output in vote_outputs:
                pattern = r".*best choice is .*(\d+).*"
                match = re.match(pattern, vote_output, re.DOTALL)
                if match:
                    vote = int(match.groups()[0]) - 1
                    if vote in range(len(thoughts)):
                        vote_results[vote] += 1
                else:
                    print(f'vote no match: {[vote_output]}')
            
            return np.argmax(vote_results)




    def solve(self, x):
        '''
        Given an input, uses tree of thought prompting to generate output to answer.
        '''
        
        new_ys = self.generate_thoughts(self.y, *x.values(), '\nSources:')
        best_idx = self.generate_vote(new_ys)
        self.y = new_ys[best_idx]
        new_ys = self.generate_thoughts(self.y, *x.values(), '\nAnswer:')
        best_idx = self.generate_vote(new_ys)
        self.y = new_ys[best_idx]
        new_ys = self.generate_thoughts(self.y, *x.values())
        best_idx = self.generate_vote(new_ys)
        answer = extract_text(new_ys[best_idx])
        print("answer   ",answer)
        return answer



def extract_text(string):
    pattern = r"Answer:(.*?)\n"
    match = re.search(pattern, string, re.DOTALL)
    if match:
        extracted_text = match.group(1).strip()
        return extracted_text
    else:
        return None
def run(x):
    global gpt
    gpt = partial(gpt, model='gpt-3.5-turbo', temperature=1.0)
    tree_of_thought = PromptTree()
    return tree_of_thought.solve(x)

# if __name__ == '__main__':
#     x = sys.argv[1:]
#     x = {x[0]:x[1], x[2]:x[3]}
#     print(run(x))



    

