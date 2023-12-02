import sys
sys.path.insert(0, '..')
from utils.utils import *
import os
import openai
from typing import List
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class ChatGPT():
    def __init__(self,model_name="gpt-3.5-turbo"):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model_name=model_name
    def get_answer(self,prompt:str):
        # for final reasoning
        while True:
            try:
                completion = openai.ChatCompletion.create(
                            model=self.model_name,
                            messages=[
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.2,
                            presence_penalty=1
                            )
                return completion.choices[0].message["content"]
            except Exception as e:
                print(f"openai connection error: {e}, try again")
 
    def get_answer_batched(self, questions:List[str], references:List[str], with_reasoning=True):
        # for textual qa 
        gpt_results=[]
        failed_trial=0
        if not with_reasoning:
            while questions:
                try:
                    question=questions[0]
                    reference=references[0]
                    prompt=f"You are doing extractive question answering. Given a document: {reference}. Extract a short answer to the question: {question} from the document. If there is not enough information to answer this question, just answer me with 'Unkown'. The answer should contain only one or two words: "
                    completion = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    presence_penalty=-0.2
                    )
                    gpt_results.append(completion.choices[0].message["content"])
                    questions.pop(0)
                    references.pop(0)
                except Exception as e:
                    print(f"openai connection error #{failed_trial}: {e}, try again")
                    failed_trial+=1
            return gpt_results,None
        else:
            intermediate_reasoning = []
            while questions:
                try:
                    question = questions[0]
                    reference = references[0]
    
                    prompt1 = f"{reference}\n\n{question}\n\n Let's think step by step."
    
                    completion1 = openai.ChatCompletion.create(
                    model = self.model_name,
                    messages = [{"role": "user", "content": prompt1}],
                    max_tokens=100, temperature=0.2,
                    presence_penalty=-0.2)
                    response1 = completion1['choices'][0]['message']['content']
                    if check_negative_words(response1):
                        response2='Unknown bad reasoning' 
                    else:
                        prompt2 = f"{response1}\n\n'{question}\n\nGive me a very short answer, in one or two words."
                        completion2 = openai.ChatCompletion.create(
                            model = self.model_name,
                            messages = [{"role": "user", "content": prompt2}],max_tokens=100, temperature=0.2,
                        presence_penalty=-0.2)
                        response2 = completion2.choices[0].message["content"]
                    gpt_results.append(response2)
                    intermediate_reasoning.append(response1)
                    questions.pop(0)
                    references.pop(0)
                except Exception as e:
                    print(f"openai connection error #{failed_trial}: {e}, try again")
                    failed_trial+=1
        return gpt_results,intermediate_reasoning
        

class Llama():
    def __init__(self,model_name):
        self.model_name=model_name
        print(f"Loading model {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
    def get_answer(self,prompt:str):
        # for final reasoning
        with torch.no_grad():
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            output = self.model.generate(input_ids,max_new_tokens=10, temperature=0.2,top_k=40,length_penalty=-1)
            response = self.tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            del output
            torch.cuda.empty_cache()
        return response.replace(prompt,'')
    def get_answer_batched(self, questions:List[str], references:List[str],with_reasoning=True,direct_answer=False):
        if direct_answer:
            prompt=[prompt_direct_qa(q) for q in questions]
        else:
            if 'Llama-2' in self.model_name:
                prompt=[prompt_qa_llama2(r,q) for r,q in zip(references,questions)]
            else:
                prompt=[prompt_qa(r,q) for r,q in zip(references,questions)]
        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt",padding=True,max_length=1000,truncation=True).to(self.device)
            del inputs['token_type_ids']
            output = self.model.generate(**inputs, max_new_tokens=10,temperature=0.2,top_k=40,length_penalty=-1)
            response=self.tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            del output
            torch.cuda.empty_cache()
        return [get_sentence_before_first_newline(r.replace(p,'').strip()) for r,p in zip(response,prompt)],None
    
