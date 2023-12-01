'''
@TranRick 2023/06/30
This code features the following:
1. Using GPT-3.5 or GPT-4 to Create Instruction input for Blip2 or InstructBLIP 
2. Using BLIP2 or InstructBLIP to generate a Answer (Abstract Visual Information Summary )

'''

import os
import yaml
from tqdm import tqdm
import torch
import openai

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  
import backoff # for exponential backoff

# ## Efficient OpenAI Request 
# from gptcache import cache
# from gptcache.adapter import openai

# cache.init()
# cache.set_openai_key() 

#***************  Section 1 GPTs model to Create Prompt *****************#

def set_openai_key():
    openai.api_type = "azure"
    openai.api_version = "2023-03-15-preview" 
    openai.api_base ="https://sslgroupservice.openai.azure.com/" # "https://agentgpt.openai.azure.com/" #
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
def get_instructions(input_INSTRUCTION,sub_INSTRUCTION, solution_INSTRUCTION,ANSWER_INSTRUCTION, SUB_ANSWER_INSTRUCTION, FIRST_instruction):
    instructions_dict = {
        'question': input_INSTRUCTION, 
        'sub_question': sub_INSTRUCTION,
        'summary': solution_INSTRUCTION,
        'answer': ANSWER_INSTRUCTION,
        'sub_answer': SUB_ANSWER_INSTRUCTION,
        'first_question': FIRST_instruction
    }
    return instructions_dict

def prepare_gpt_prompt(task_prompt, questions, answers, sub_prompt):
    gpt_prompt = '\n'.join([task_prompt, 
                             get_chat_log(questions, answers), 
                             sub_prompt])
    return gpt_prompt

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
def call_gpt3(gpt3_prompt, max_tokens=40, model="text-davinci-003"):  # 'text-curie-001' does work at all to ask questions
    
    response = openai.Completion.create(engine=model, prompt=gpt3_prompt, max_tokens=max_tokens)  # temperature=0.6, 
    reply = response['choices'][0]['text']
    total_tokens = response['usage']['total_tokens']
    return reply, total_tokens

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
def call_chatgpt(chatgpt_messages, max_tokens=40, model="gpt-35-turbo"):
    response = openai.ChatCompletion.create(engine=model, messages=chatgpt_messages,#[chatgpt_messages],
    temperature=0.7,
    max_tokens=max_tokens,
    top_p=0.95,
    frequency_penalty=1.2,
    presence_penalty=0,
    stop=None)
    reply = response['choices'][0]['message']['content']
    total_tokens = response['usage']['total_tokens']
    return reply, total_tokens

## Building Multiple Input Prompt to Maximizing the Information 
def prepare_chatgpt_message(task_prompt, questions, answers, sub_prompt):
    messages = [{"role": "system", "content": task_prompt}]
    
    assert len(questions) == len(answers)
    
    for q, a in zip(questions, answers):
        messages.append({'role': 'assistant', 'content': 'Question: {}'.format(q)})
        messages.append({'role': 'user', 'content': 'Answer: {}'.format(a)})
    messages.append({"role": "system", "content": sub_prompt})
    
    return messages

def get_chat_log(questions, answers, last_n=-1):
    n_addition_q = len(questions) - len(answers)
    assert (n_addition_q) in [0, 1]
    template = 'Question: {} \nAnswer: {} \n'
    chat_log = ''
    if last_n > 0:
        answers = answers[-last_n:]
        questions = questions[-(last_n+n_addition_q):]
    elif last_n == 0:
        answers = []
        questions = questions[-1:] if n_addition_q else []
    
    for i in range(len(answers)):
        chat_log = chat_log + template.format(questions[i], answers[i])
    if n_addition_q:
        chat_log = chat_log + 'Question: {}'.format(questions[-1])
    else:
        chat_log = chat_log[:-2]  # remove the last '/n'
    return chat_log

class Generate_instruction_Input_output():
    
    def __init__(self, img, blip2, GPT_model, 
                 FIRST_instruction, input_INSTRUCTION,
                 sub_INSTRUCTION, 
                 VALID_CHATGPT_MODELS, VALID_GPT3_MODELS,
                 ANSWER_INSTRUCTION, SUB_ANSWER_INSTRUCTION,
                 max_gpt_token=100, n_blip2_context=-1,debug=False):
        
        self.img = img
        self.blip2 = blip2
        self.model = GPT_model
        self.max_gpt_token = max_gpt_token
        self.n_blip2_context = n_blip2_context

        ## Initial Model and Instruction input 
        self.FIRST_instruction = FIRST_instruction
        self.input_INSTRUCTION=input_INSTRUCTION
        self.sub_INSTRUCTION = sub_INSTRUCTION
        self.VALID_CHATGPT_MODELS = VALID_CHATGPT_MODELS
        self.VALID_GPT3_MODELS =VALID_GPT3_MODELS

        ## Initialize the Answer Instruction format for BLIP2 & InstructBLIP \
        self.ANSWER_INSTRUCTION = ANSWER_INSTRUCTION
        self.SUB_ANSWER_INSTRUCTION = SUB_ANSWER_INSTRUCTION


        self.questions = []
        self.answers = []
        self.total_tokens = 0
        self.debug = debug

    def reset(self, img):
        """
        Resets the state of the generator.
        """
        self.img = img
        self.questions = []
        self.answers = []
        self.total_tokens = 0

    ## Type 1 Instruction input 
    def ask_question(self):
        
        if len(self.questions) == 0:
            
            # first question is given by human to request a general discription
            question = self.FIRST_instruction
        else:
            
            if self.model in self.VALID_CHATGPT_MODELS:
                chatgpt_messages = prepare_chatgpt_message(
                    self.input_INSTRUCTION,
                    self.questions, self.answers,
                    self.sub_INSTRUCTION
                )
                question, n_tokens = call_chatgpt(chatgpt_messages, model=self.model, max_tokens=self.max_gpt_token)
            
            elif self.model in self.VALID_GPT3_MODELS:
                # prepare the context for GPT3
                gpt3_prompt = prepare_gpt_prompt(
                    self.input_INSTRUCTION, 
                    self.questions, self.answers,
                    self.sub_INSTRUCTION
                )

                question, n_tokens = call_gpt3(gpt3_prompt, model=self.model, max_tokens=self.max_gpt_token)

            else:
                raise ValueError('{} is not a valid question model'.format(self.model))

            self.total_tokens = self.total_tokens + n_tokens
        if self.debug:
            print(question)
        return question

    def question_trim(self, question):
        question = question.split('Question: ')[-1].replace('\n', ' ').strip()
        if 'Answer:' in question:  # Some models make up an answer after asking. remove it
            q, a = question.split('Answer:')[:2]
            if len(q) == 0:  # some not so clever models will put the question after 'Answer:'.
                question = a.strip()
            else:
                question = q.strip()
        
        return question

    def answer_question(self, decoding_strategy="nucleus", max_length=100, min_length=50):
        # prepare the context for blip2
        blip2_prompt = '\n'.join([self.ANSWER_INSTRUCTION,
                                  get_chat_log(self.questions, self.answers, last_n=self.n_blip2_context),
                                  self.SUB_ANSWER_INSTRUCTION])

        answer = self.blip2.abstract_visual_output(self.img, blip2_prompt,
                                                   llm_decoding_strategy=decoding_strategy,
                                                   max_length=max_length, min_length=min_length) 
        if self.debug:
            print("Answer:", answer)
        return answer

    def answer_trim(self, answer):
        answer = answer.split('Question:')[0].replace('\n', ' ').strip()
        return answer

    
    def chatting(self, n_rounds, print_mode, BLIP_llm_decoding_strategy="nucleus", BLIP_max_length_token_output=100, BLIP_min_length_token_output=50):
        
        if print_mode == 'chat':
            print('-------- Instruction Input & Response ----------')

        for i in tqdm(range(n_rounds), desc='Chat Rounds', disable=print_mode != 'bar'):
            question = self.ask_question()
            # print('Raw: {}'.format(question))
            question = self.question_trim(question)
            self.questions.append(question)

            if print_mode == 'chat':
                #print('GPT: {}'.format(question))
                print(f"Model: {self.model}, question: {question}")

            answer = self.answer_question(decoding_strategy=BLIP_llm_decoding_strategy, max_length=BLIP_max_length_token_output, min_length=BLIP_min_length_token_output)
            answer = self.answer_trim(answer)
            self.answers.append(answer)

            if print_mode == 'chat':
                #print('BLIP-2: {}'.format(answer))
                print(f"BLIP_Model: {self.blip2.visual_understand}, answer: {answer}")


        if print_mode == 'chat':
            print('--------     Ends  ----------')

        return self.questions, self.answers, self.total_tokens
