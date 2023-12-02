import os
import time
import anthropic
import openai
import numpy as np
from dotenv import load_dotenv

#! 
def get_llm(engine, temp=0.0, max_tokens=1, arms=('F', 'J'), with_suffix=False):
    '''
    Based on the engine name, returns the corresponding LLM object
    '''
    Q_, A_ = '\n\nQ:', '\n\nA:'
    if engine.startswith("gpt"):
        # load_dotenv(); gpt_key = os.getenv(f"OPENAI_API_KEY{2 if engine == 'gpt-4' else ''}")
        load_dotenv(); gpt_key = os.getenv(f"OPENAI_API_KEY")
        llm = GPT4LLM((gpt_key, engine))
    elif engine.startswith("claude"):
        load_dotenv(); anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        llm = AnthropicLLM((anthropic_key, engine))
        Q_, A_ = anthropic.HUMAN_PROMPT, anthropic.AI_PROMPT
    else:
        print('No key found')
        llm = DebuggingLLM(arms)
    return llm, Q_, A_

class LLM:
    def __init__(self, llm_info):
        self.llm_info = llm_info

    def generate(self, text, temp=0.0, max_tokens=1, arms=None):
        raise NotImplementedError

class DebuggingLLM(LLM):
    def __init__(self, llm_info):
        super().__init__(llm_info)
        print("DEBUGGING MODE")
        info = llm_info
        if info[0] == 'not a 2 armed bandit':
            self.random_fct = info[1]
        else:
            arm1 = info[0]
            arm2 = info[1]
            self.random_fct = lambda : arm1 if np.random.rand() < 0.5 else arm2

    def generate(self, text, temp=0.0, max_tokens=1, arms=None):
        if arms:
            if  arms[0] == 'not a 2 armed bandit': 
                random_fct = arms[1]
            else:
                random_fct = lambda : arms[0] if np.random.rand() < 0.5 else arms[1]
        else:
            random_fct = self.random_fct
        return random_fct()

class GPT4LLM(LLM):
    def __init__(self, llm_info):
        self.gpt_key, self.engine = llm_info
        openai.api_key = self.gpt_key

    def generate(self, text, temp=0, max_tokens=1, arms=None):
        text = [{"role": "user", "content": text}]  
        time.sleep(1) # to avoid rate limit error which happens a lot for gpt4
        for iter in range(10):
            try:
                response = openai.ChatCompletion.create(
                    model = self.engine,
                    messages = text,
                    max_tokens = max_tokens,
                    temperature = temp
                )
                return response.choices[0].message.content.replace(' ', '')
            except:
                time.sleep(3**iter)
                if iter == 5:
                    import ipdb; ipdb.set_trace()

class AnthropicLLM(LLM):
    def __init__(self, llm_info):
        self.anthropic_key, self.engine = llm_info

    def generate(self, text, temp=0.0, max_tokens=1, arms=None):
        c = anthropic.Anthropic(api_key=self.anthropic_key)

        response = c.completions.create(
            prompt = anthropic.HUMAN_PROMPT + text,
            stop_sequences=[anthropic.HUMAN_PROMPT],
            model=self.engine,
            temperature=temp,
            max_tokens_to_sample=max_tokens,
        ).completion.replace(' ', '')
        c.close()
        return response