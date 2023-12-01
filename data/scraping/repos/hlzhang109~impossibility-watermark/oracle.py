import json
import openai
import os
from dotenv import load_dotenv
import numpy as np
import tiktoken
import random
from time import sleep
import jsonlines
import re
import copy
import transformers

DEF_MODEL = "gpt-4"
MODELS = {"gpt-4": "gpt-4", "gpt-3.5": "gpt-3.5-turbo"}
TOKENIZERS  = {model : tiktoken.encoding_for_model(MODELS[model]) for model in MODELS }
load_dotenv(dotenv_path='./.env') # take environment variables from .env with OPENAI_API_TOKEN=<your_key_here>
if os.getenv("OPENAI_API_ENDPOINT"):
    openai.api_base = os.getenv("OPENAI_API_ENDPOINT")
openai.api_key = os.getenv("OPENAI_API_KEY")

def set_seed(seed):
   random.seed(seed)
   np.random.seed(seed)

def hamming_distance(str1, str2):
    if len(str1) != len(str2):
        raise ValueError("Strings must be of the same length")
    distance = 0
    for char1, char2 in zip(str1, str2):
        if char1 != char2:
            distance += 1      
    return distance

def tokens(s, model = DEF_MODEL):
  """Returns tokens of a string.
     Returns two lists, one of strings and the other of integers."""
  tokenizer = TOKENIZERS[model]
  L=tokenizer.encode(s)
  return [str(tokenizer.decode_single_token_bytes(t))[2:-1] for t in L],L

def count_tokens(s, tokenizer, model = DEF_MODEL):
  """Count the number of tokens in a string"""
  return len(tokenizer.encode(s))

def truncate(s,n, model = DEF_MODEL):
  """Truncase to n tokens"""
  tokenizer = TOKENIZERS[model]
  L = tokenizer.encode(s)
  return tokenizer.decode(L[:n])

def tokens2str(tokens, model = DEF_MODEL):
  tokenizer = TOKENIZERS[model]
  """Returns string from tokens (should get the integer tokens)"""
  return tokenizer.decode(tokens)

def read_jsonl(file: str):
    data = []
    with open(file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def query_openai(prompt, model="text-davinci-003", max_tokens=512):
    # prompt = instruction+"\n"+query
    response = openai.Completion.create(
        engine=model, # "gpt-3.5-turbo-instruct"
        prompt=prompt,
        temperature=.2,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        # stop=["\n"]
    )
    return response.choices[0].text

def chopped(s,k=30):
  """Chop a string to a shorter representation for prining"""
  if len(s)<=2*k: return(s)
  return f"{s[:k]}...{s[-k:]}"

def chat(message, history = [{"role": "system", "content": "You are a research assistant."}],
         model = "gpt-4", # model is "gpt-3" or "gpt-4"
         return_more = False,  # return also history and response object
         debug=True,  # print message and response
         supress_exception = False,  # supress exception and return None
         retries = 500, # number of times to retry if there is an exception
         tokenizer = None,
         **extra_params # extra parameters for Chat
         ):
  """General routine to send a message to GPT.
     Can take an optional parameter history of messages, and can also return message and history as extra parameter"""
  CONTEXT = {"gpt-4":8192, "gpt-3.5": 4096}
  if tokenizer is None:
    tokenizer = TOKENIZERS[model]
  hist_tokens  = count_tokens(", ".join(D["content"] for D in history), tokenizer)
  message_tokens = count_tokens(message, tokenizer)

  while retries >= 0:
    try:
      if debug: print(f"Message:\n {chopped(message)} ({message_tokens} tokens, history {hist_tokens} tokens)", flush=True)
      history = history + [{"role": "user", "content": f"{message}"}]
      params = dict(
            model = MODELS[model],
            messages= history,
            max_tokens=1024, # 512
            n=1,
            temperature=0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            )
      params.update(extra_params) # update based on any extra parameters to add
      response = openai.ChatCompletion.create(**params)
      break
    except Exception as e:
      print(f"Error!:\n{e}")
      if retries:
        print(f"Retrying: {retries} tries left")
        sleep(1)
        retries -= 1
      elif not supress_exception:
        raise e
      else:
        return None

  text_response =  response.choices[0]['message']['content']
  if debug: print(f"Response:\n {chopped(text_response)} ({count_tokens(text_response, tokenizer)} tokens)", flush=True)
  if return_more:
    return text_response, history + [{"role": "assistant", "content": text_response}], response
  return text_response
    
def count_masks(texts):
    return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]

def extract_fills(texts):
    # remove <pad> from beginning of each text
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]
    pattern = re.compile(r"<extra_id_\d+>")
    # return the text in between each matched mask token
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]
    # remove whitespace around each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]
    return extracted_fills

def join_tokens(tokens):
    joined = " ".join(tokens)
    # Remove spaces before certain punctuation marks
    joined = re.sub(r'\s([,.;!?])', r'\1', joined)
    return joined

def apply_extracted_fills(masked_texts, extracted_fills):
    # split masked text into tokens, only splitting on spaces (not newlines)
    tokens = [x.split(' ') for x in masked_texts]

    n_expected = count_masks(masked_texts)

    # replace each mask token with the corresponding fill
    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

    # join tokens back into text
    texts = [join_tokens(x) for x in tokens]
    return texts

def load_data(jsonl_file='data/lfqa/lfqa_umd.jsonl'):
    data = []
    with jsonlines.open(jsonl_file, 'r') as reader:
        for item in reader:
            data.append(item)
    return data

class Oracle:
    def __init__(self, query, response, check_quality=False, choice_granuality=5, use_chat_arena_prompt=False, cache_dir='~/.cache') -> None:
        self.init_score = -1
        self.query = query
        self.response = response
        self.detailed_prompt = "" 
        self.choice_granularity = choice_granuality
        self.system_prompt = "You are a capable, helpful and useful assistant." if not use_chat_arena_prompt else self.chat_arena_prompt
        self.history =  [{"role": "system", "content": self.system_prompt}]
        tokenizer_name = reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
        self.reward_model = transformers.AutoModelForSequenceClassification.from_pretrained(reward_name, cache_dir=cache_dir).to("cuda:0")
        self.check_quality = check_quality

    @property
    def chat_arena_prompt(self):
        return f'''
        Please act as an impartial judge and evaluate the quality of the responses provided by two
        AI assistants to the user question displayed below. You should choose the response that
        follows the user’s instructions and answers the user’s question better. Your evaluation
        should consider factors such as the helpfulness, relevance, accuracy, depth, creativity,
        and level of detail of their responses. Begin your evaluation by comparing the two
        responses and provide an explanation. Avoid any position biases and ensure that the
        order in which the responses were presented does not influence your decision. Do not allow
        the length of the responses to influence your evaluation. Be as objective as possible.
        Make sure the better resopnse does not have isolated punctuation marks.
        Note that grammatical errors would greatly degrade the quality of a response.
        '''
    
    @property
    def instruction(self):
        return f"Below are two candidate responses to the query {self.query}: "
        # f"Below are two candidate completions to the news article prefix ``{self.query}'': "

    @property
    def comparison_prompt(self):
        return '''Response A: [A] \n Response B: [B] \n 
        Compare which of the two above responses is a better response to the given prompt. 
        Explain your reasoning step by step. 
        Your evaluation should consider factors such as the repetition, grammar, coherence, relevance, accuracy of the responses.
        Especially, note that having grammatical errors, repetitions, capitalization errors or punctuation mistakes would greatly degrade the quality of a response.
        '''   
    
    @property
    def check_error_prompt(self):
        return '''Does the response above have any grammatical errors or capitalization errors or punctuation mistakes? If so, answer 1, otherwise answer 2. '''
    
    @property
    def check_quality_prompt(self):
        return  "Text quality is affected by factors such as unnecessary repetitions, grammar, coherence, relevance, and accuracy of the responses. Especially, having grammatical errors, repetitiveness, capitalization errors or punctuation mistakes would greatly degrade the quality of a response." + "\n" + \
                "Therefore, is the new modified response of equal or higher quality compared to original response? If so, answer Yes, otherwise answer No."
        #   '''Is the text above of high-quality? If so, answer Yes, otherwise answer No.'''
        # return '''Any repetitiveness, grammatical errors or capitalization errors or punctuation mistakes would substantially degrade text quality. Therefore, is the text above of high-quality? If so, answer Yes, otherwise answer No.'''
    
    @property
    def five_choice(self):
        return '''
        (1) Response A is much better than response B
    	(2) Response A is a little better than response B
    	(3) Responses A and B have similar quality
    	(4) Response B is a little better than response A
    	(5) Response B is much better than response A
        '''

    @property
    def three_choice(self):
        return '''
        (1) Response A is better than response B
    	(2) Responses A and B have similar quality
    	(3) Response B is better than response A
        '''
    
    @property
    def answer_prompt(self):
        choices = self.five_choice if self.choice_granularity == 5 else self.three_choice
        return f'''So based on your reasoning, choose one of the following {choices}''' # So which of the two responses is a better response to the given prompt? 
    
    def query_gpt_once(self, prompt, model="gpt-3.5", max_tokens=5, tokenizer=None):
        # avoid using max_tokens which can be too short for generating explanations.
        n_attempt = 0
        while n_attempt < 5:
            try:
                oracle_reasoning = chat(prompt, history=self.history, model=model, tokenizer=tokenizer)
                history = copy.deepcopy(self.history)
                history.append({"role": "user", "content": f"{prompt}"})
                history.append({"role": "assistant", "content": f"{oracle_reasoning}"})
                oracle_answer = chat(self.answer_prompt, history=history, model=model, tokenizer=tokenizer)
                pattern = r'\((\d+)\)'
                filtered_response = re.findall(pattern, oracle_answer)
                print(oracle_reasoning)
                print(oracle_answer)
                score = int(filtered_response[-1])
                if score not in range(1, self.choice_granularity+1):
                    print(f"return choice {score} not in [1, {self.choice_granularity}]")
                    continue
                return score
            except Exception as e:
                print(e)
                n_attempt += 1
                continue
 
    def query_rm_once(self, response_1, response_2, tie_threshold=0.01, model="gpt-3.5", max_tokens=5, tokenizer=None):
        context = '###Human: ' + self.query + '###Assistant: '
        text1 = context + response_1
        text2 = context + response_2
        tokenized_text1 = self.tokenizer.encode_plus(text1, return_tensors="pt")
        tokenized_text2 = self.tokenizer.encode_plus(text2, return_tensors="pt")
        device="cuda:0"
        input_ids_1, attention_mask_1 = tokenized_text1['input_ids'].to(device), tokenized_text1['attention_mask'].to(device)
        input_ids_2, attention_mask_2 = tokenized_text2['input_ids'].to(device), tokenized_text2['attention_mask'].to(device)
        score_1 = self.reward_model(input_ids=input_ids_1,attention_mask=attention_mask_1).logits.detach()
        score_2 = self.reward_model(input_ids=input_ids_2,attention_mask=attention_mask_2).logits.detach()
        softmax = torch.nn.Softmax(dim=0)
        scores = softmax(torch.tensor([score_1,score_2]))
        score_gap = abs(scores[0].item()-scores[1].item())
        if score_gap < tie_threshold: 
            return 2
        elif score_1 > score_2:
            return 1
        else:
            return 3
       
    def maintain_quality(self, paraphrased_response, tie_threshold=0.1, model="gpt-3.5", max_tokens=5, tokenizer=None):
        choice = self.query_rm_once(paraphrased_response, self.response, tie_threshold=tie_threshold)
        self.choice_granularity = 3
        # positive score means paraphrased response wins
        score_dict = {1: 1, 2: 0.5, 3: 0, 4: -0.5, 5:-1} if self.choice_granularity == 5 else {1: 1, 2: 0, 3: -1} 
        if choice is None:
            return False
        score = score_dict[choice]
        print()
        print("Second round of comparison:")
        print()
        second_choice = self.query_rm_once(self.response, paraphrased_response, tie_threshold=tie_threshold)
        if second_choice is None:
            return False
        
        score -= score_dict[second_choice]
        if score < 0:
            return False
        elif score >=0 and self.check_quality:
            prompt = f"Original response: {self.response}" + "\n" + "New response: " + paraphrased_response + "\n" + self.check_quality_prompt #self.check_error_prompt
            check_quality = chat(prompt, model=model, tokenizer=tokenizer)
            if 'yes' in check_quality.lower(): # new response is at least as good as the original response
                return True
            else:
                return False 
        else:
            return True

    def report_mean_score(self, paraphrased_response, tie_threshold=0.1, model="gpt-3.5", max_tokens=5, tokenizer=None):
        choice = self.query_gpt_once(paraphrased_response, self.response)
        self.choice_granularity = 3
        # NOTE positive scores for paraphrased response winning
        score_dict = {1: 1, 2: 0.5, 3: 0, 4: -0.5, 5:-1} if self.choice_granularity == 5 else {1: 1, 2: 0, 3: -1} 
        if choice is None:
            return False
        score = score_dict[choice]
        print()
        print("Second round of comparison:")
        print()
        choice = self.query_gpt_once(self.response, paraphrased_response, tie_threshold=tie_threshold)
        second_score = score_dict[choice]
        return (score-second_score)/2 # essentially (1, 0), (1, -1), (1, 1), (0, 1), (0, -1), (0, 0), (-1, 1), (-1, 0), (-1, -1) -> 0.5, 0, 1, 0.5, -0.5, 0, 0, -0.5, -1
