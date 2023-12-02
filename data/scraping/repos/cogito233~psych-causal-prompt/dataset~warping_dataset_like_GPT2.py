import torch
from torch import nn
from transformers import AutoTokenizer
import sys
from icecream import ic

def generate_prompts(df, setup):
    perfix = []
    postfix = []
    ic(df)
    for i in range(len(df)):
        if (int(df['label'].iloc[i])!=setup and setup!=0):
            continue
        current_str = df['text'].iloc[i].replace("\\n","\n")
        perfix.append(current_str.split(" f{review_text} ")[0])
        postfix.append(current_str.split(" f{review_text} ")[-1])
    ic(len(perfix))
    print(perfix)
    print(postfix)
    return perfix, postfix

from transformers import GPT2Tokenizer
# Use the maxlength method to warp the sentence
class TokenizerWarper(object):
    def __init__(self, prompt, tokenizer_path = "./gpt2-large_saved/token"):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.prompt = prompt
        self.perfix_token = self.tokenizer(prompt[0]+"\"")
        self.postfix_token = self.tokenizer("\" "+prompt[1])
        self.postfix_token_2 = self.tokenizer("...\" "+prompt[1])
        self.max_length = 1024

    def padding(self, padded_dict, max_length):
        if (len(padded_dict['input_ids'])>=max_length):
            #ic(len(padded_dict['input_ids']))
            pass
        for i in padded_dict:
            padded_dict[i] = padded_dict[i] + [0]*(max_length-len(padded_dict[i]))
        return padded_dict

    def encode_single_sentence(self, str):
        #str = '\n'.join(str.split('\\n'))
        mid = self.tokenizer(str)
        perfix_token = self.perfix_token
        postfix_token = self.postfix_token
        postfix_token_2 = self.postfix_token_2
        if (len(perfix_token['input_ids'])+len(mid['input_ids'])+len(postfix_token['input_ids'])>self.max_length):
            #ic(len(perfix_token['input_ids'])+len(mid['input_ids'])+len(postfix_token['input_ids']))
            pass
        if (len(perfix_token['input_ids'])+len(mid['input_ids'])+len(postfix_token['input_ids'])>self.max_length):
            return {
                'input_ids':perfix_token['input_ids']+mid['input_ids'][:self.max_length-len(perfix_token['input_ids'])-len(postfix_token_2['input_ids'])]+postfix_token_2['input_ids'],
                'attention_mask':perfix_token['attention_mask']+mid['attention_mask'][:self.max_length-len(perfix_token['attention_mask'])-len(postfix_token_2['attention_mask'])]+postfix_token_2['attention_mask'],
            }
        else:
            return {
                'input_ids':perfix_token['input_ids']+mid['input_ids']+postfix_token['input_ids'],
                'attention_mask':perfix_token['attention_mask']+mid['attention_mask']+postfix_token['attention_mask'],
            }

    def encode(self, str):
        answer = {
            "input_ids":[],
            "attention_mask":[],
        }
        if (type(str)==list):
            max_length = 0
            answers = []
            for i in str:
                answers.append(self.encode_single_sentence(i))
                max_length = max(max_length, len(answers[-1]['input_ids']))
            max_length = min(max_length, self.max_length)
            for subanswer in answers:
                paded_subanswer = self.padding(subanswer, max_length)
                answer['input_ids'].append(subanswer['input_ids'])
                answer['attention_mask'].append(subanswer['attention_mask'])
            ic(max_length)
        else:
            subanswer = self.encode_single_sentence(str)
            answer['input_ids'].append(subanswer['input_ids'])
            answer['attention_mask'].append(subanswer['attention_mask'])
            return answer
        return {'input_ids': torch.tensor(answer['input_ids']), 'attention_mask': torch.tensor(answer['attention_mask'])}

    def __call__(self, str):
        return self.encode(str)

# We want to do is to warp the dataset input text like the gpt2_large
# So we first tokenize by GPT2 tokenlizer, then convert it to the text
class GPT3TokenizerWarper(TokenizerWarper):
    def __init__(self, prompt, tokenizer_path = "../gpt2-large_saved/token"):
        super().__init__(prompt, tokenizer_path)

    def calc_single_str(self, str):
        token_ids = self.encode_single_sentence(str)['input_ids']
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(token_ids))

    def __call__(self, str):
        output_dict = {
            "input_prompts":[],
        }
        if (type(str)==list):
            for i in str:
                output_dict["input_prompts"].append(self.calc_single_str(i))
        else:
            output_dict["input_prompts"].append(self.calc_single_str(str))
        return output_dict

    def get_length(self, str):
        return len( self.encode_single_sentence(str)['input_ids'])-len(self.encode_single_sentence('')['input_ids'])

import openai
import os
def load_key(path):
    f = open(path, "r")
    return f.read()[:-1]
import time
class GPT3Warper(nn.Module):
    def __init__(self, open_ai_key_path,offset = [0, 0, 0, 0, 0]):
        super().__init__()
        openai.api_key = load_key(open_ai_key_path)
        self.output_path = "./1.json"
        ic(openai.api_key)

    def forward(self, input_prompts):
        completions = []
        import json

        with open(self.output_path, 'a') as outfile:
            for input_prompt in input_prompts:
                while (True):
                    try:
                        completion = openai.Completion.create(engine="text-davinci-002", prompt=input_prompt, temperature=1,
                                                              max_tokens=400)
                        break
                    except:
                        print("openai.error.ServiceUnavailableError: The server is overloaded or not ready yet.")
                        time.sleep(60)
                        continue

                json.dump(dict(completion), outfile)
                outfile.write('\n')
                #completions = completions[0]
                completion = [completion, input_prompt]
                completions.append(completion)
        #response = completion.choices[0].text
        return completions

if __name__=='__main__':
    from datasets import load_from_disk
    #ic(tokens)
    model = GPT3Warper("path2OpenAI_Key")
    input_prompts = ["Generate a long story in detail about Alice dated with Bob, and they have sex，trying a lot of things to play:"]
    print(model(input_prompts))
