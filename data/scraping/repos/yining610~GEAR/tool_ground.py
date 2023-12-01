from pattern import PATTERN_PROB, PATTERN
from api import BaseAPI
from utils import _extract_api_request_content, _process_potential_answers

import argparse

import numpy as np
import re
# import warnings
# warnings.filterwarnings("ignore")

import torch
from torchtyping import TensorType

from typing import List, Tuple, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer, util
from tool_execute import OpenAI_Zero_Executor
import time

CACHE_DIR="/scratch/ylu130/huggingface/models"

class APIFilter:
    def __init__(self, args: argparse.Namespace, apis: List[BaseAPI]):
        self.args = args
        self.apis = apis
        # order the apis by if the prompt_templete is None
        self.apis.sort(key=lambda x: x.prompt_template is None)
        self.device = torch.device(args.fdevice if torch.cuda.is_available() else "cpu")
        self.nli_model = SentenceTransformer(args.slm2)
        if "chatgpt" in args.slm1:
            self.lm_model = OpenAI_Zero_Executor(args)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(args.slm1, padding_side='left', cache_dir=CACHE_DIR)
            self.tokenizer.pad_token = self.tokenizer.eos_token

            self.lm_model = AutoModelForCausalLM.from_pretrained(args.slm1, pad_token_id = self.tokenizer.eos_token_id,cache_dir=CACHE_DIR).to(self.device)
        
        self.max_tokens = args.max_tokens

        self.verbose = args.verbose
        self.top_k = args.top_k

    def _encode_patterns(self, input: str, prior_pattern: str=None) -> Dict[str, int]:
        encoded_pattern = {}
        if input is None:
            for pattern_type in PATTERN.keys():
                encoded_pattern[pattern_type] = 0
        else:
            if prior_pattern is None:
                for pattern_type, pattern_regex in PATTERN.items():
                    encoded_pattern[pattern_type] = len(pattern_regex.findall(input))
            else:
                # API has specifc output pattern
                for pattern_type, pattern_regex in PATTERN.items():
                    encoded_pattern[pattern_type] = len(pattern_regex.findall(input)) if pattern_type == prior_pattern else 0

        return encoded_pattern
    
    def generate_call(self, input: str) -> List[str]:
        """
        Put input question and api prompt into language model and return the generated API calls
        Args:
            input (str): input question
        Returns:
            List[str]: Generated API calls
        """
        if "chatgpt" not in self.args.slm1:

            eos_token_id = self.tokenizer("\n")["input_ids"][0] # eos_token_id list for EleutherAI/gpt-neo-1.3B

            # batch generation for all APIs
            prompts_batch = [api.prompt_template.format(input=input) for api in self.apis if api.prompt_template is not None]
            inputs_batch = self.tokenizer(prompts_batch, padding=True, return_tensors="pt")
            PROMPT_LENGTH = len(inputs_batch['input_ids'][0])

            mtokens = self.max_tokens
            while mtokens > 0:
                try:
                    outputs_ids = self.lm_model.generate(inputs_batch['input_ids'].to(self.device),
                                                         attention_mask=inputs_batch['attention_mask'].to(self.device),
                                                         do_sample=True,
                                                         top_p=0.9,
                                                         eos_token_id = eos_token_id,
                                                         max_new_tokens = mtokens)
                    outputs_ids = outputs_ids[:, PROMPT_LENGTH:]
                    outputs = self.tokenizer.batch_decode(outputs_ids, skip_special_tokens=True)
                    break
                except:
                    outputs = [""] * len(prompts_batch)
                    mtokens = mtokens // 2
                    print(f"New Max tokens: {mtokens}")
                    continue
            
        else:
            outputs = []
            for api in self.apis:
                if api.prompt_template is not None:
                    outputs.append(self.lm_model.execute(input, api.prompt_template))
    
        # for apis with no prompt_template, use the input question as the API call
        if len(outputs) < len(self.apis):
            outputs.extend([None] * (len(self.apis) - len(outputs)))

        if self.verbose:
            for i in range(len(self.apis)):
                print(f"{self.apis[i].name} API Call: {outputs[i]}")

        return outputs

    def obtain_api_response(self, input: str, outputs: List[str]) -> List[Optional[str]]:
        """
        Obtain the response from generated outputs
        """
        api_responses = []

        for api, output in zip(self.apis, outputs):
            if api.prompt_template is None: # APIs do not need API calls from LM
                request_args = None
                if api.name != "MultilingualQA":
                    api_response = api(input)  
                else: 
                    trying = 0
                    while trying < 3:
                        try:
                            api_response=api(input,
                                         mtdest = 'en', 
                                         mtsrc = api.apis['MT'].translator.detect(re.search('question: (.*)context:', input).group(1)).lang if re.search('question: (.*)context:', input) is not None else 'en')
                            break
                        except:
                            api_response = None
                            trying += 1
                            time.sleep(60)
            else:                           # APIs need generated API calls
                request_args = _extract_api_request_content(output, api.name)
                if request_args is None:
                    api_response = None
                else:
                    api_response = api(*request_args) if api.name != "MultilingualQA" else None


            if self.verbose:
                print(f"{api.name} request content: {request_args}")
                print(f"{api.name} response: {api_response}")
            api_responses.append(api_response)

        return api_responses

    def semantic_similarity_score(self, input: str) -> TensorType["num_apis"]:
        """
        Compute the semantic score between the input and the API description. 
        Args:
            input (str): the input string to be matched with the API description.
        Returns:
            List[float]: the semantic similarity score between the input and the API description.
        """
        input_batch = [input] * len(self.apis)
        description_batch = [api.description for api in self.apis]

        input_emb = self.nli_model.encode(input_batch, convert_to_tensor=True)
        description_emb = self.nli_model.encode(description_batch, convert_to_tensor=True)

        #Compute cosine-similarities
        cosine_scores = util.cos_sim(input_emb, description_emb)
        #Compute cosine-similarities
        cosine_scores = util.cos_sim(input_emb, description_emb)
        
        if self.verbose:
            for i in range(len(input_batch)):
                print("{} \t\t Semantic Score: {:.4f}".format(self.apis[i].name, cosine_scores[i][i]))
        
        return cosine_scores[0].cpu().numpy()

    def encode_answers(self, api_responses: List[Optional[str]], candidate: Optional[str]) -> Tuple[List[Optional[Dict[str, int]]], Optional[Dict[str, int]]]:
        """
        Encode the api responses and candidate answers based on token patterns
        """ 
        encoded_response_patterns = []

        for api_index, api_response in enumerate(api_responses):
            # encode all tokens to the same pattern for the special API
            if (self.apis[api_index].pattern is not None) and (api_response is not None):
                encoded_pattern = self._encode_patterns(api_response, self.apis[api_index].pattern)
            elif (self.apis[api_index].pattern is None) and (api_response is not None):
                # unmeaningful response
                if api_response.lower().strip() in ["", "unknown", "none", "no answer"]: 
                    encoded_pattern = None
                else:
                    encoded_pattern = self._encode_patterns(api_response)
            else:
                encoded_pattern = None
            
            if self.verbose:
                print(f"{self.apis[api_index].name}: (Orignal Response: {api_response}, Encoded Response: {encoded_pattern})")

            encoded_response_patterns.append(encoded_pattern)
        
        # encode the candidate answer
        encoded_candidate_response = self._encode_patterns(candidate)
        if self.verbose:
            print(f"Potential Answer: (Orignal candidate: {candidate} Encoded candidate: {encoded_candidate_response})")

        return encoded_response_patterns, encoded_candidate_response

    def pattern_similarity_score(self, response_patterns: List[Optional[Dict[str, int]]], candidate_pattern: Optional[Dict[str, int]]) -> List[float]:

        candidate_length = sum(candidate_pattern.values())
        pattern_similarity_scores = []

        for api_index in range(len(response_patterns)):
            if response_patterns[api_index] is None:
                pattern_similarity_scores.append(0) # assign the minimum pattern score if the response is empty
                continue
            else:
                pattern_similarity_score = 0
                # find the number of each pattern in the response
                pattern_count = response_patterns[api_index]
                response_length = sum(response_patterns[api_index].values())
                # compute the pattern similarity score
                for pattern in PATTERN_PROB.keys():
                    # add-1 smoothing
                    pattern_similarity_score += pattern_count[pattern] * (candidate_pattern[pattern]+1) / ((candidate_length+len(PATTERN_PROB)) * response_length) * np.log(1 / PATTERN_PROB[pattern])
            
                pattern_similarity_scores.append(pattern_similarity_score)
        
        if self.verbose:
            for i in range(len(self.apis)):
                print(f"{self.apis[i].name} pattern similarity score: {pattern_similarity_scores[i]}")
        
        return pattern_similarity_scores

    def filter(self, input: str) -> Tuple[List[BaseAPI], Dict[str, List[float]]]:
        """
        Filter the APIs by two similarity scores
        Args:
            input (str): input question
        Returns:
            List[BaseAPI]: list of APIs that pass the filtering
        """
        if "chatgpt" not in self.args.slm1:
            # generate potential answers from LM1 without prompt. Open-domain QA
            input_ids = self.tokenizer.encode(input, return_tensors="pt").to(self.device)
            eos_token_id = self.tokenizer(".\n")["input_ids"][0] 

            with torch.no_grad():
                output_ids_without_prompt = self.lm_model.generate(input_ids=input_ids,
                                                                   do_sample = False,
                                                                   eos_token_id=eos_token_id,
                                                                   max_new_tokens =self.max_tokens)

            output_ids_without_prompt = output_ids_without_prompt[:, input_ids.shape[1]:]
            output_without_prompt = self.tokenizer.decode(output_ids_without_prompt[0], skip_special_tokens=True).strip()
            output_without_prompt = _process_potential_answers(output_without_prompt)
        else:
            output_without_prompt = self.lm_model.execute(input)
        
        if self.verbose:
            print("***********************Start Filtering***********************")
            print(f"Input query: {input}")
            print(f"Potential answers: {output_without_prompt}")
        
        generated_outputs = self.generate_call(input)

        api_responses = self.obtain_api_response(input, generated_outputs)
        assert len(api_responses) == len(self.apis), "api_responses and apis list should have the same length"
        
        # encode the api responses and candidate answers
        encoded_response_patterns, encoded_candidate_pattern = self.encode_answers(api_responses, output_without_prompt)

        # compute the semantic similarity score between the input question and the API description
        semantic_similarity_scores = self.semantic_similarity_score(input)
        # compute the pattern similarity score between the api responses and the candidate answer
        pattern_similarity_scores = self.pattern_similarity_score(encoded_response_patterns, encoded_candidate_pattern)

        final_similarity_scores = [semantic_similarity_scores[i]*self.args.ALPHA + pattern_similarity_scores[i]*(1-self.args.ALPHA) for i in range(len(self.apis))]

        if self.verbose:
            for i in range(len(self.apis)):
                print(f"{self.apis[i].name} final similarity score: {final_similarity_scores[i]}")

        filtered_apis = [self.apis[i] for i in np.argsort(final_similarity_scores)[-self.top_k:]]

        filtered_apis_with_scores = {self.apis[i].name: [float(semantic_similarity_scores[i]), float(pattern_similarity_scores[i])] for i in np.argsort(final_similarity_scores)[-self.top_k:]}
        if self.verbose:
            print(filtered_apis_with_scores)
            print("***********************End filtering***********************")
        return filtered_apis, filtered_apis_with_scores
