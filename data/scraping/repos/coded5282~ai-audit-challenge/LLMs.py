from tqdm.auto import tqdm
from utils import List_Dataset

''' Base class '''
class LLM():
    def __init__(self, nout_per_prompt, max_tokens_per_prompt):
        self.nout_per_prompt = nout_per_prompt
        self.max_tokens_per_prompt = max_tokens_per_prompt
        return
    
    def generate(self, prompts, wrap_by_input=False, **kwargs):
        responses = self._generate(prompts, **kwargs)
        assert len(responses) == len(prompts) * self.nout_per_prompt
        assert type(responses) == type([])
        
        for r in responses:
            assert type(r) == type(()), r        
            assert type(r[0]) == type("prompt"), r
            assert type(r[1]) == type("response"), r
            assert type(r[2]) == type(00.00) or r[2] is None, r  
        
        if wrap_by_input:
            n = len(prompts)
            k = self.nout_per_prompt
            responses = [responses[i*k:(i+1)*k] for i in range(n)]
            
        return responses
    
    def _generate(self):
        raise NotImplementedError()
    
''' GPT 3 from Open AI '''
import numpy as np

# API docs: 
# https://beta.openai.com/docs/api-reference/completions/create

class GPT3(LLM):
    def __init__(self, nout_per_prompt, max_tokens_per_prompt, key, engine="text-ada-001"): 
        super().__init__(
            nout_per_prompt=nout_per_prompt, 
            max_tokens_per_prompt=max_tokens_per_prompt) 
        
        self.engine = engine
        self.key = key

        # more expensive but more performant: "text-davinci-002"
        return
    
    def _generate(self, prompts, temperature=0.7, stop='\n'):
        import openai
        openai.api_key = self.key

        responses = openai.Completion.create(
            engine=self.engine, 
            prompt=prompts, 
            max_tokens=self.max_tokens_per_prompt,
            temperature=0.7, 
            n=self.nout_per_prompt, 
            logprobs=1,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )['choices']

        prompts = [prompts[i] for i in range(len(prompts)) for j in range(self.nout_per_prompt)]
        outs = [x["text"] for x in responses]
        scores = [float(np.sum(x["logprobs"]["token_logprobs"])) for x in responses]
        return list(zip(prompts, outs, scores))

# gpt3 = GPT3()
# print(gpt3.generate(["black people are"]))

''' Base class for HuggingFace models '''
from transformers import pipeline
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# HuggingFace Pipeline API 
# https://huggingface.co/docs/transformers/v4.21.1/en/main_classes/pipelines#transformers.pipeline
# For generation specifically: 
# https://huggingface.co/docs/transformers/main_classes/text_generation

class HuggingFaceLLM(LLM):
    def __init__(self, engine, nout_per_prompt, num_beams, max_tokens_per_prompt, device, batch_size):
        super().__init__(nout_per_prompt=nout_per_prompt, max_tokens_per_prompt=max_tokens_per_prompt) 
        
        self.engine = engine
        self.device = device
        self.batch_size = batch_size
        self.num_beams = num_beams
        
        self.pipe = self._create_pipe(self.num_beams, self.nout_per_prompt, self.max_tokens_per_prompt)
        self.pipe.tokenizer.pad_token_id = self.pipe.model.config.eos_token_id
        return
    
    def _generate(self, prompts):
        # wrap so can get progress bar
        prompts = List_Dataset(prompts)
        responses = tqdm(self.pipe(prompts))
        
        outs = [x['generated_text'] for lst in responses for x in lst]
        prompts = [prompts[i] for i in range(len(prompts)) for j in range(self.nout_per_prompt)]
        scores = [None for x in outs]
        
        return list(zip(prompts, outs, scores))
    
    def _create_pipe(self, num_beams, nout_per_prompt, max_tokens_per_prompt):
        '''
        greedy decoding by calling greedy_search() if num_beams=1 and do_sample=False.
        multinomial sampling by calling sample() if num_beams=1 and do_sample=True.
        beam-search decoding by calling beam_search() if num_beams>1 and do_sample=False.
        beam-search multinomial sampling by calling beam_sample() if num_beams>1 and do_sample=True.
        diverse beam-search decoding by calling group_beam_search(), if num_beams>1 and num_beam_groups>1.
        constrained beam-search decoding by calling constrained_beam_search(), if constraints!=None or force_words_ids!=None.
        '''
        
        if nout_per_prompt > 1: assert num_beams > nout_per_prompt

        kwargs = {
            'task' : 'text-generation', 
            'model' : self.engine, 
            'do_sample' : False, 
            'num_beams' : num_beams,
            # 'num_return_sequences' : self.nout_per_prompt,
            'max_new_tokens' : self.max_tokens_per_prompt, 
            'output_scores' : True,
            'batch_size' : self.batch_size,
            'repetition_penalty' : 2.0,
        }
        
        if self.device != 'cpu':
            assert type(self.device) == type(0)
            kwargs['device'] = self.device
            
        return pipeline(**kwargs)
        
    def change_nout_per_prompt(self, new_nout_per_prompt):
        self.pipe = self._create_pipe(self.new_nout_per_prompt, self.max_tokens_per_prompt)
        return
        
''' OPT from Facebook '''

class OPT(HuggingFaceLLM):
    def __init__(self, nout_per_prompt, num_beams, max_tokens_per_prompt, device='cpu', batch_size=128):
        super().__init__(
            # engine = "facebook/opt-350m",
            engine = "facebook/opt-1.3b",
            # engine = "facebook/opt-2.7b",
            nout_per_prompt = nout_per_prompt, 
            num_beams = num_beams,
            max_tokens_per_prompt = max_tokens_per_prompt,
            device = device,
            batch_size = batch_size
        ) 
        return

# opt = OPT()
# print(opt.generate(["black people are"]))

''' GPT2 from OpenAI '''

class GPT2(HuggingFaceLLM):
    def __init__(self, nout_per_prompt, num_beams, max_tokens_per_prompt, device='cpu', batch_size=128):
        super().__init__(
            engine = "gpt2", 
            nout_per_prompt = nout_per_prompt, 
            num_beams = num_beams,
            max_tokens_per_prompt = max_tokens_per_prompt,
            device = device,
            batch_size = batch_size
        ) 
        return

# gpt2 = GPT2()
# print(gpt2.generate(["black people are"]))