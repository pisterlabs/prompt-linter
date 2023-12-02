from dataclasses import dataclass
import torch
import lmql
from lmql.runtime.postprocessing.conditional_prob import ScoringQuery
import lmql.runtime.bopenai as openai
import asyncio
import sys
import numpy as np
from lmql.utils.nputil import log_softmax
from utils.openai_baseline_prompter import OpenAIPrompter

import lmql.runtime.dclib as dc
from lmql.runtime.hf_integration import transformers_model

@dataclass
class MockLMQLResult:
    distribution_variable: str
    variables: any

def HFPrompter(model, **kwargs):
    if model.startswith("openai/"):
        return OpenAIPrompter(model)
    return DCPrompter(model, **kwargs)

class DCPrompter:
    def __init__(self, model, include_bos_token=False):
        self.client = lmql.model_registry.get(model)
        
        self.last_num_steps = -1
        self.model = lmql.model_registry.get(model)
        # self.model = transformers_model("http://localhost:8080", model)()
        self.model.set_decoder("argmax")
        self.dcmodel = self.model.get_dclib_model()
        self.include_bos_token = include_bos_token

    async def generate(self, prompt, remove_stopping_phrases = True, truncate=None, **kwargs):
        input_ids = np.array(([self.client.bos_token_id] if self.include_bos_token else []) + await self.client.tokenize(prompt), dtype=np.int64).reshape(-1)

        if truncate is not None: input_ids = input_ids[:, -truncate:]

        max_new_tokens = kwargs.get("max_new_tokens", 1)
        if "max_new_tokens" in kwargs: del kwargs["max_new_tokens"]
        stopping_phrases = kwargs.get("stopping_phrases", None)
        if "stopping_phrases" in kwargs: del kwargs["stopping_phrases"]
        max_length = len(input_ids) + max_new_tokens
        step_size = kwargs.get("step_size", 1)
        if "step_size" in kwargs: del kwargs["step_size"]

        num_steps = 0
        s = dc.seqs([dc.seq(input_ids)])
        at_eos = False

        while len(s.item(0).input_ids) < max_length:
            max_length_step = min(max_length, len(input_ids) + step_size)

            self.model.served_model.num_generate_calls += 1
            
            for _ in range(step_size):
                s = s.extend(await self.dcmodel.argmax(s))
                if dc.eos(s.item(0)): 
                    s = dc.seqs([dc.seq(s.item(0).input_ids[:-1].reshape(-1))])
                    at_eos = True
                    break
            self.model.served_model.billable_tokens += len(s.item(0).input_ids) - (1 if self.include_bos_token else 0)

            # input_ids = await self.model.generate(input_ids, max_length = max_length_step, do_sample=False, num_return_sequences=1, early_stopping=True, **kwargs)
            input_ids = s.item(0).input_ids
            if self.include_bos_token:
                text = await self.client.detokenize(input_ids.tolist()[1:])
            else:
                text = await self.client.detokenize(input_ids.tolist())
            
            if len(input_ids) >= max_length:
                break

            for sp in stopping_phrases:
                if sp.startswith("keep:"):
                    sp = sp[5:]
                if sp is not None and sp in text[len(prompt):]:
                    break
            num_steps += 1

            if at_eos: break

        self.last_num_steps = num_steps
    
        if remove_stopping_phrases:
            generated_text = text[len(prompt):]
            truncation_index = len(generated_text)
            # remove stop phrase and everything after it
            for sp in stopping_phrases:
                keep = False
                if sp.startswith("keep:"):
                    sp = sp[5:]
                    keep = True
                if sp is not None and sp in generated_text:
                    if keep:
                        truncation_index = min(truncation_index, generated_text.index(sp) + len(sp))
                    else:
                        truncation_index = min(truncation_index, generated_text.index(sp))
            text = text[:len(prompt) + truncation_index]
        
        return text

    async def cond_logprob(self, prompt, values):
        result = MockLMQLResult("PREDICTION", {})
        scorer = ScoringQuery(result, 0, prompt, values, None)
        scores = await scorer.score(self.model, batch_size=1)

        scores = np.stack([s.sum() for s in scores], axis=0)
        log_probs = log_softmax(scores)

        return list(zip(values, log_probs.tolist()))