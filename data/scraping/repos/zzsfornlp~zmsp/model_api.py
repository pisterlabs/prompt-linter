#

# remote model by calling api

__all__ = [
    'NmApiConf', 'NmApi', 'match_tokens_to_str'
]

from mspx.utils import zwarn
from mspx.nn import NnConf, NnLayer
from .base import *
import os
import json

@NnConf.rd('nm_api')
class NmApiConf(NewBaseModelConf):
    def __init__(self):
        super().__init__()
        # --
        self.model_name = 'local-llama'
        self.llama_addr = "http://localhost:8009"

# --
# helper
def match_tokens_to_str(toks, s, warning=True):
    ret = []
    idx_char = 0
    hit_end = False
    for ii, tt in enumerate(toks):
        tt = tt.strip()
        # print(hit_end)
        if tt == '' or hit_end:
            ret.append(None)  # no matching for spaces!
            continue
        try:
            new_i0 = s.index(tt, idx_char)
            idx_char = new_i0 + len(tt)  # get it!
            ret.append([new_i0, idx_char])  # [start, end)
        except:
            _remain_s0 = s[idx_char:].strip()
            ret.append(None)
            if _remain_s0 == "":  # ok, running out s0!
                hit_end = True
            elif warning:  # still have s0, maybe slight mismatch!
                zwarn(f"Unmatched tok of {tt} against {_remain_s0}")
    return ret
# --

@NmApiConf.conf_rd()
class NmApi(NewBaseModel):
    def __init__(self, conf):
        super().__init__(conf)
        conf: NmApiConf = self.conf
        # --

    def _get_logprob(self, inputs0, inputs1, results):
        # align subtoks to tokens
        assert len(inputs0) == len(results) and len(inputs1) == len(results)
        ret = []
        for one_input0, one_input1, one_result in zip(inputs0, inputs1, results):
            choice = one_result['choices'][0]
            choice_tokens, choice_logprobs = choice['logprobs']['tokens'], choice['logprobs']['token_logprobs']
            match0 = match_tokens_to_str(choice_tokens, one_input0)
            # locate the split point
            split_idx = len(match0)
            while split_idx>0 and match0[split_idx-1] is None:
                split_idx -= 1
            while split_idx<len(match0) and choice_tokens[split_idx].strip()=="":
                split_idx += 1
            # check
            valid_toks_after = [z.strip() for z in choice_tokens[split_idx:] if z.strip() != ""]
            if len(valid_toks_after)>0 and not one_input1.strip().startswith(valid_toks_after[0]):
                zwarn(f"Maybe mismatches with {choice_tokens[split_idx:]} vs {one_input1}")
            # --
            _DEBUG = 0
            if _DEBUG:
                print(f"#--\n{one_input0}\n{choice_tokens[:split_idx]}\n{one_input1}\n{choice_tokens[split_idx:]}\n#--")
            # --
            ret.append(choice_logprobs[split_idx:])
            # --
            if _DEBUG:
                breakpoint()
            # --
        return ret

    def run_logprob(self, inputs0, inputs1, avg_tok=True):
        conf: NmApiConf = self.conf
        # --
        cat_inputs = [(a+b) for a,b in zip(inputs0, inputs1)]
        if conf.model_name == 'local-llama':
            import requests
            responses = json.loads(requests.post(conf.llama_addr, data=json.dumps({"prompts": cat_inputs, "max_tokens": 0})).text)['rets']
        else:  # use openai api
            import openai
            openai.api_key = os.getenv("OPENAI_API_KEY")
            responses = []
            for one_input in cat_inputs:
                response = openai.Completion.create(
                    model=conf.model_name,
                    prompt=one_input,
                    temperature=0.,
                    top_p=1,
                    max_tokens=0,
                    logprobs=0,
                    echo=True,
                )
                responses.append(response)
        # --
        ret = self._get_logprob(inputs0, inputs1, responses, avg_tok=avg_tok)
        return ret, responses
