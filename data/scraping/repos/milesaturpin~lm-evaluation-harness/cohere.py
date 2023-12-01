import os
import numpy as np
import transformers
from lm_eval.base import LM
from lm_eval import utils
from tqdm import tqdm
import sys
import cohere
# sys.path.append('/home/miles_cohere_ai/wit')
# from wit.data.encoders import OpenWebTextEncoder
from lm_eval.tokenizer import OpenWebTextEncoder
import time

class CohereTokenizer():

    def __init__(self):
        """
        
        TODO: just import the code for this to make the dependencies contained

        """
        # self.co = co
        self.encoder = OpenWebTextEncoder(encoder_dir='gs://cohere-prod/encoders/coheretext50k-30Mplus1')

    def num_tokens(self, text):
        # return len(self.co.likelihood(model='baseline-shrimp', text=text).token_likelihoods)
        return len(self.encode(text))

    def encode(self, text):
        return self.encoder.encode(text)

    def decode(self, tokens):
        return self.encoder.decode(tokens)

    # def split_string_by_tokens(self, text):
    #     return [x['token'] for x in self.co.likelihood(model='baseline-shrimp', text=text).token_likelihoods]

    # def truncate_by_tokens(self, text, max_tokens, return_num_truncated=False):
    #     """Truncates by cutting off the FRONT, not the back."""
    #     if max_tokens is None or not text:
    #         return text
    #     text_split = self.split_string_by_tokens(text)
    #     new_text_length = min(max_tokens, len(text_split))
    #     text_split = text_split[-new_text_length:]
    #     text_new = "".join(text_split)
    #     if return_num_truncated:
    #         return text_new, max(0, len(text_split) - max_tokens)
    #     else:
    #         return text_new

    def truncate_by_tokens(self, text, max_tokens, return_num_truncated=False):
        """Truncates by cutting off the FRONT, not the back."""
        if max_tokens is None or not text:
            return text
        text_split = self.split_string_by_tokens(text)
        new_text_length = min(max_tokens, len(text_split))
        text_split = text_split[-new_text_length:]
        text_new = "".join(text_split)
        if return_num_truncated:
            return text_new, max(0, len(text_split) - max_tokens)
        else:
            return text_new


def get_result(response, ctxlen):
    # import ipdb; ipdb.set_trace()
    is_greedy = True
    # logprobs = response["logprobs"]["token_logprobs"]
    logprobs = [x['likelihood'] if 'likelihood' in x.keys() else 0.0 for x in response.token_likelihoods]
    #scalar
    continuation_logprobs = sum(logprobs[ctxlen:])

    # for i in range(ctxlen, len(response.token_likelihoods)):
    #     token = response.token_likelihoods[i]['token']
    #     top_tokens = response["logprobs"]["top_logprobs"][i]
    #     top_token = max(top_tokens.keys(), key=lambda x: top_tokens[x])
    #     if top_token != token:
    #         is_greedy = False
    #         break
    is_greedy=False
    
    return continuation_logprobs, is_greedy


def oa_completion(**kwargs):
    import openai

    backoff_time = 3
    while True:
        try:
            return openai.Completion.create(**kwargs)
        except openai.error.OpenAIError:
            time.sleep(backoff_time)
            backoff_time *= 1.5

def cohere_likelihood(co, **kwargs):
    
    attempts_remaining = 5
    backoff_time = 3
    while True:
        try:
            # assert False
            return co.likelihood(**kwargs)
        except cohere.error.CohereError as e:
            print('TRYING AGAIN')
            time.sleep(backoff_time)
            backoff_time *= 1.5
            attempts_remaining =- 1
            if attempts_remaining == 0:
                raise e
        except Exception as e:
            raise e

    print('MISSED ALL EXCEPTION CATCHING')
    

# def get_likelihood_helper_global(example):

def get_likelihood_helper(example):
    """Compute likelihood of continuation for a single example"""
    # import ipdb; ipdb.set_trace()
    # inps = []
    # ctxlens = []
    # for cache_key, context_enc, continuation_enc in example:
    cache_key, context_enc, continuation_enc, max_length, co, model, tokenizer = example

    # import cohere tokenizer method
    # inp = (context_enc + continuation_enc)[-self.MAX_LENGTH:]
    inp = (context_enc + continuation_enc)[-max_length:]
    inp = tokenizer.decode(inp)
    ctxlen = len(context_enc) - max(0, len(context_enc) + len(continuation_enc) - max_length)
    
    # Cohere API likelihood "tokenizer" method
    # returning num truncated here saves having to do another call to num_tokens later
    # TODO: maybe just try using GPT2 tokenizer and just do error handling to skip examples that fail
    # inp, num_truncated = self.tokenizer.truncate_by_tokens(
    #     cache_key[0] + cache_key[1], self.MAX_LENGTH, return_num_truncated=True)
    # # subtract off how much longer the full thing is from the max length,
    # context_len = self.tokenizer.num_tokens(cache_key[0])
    # # continuation_len = self.tokenizer.num_tokens(cache_key[1])
    # ctxlen = context_len - num_truncated

    # TODO: am just subtracting 1 but really should be len(answer) [actually not sure about this]
    # ctxlen = self.tokenizer.num_tokens(inp) - 1

    # inps.append(inp)
    # ctxlens.append(ctxlen)

    # TODO: use multiprocessing to process chunks

    # response = oa_completion(
    #     engine=self.engine,
    #     prompt=inps,
    #     echo=True,
    #     max_tokens=0, temperature=0.,
    #     logprobs=10,
    # )
    # chunks are actually just size 1 because we don't have batching
    response = cohere_likelihood(co, model=model, text=inp)
    # response = co.likelihood(
    #     model=model,
    #     text=inp # chunks are actually just size 1 because we don't have batching
    # )

    # for resp, ctxlen, (cache_key, context_enc, continuation_enc) in zip(response, ctxlens, chunk):
        # import ipdb;ipdb.set_trace()
    answer = get_result(response, ctxlen)

    # TODO: this???
    # res.append(answer)

    # partial caching
    # if cache_key is not None:
    #     self.cache_hook.add_partial("loglikelihood", cache_key, answer)

    # return answer
    # return cache_key with the answer so gets passed to callback for caching
    return (cache_key, answer)



class CohereLM(LM):

    MAX_LENGTH = 2048
    REQ_CHUNK_SIZE = 1 #20
    MAX_GEN_TOKS = 256

    def __init__(self, model, truncate=False):
        """

        :param engine: str
            OpenAI API engine (e.g. davinci)
        :param truncate: bool
            Truncate input if too long (if False and input is too long, throw error)
        """
        super().__init__()
        import cohere
        self.model = model
        self.MAX_LENGTH = 1024 if self.model in ['baseline-shrimp', 'baseline-otter'] else 2048
        apikey = os.environ["COHERE_API_SECRET_KEY"]
        self.co = cohere.CohereClient(apikey)
        # self.tokenizer = CohereTokenizer(self.co)
        self.tokenizer = CohereTokenizer()


        # to make the annoying "Using pad_token, but it is not set yet." error go away
        # self.tokenizer.pad_token = "<|endoftext|>"
        # assert self.tokenizer.encode('hello\n\nhello') == [31373, 198, 198, 31373]
        self.truncate = truncate
        # self.end_of_text_token_id = self.tokenizer.convert_tokens_to_ids(["<|endoftext|>"])[0]

        # Read from environment variable OPENAI_API_SECRET_KEY
        # openai.api_key = os.environ["OPENAI_API_SECRET_KEY"]

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config={}):
        args = utils.simple_parse_args_string(arg_string)
        args2 = {k: v for k, v in additional_config.items() if v is not None}
        return cls(**args, **args2)

    def loglikelihood(self, requests):
        new_reqs = []
        # import ipdb; ipdb.set_trace()
        # requests = requests[:10]
        for context, continuation in requests:
            if context == "":
                # end of text as context
                context_enc = [50256]
            else:
                context_enc = self.tokenizer.encode(context)
                # context_enc = None

            continuation_enc = self.tokenizer.encode(continuation)
            # continuation_enc = None

            # TODO: just pass CohereLM as argument, self
            new_reqs.append(((context, continuation), context_enc, continuation_enc, self.MAX_LENGTH, self.co, self.model, self.tokenizer))

        return self._loglikelihood_tokens(new_reqs)

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def test(self, *args):
        print(self.MAX_LENGTH)

    def _loglikelihood_tokens(self, requests):
        # import openai
        # import ipdb; ipdb.set_trace()
        import multiprocessing as mp
        pool = mp.Pool(mp.cpu_count())
        res = []

        def _collate(x):
            # this doesn't efficiently handle last-token differences yet, but those are kinda annoying because
            # it's not guaranteed that the 100 or so logprobs we get to see actually contain all the continuations
            # we care about and so we need some kind of backup for when it isn't
            # import ipdb; ipdb.set_trace()
            text = x[0][0] + x[0][1]
            return (-self.tokenizer.num_tokens(text), tuple(text))
        
        # import ipdb; ipdb.set_trace()
        # I think this is literally just to sort by length of example
        # reord = utils.Reorderer(requests, _collate)

        # for chunk in tqdm(list(utils.chunks(reord.get_reordered(), self.REQ_CHUNK_SIZE))):
            # import ipdb; ipdb.set_trace()

        # results = list(tqdm(pool.imap(get_likelihood_helper, requests), total=len(requests)))


        pbar = tqdm(total=len(requests))

        def cache_and_update_pbar(res):
            pbar.update()
            cache_key, answer = res
            # import ipdb; ipdb.set_trace()
            if cache_key is not None:
                self.cache_hook.add_partial("loglikelihood", cache_key, answer)
        # results = pool.map_async(get_likelihood_helper, requests, callback=update_pbar)
        # results = results.get()

        results = [pool.apply_async(get_likelihood_helper, args=(request,), callback=cache_and_update_pbar) for request in requests]
        # get rid of cache_key

        # import ipdb; ipdb.set_trace()
        results = [result.get()[1] for result in results]
        # for example, answer in zip(requests, results):
        #     # import ipdb;ipdb.set_trace()
        #     cache_key = example[0]
        #     if cache_key is not None:
        #         self.cache_hook.add_partial("loglikelihood", cache_key, answer)

        # return reord.get_original(res)
        return results

    def get_token_logprobs(self, input_tokens, pred_tokens):
        pred_start = len(input_tokens) - len(pred_tokens) + 1
        # We're going to stitch together the input_tokens and pred_tokens
        # In the longest case, this gets us to length = max_seq_len+1 (which the API works with)
        assert input_tokens[pred_start:] == pred_tokens[:-1]
        token_ids = input_tokens + [pred_tokens[-1]]
        response = oa_completion(
            engine=self.engine,
            prompt=token_ids,
            max_tokens=0,
            temperature=0.0,
            logprobs=0,
            echo=True,
        )
        logprobs = np.array(response["choices"][0]["logprobs"]["token_logprobs"][pred_start:])
        positions = np.arange(pred_start-1, pred_start-1 + len(token_ids[pred_start:]))
        return {
            "logprobs": logprobs,
            "positions": positions,
        }

    def greedy_until(self, requests):
        raise NotImplementedError
