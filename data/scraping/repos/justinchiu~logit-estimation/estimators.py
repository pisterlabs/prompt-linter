from abc import ABC, abstractmethod
from dataclasses import dataclass
import sys
import math
import torch
import numpy as np
from scipy.special import logsumexp
import tiktoken
import openai
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
#from rich.progress import track

from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.distributions import Categorical, kl_divergence


np.seterr(all='raise')

@dataclass
class Output:
    samples: list[int] | None = None
    argmax: int | None = None
    true_dist: torch.distributions.Distribution | None = None


class Sampler(ABC):
    @abstractmethod
    def sample(self, prefix, n, logit_bias, temperature):
        ...

def construct_logit_bias_tensor(logit_bias_dict, vocab_size):
    logit_bias = torch.zeros(vocab_size)
    for idx, value in logit_bias_dict.items():
        logit_bias[idx] = value
    return logit_bias

class HfSampler(Sampler):
    def __init__(self, model):
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16)
        #self.model = AutoModelForCausalLM.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.vocab_size = len(self.tokenizer)
        self.cached_logits = None
        self.cached_prefix = None

    def sample(self, prefix, K, logit_bias=None, temperature=1):
        if prefix != self.cached_prefix:
            self.cached_logits = None
            self.cached_prefix = prefix

        logits = (
            self.get_true_logits(prefix, K)
            if self.cached_logits is None
            else self.cached_logits
        )
        if logit_bias is not None:
            #logits = logits + construct_logit_bias_tensor(logit_bias, len(logits))
            logits = logits + torch.tensor(logit_bias)

        true_dist = Categorical(logits=logits)

        return Output(
            samples = true_dist.sample_n(K).tolist() if temperature > 0 else None,
            argmax = logits.argmax(-1),
            true_dist = true_dist,
        )

    def get_true_logits(self, prefix, K):
        inputs = self.tokenizer([prefix], return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=True,
            num_return_sequences=1,
            top_p=1.0,
            top_k=0,
            num_beams=1,
        )
        logits = outputs.scores[0][0].log_softmax(0)
        self.cached_logits = logits
        return logits

class GptSampler(Sampler):
    def __init__(self, model):
        enc = tiktoken.encoding_for_model(model)
        self.model = model
        self.vocab_size = enc.n_vocab

    def topk(self, prefix, logit_bias=None, temperature=1, system=None):
        model = self.model

        enc = tiktoken.encoding_for_model(model)
        if model == "gpt-3.5-turbo-instruct":
            if logit_bias is not None:
                response = openai.Completion.create(
                    model=model,
                    prompt=prefix,
                    temperature=temperature,
                    max_tokens=1,
                    logit_bias=logit_bias,
                    logprobs=5,
                )
            else:
                response = openai.Completion.create(
                    model=model,
                    prompt=prefix,
                    temperature=temperature,
                    max_tokens=1,
                    logprobs=5,
                )
        topk_dict = response.choices[0].logprobs.top_logprobs[0]
        #print(topk_dict)
        # convert to index representation
        return {
            enc.encode(x)[0]: y
            for x,y in topk_dict.items()
        }

    def sample(self, prefix, K, logit_bias=None, temperature=1, system=None):
        model = self.model
        system = "You are a helpful assistant." if system is None else system

        enc = tiktoken.encoding_for_model(model)
        if model == "gpt-3.5-turbo-instruct":
            if logit_bias is not None:
                response = openai.Completion.create(
                    model=model,
                    prompt=prefix,
                    temperature=temperature,
                    max_tokens=1,
                    logit_bias=logit_bias,
                    n=K,
                )
            else:
                response = openai.Completion.create(
                    model=model,
                    prompt=prefix,
                    temperature=temperature,
                    max_tokens=1,
                    n=K,
                )
            output = response.choices[0].text
            eos_idx = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>", "<|im_start|>"})[0]
            outputs = [choice.text for choice in response.choices]
        else:
            if logit_bias is not None:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prefix},
                    ],
                    temperature=temperature,
                    max_tokens=1,
                    logit_bias=logit_bias,
                    n=K,
                )
            else:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prefix},
                    ],
                    temperature=temperature,
                    max_tokens=1,
                    n=K,
                )
            output = response.choices[0].message["content"]
            outputs = [choice.message["content"] for choice in response.choices]
            eos_idx = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>", "<|im_start|>"})[0]

        # just give eos_idx if there's a weird failure
        if response.choices[0].finish_reason == "length":
            idx = enc.encode(output)[0] if output else eos_idx
        elif response.choices[0].finish_reason == "stop":
            idx = eos_idx
        else:
            import pdb; pdb.set_trace()
        #print(response)

        if temperature > 0:
            samples = [enc.encode(output) for output in outputs]

        return Output(
            samples = samples if temperature > 0 else None,
            argmax = idx if temperature == 0 else None,
            true_dist = None,
        )


class Estimator:
    def __init__(self, vocab_size, threshold = 50, estimated_logits=None, idxs=None):
        self.vocab_size = vocab_size
        self.samples = []
        self.log_weights = []

        self.Zs = np.zeros(vocab_size, dtype=np.float64)
        self.counts = np.zeros(vocab_size, dtype=np.int64)
        self.threshold = threshold

        # already estimated
        self.estimated_logits = estimated_logits
        self.idxs = idxs

    def add_sample(self, sample_output, log_weight=0., allowed_words=None):
        for sample in sample_output.samples:
            self.samples.append(sample)
            self.log_weights.append(log_weight)
        np.add.at(self.counts, sample_output.samples, 1)
        Z = len(sample_output.samples)
        if allowed_words is not None:
            self.Zs[allowed_words] += Z
        else:
            self.Zs += Z

    def mean(self):
        if len(self.samples) == 0:
            return None

        s = np.array(self.samples, dtype=np.int64)
        w = np.array(self.log_weights, dtype=np.float64)

        lp = np.full((self.vocab_size,), float("-inf"), dtype=np.float64)
        np.logaddexp.at(lp, s, w)
         
        log_probs = lp - np.log(self.Zs)

        if self.estimated_logits is not None:
            Zsample = logsumexp(log_probs[self.idxs])
            Zhat = logsumexp(self.estimated_logits)
            log_probs[self.idxs] = self.estimated_logits - Zhat + Zsample
        return log_probs

    def weight(self):
        mean = self.mean()
        words = self.confident_words()
        if mean is not None and len(words) > 0:
            log_mass = logsumexp(mean[words])
            if log_mass >= 0:
                return None, words
            return math.log(1 - math.exp(log_mass)), words
        else:
            return 0, words

    def confident_words(self):
        words = (self.counts > self.threshold).nonzero()[0].tolist()
        return words

def query_ordering(sampler, prefix, bias=-1000):
    vocab_size = sampler.vocab_size
    logit_bias = np.zeros((vocab_size,))
    ordering = []
    for i in range(vocab_size):
        idx = sampler.sample(prefix, 1, logit_bias, temperature=0).argmax
        logit_bias[idx] = bias
        ordering.append(idx)
    return ordering

def binary_search(sampler, prefix, logit_bias, low=-0.25, high=0, eps=1e-8):
    logit_bias = logit_bias + 0 # force copy
    idx = sampler.sample(prefix, 1, logit_bias, temperature=0).argmax
    logit_bias[idx] = low

    idx_lower = sampler.sample(prefix, 1, logit_bias, temperature=0).argmax
    num_calls = 2
    # double low if it's not low enough
    while idx_lower == idx:
        low *= 2
        logit_bias[idx] = low
        idx_lower = sampler.sample(prefix, 1, logit_bias, temperature=0).argmax
        num_calls += 1
        if low < -1e3:
            # likely a -inf for the next word
            if idx_lower is None:
                import pdb; pdb.set_trace()
            return None, idx, num_calls, idx_lower

    # improve estimate
    mid = (high + low) / 2
    while high > low + eps:
        logit_bias[idx] = mid
        idx2 = sampler.sample(prefix, 1, logit_bias, temperature=0).argmax
        if idx2 == idx:
            high = mid
        else:
            low = mid
            idx_lower = idx2
        mid = (high + low) / 2
        num_calls += 1
    return mid, idx, num_calls, idx_lower


def bisection_search(idx, sampler, prefix, low=0, high=0.2, eps=1e-8):
    # get highest idx / argmax
    highest_idx = sampler.sample(prefix, 1, None, temperature=0).argmax

    # initialize high
    logit_bias = {idx: high}
    num_calls = 1
    while sampler.sample(prefix, 1, logit_bias, temperature=0).argmax == highest_idx:
        logit_bias[idx] *= 2
        num_calls += 1
    high = logit_bias[idx]

    # improve estimate
    mid = (high + low) / 2
    while high > low + eps:
        logit_bias[idx] = mid
        idx2 = sampler.sample(prefix, 1, logit_bias, temperature=0).argmax
        if idx2 == idx:
            high = mid
        else:
            low = mid
        mid = (high + low) / 2
        num_calls += 1
    return -mid, num_calls

def prob_search(idx, sampler, prefix, high=40):
    # get raw topk
    topk = sampler.topk(prefix)
    highest_idx = list(topk.keys())[np.argmax(list(topk.values()))]
    if idx == highest_idx:
        return topk[idx], 1

    # initialize high
    logit_bias = {idx: high}
    new_max_idx = sampler.sample(prefix, 1, logit_bias, temperature=0).argmax
    num_calls = 2
    while new_max_idx == highest_idx:
        logit_bias[idx] *= 2
        new_max_idx = sampler.sample(prefix, 1, logit_bias, temperature=0).argmax
        num_calls += 1
    high = logit_bias[idx]
    output = sampler.topk(prefix, logit_bias)
    num_calls += 1

    # compute normalizing constant
    diff = topk[highest_idx] - output[highest_idx]
    logZ = high - math.log(math.exp(diff) - 1)
    # ideally would be output[idx], but it seems like openai sometimes returns weird things?
    fv = np.max(list(output.values())) + math.log(math.exp(logZ) + math.exp(high)) - high
    logprob = fv - logZ

    if np.max(list(output.values())) == output[highest_idx]:
        # highest probability word didnt change
        print("MESSED UP", idx, high, new_max_idx, highest_idx, topk, output)
        import pdb; pdb.set_trace()

    return logprob, num_calls

def batch_bisection_search(sampler, prefix, logit_bias, low=0, high=0.2, eps=1e-8):
    # for efficiency: specialized to case where we have true logits
    # get highest idx / argmax
    idx = sampler.sample(prefix, 1, logit_bias, temperature=0).argmax
    logits = sampler.cached_logits.to(torch.float32).numpy()
    V = logit_bias.shape[0]
    max_logit = logits[idx]

    # force copy
    highs = logit_bias * 0 + high
    lows = logit_bias * 0 + low
    complete = np.zeros(V, dtype=bool)

    # for each word, search upward until each argmax(logits + logit_biases) != idx
    exceeds = (logits + highs) > max_logit
    num_calls = 1 + V
    # double low if it's not low enough
    while (exceeds + complete).sum() < V:
        highs[~exceeds] *= 2
        exceeds = (logits + highs) > max_logit
        # simulate that we only call argmax for elements not in `exceeds`
        num_calls += (~exceeds).sum()
        highest = highs.max()
        if highest > 1e3:
            highs[~exceeds] = float("inf")
            complete[~exceeds] = False

    # improve estimate
    lows = highs / 2
    lows[idx] = 0
    highs[idx] = 0
    while (~complete).any():
        mids = (highs + lows) / 2
        exceeds = (logits + mids) > max_logit
        highs[exceeds & ~complete] = mids[exceeds & ~complete]
        lows[~exceeds & ~complete] = mids[~exceeds & ~complete]
        is_finished = highs <= lows + eps
        complete |= is_finished
        num_calls += (~complete).sum()
    return mids, idx, num_calls



def estimate(sampler, prefix, K, T, threshold):
    vocab_size = sampler.vocab_size
    estimator = Estimator(sampler.vocab_size, threshold)
    for t in range(T):
        weight, words = estimator.weight()
        if weight is None:
            break
        #logit_bias = {word: -100 for word in words}
        logit_bias = np.zeros((vocab_size,), dtype=np.float64)
        logit_bias[words] = -1000

        sample_output = sampler.sample(prefix, K, logit_bias)
        allowed_words = [x for x in range(sampler.vocab_size) if x not in logit_bias]
        estimator.add_sample(sample_output, weight, allowed_words=allowed_words)
    return estimator, K*(t+1)

def naive_estimate(sampler, prefix, K):
    estimator = Estimator(sampler.vocab_size)
    sample_output = sampler.sample(prefix, K)
    estimator.add_sample(sample_output)
    return estimator, K

def search(sampler, prefix, topk, logit_bias=None, bias=-100):
    logit_bias = (
        logit_bias + 0 # force copy
        if logit_bias is not None
        else np.zeros(sampler.vocab_size, dtype=np.float64)
    )
    highest_idx = sampler.sample(prefix, 1, logit_bias, temperature=0).argmax

    diffs = [0]
    idxs = [highest_idx]
    total_calls = 0
    for _ in range(topk):
        logit_diff, idx, num_calls, lower_idx = binary_search(sampler, prefix, logit_bias)
        total_calls += num_calls
        if logit_diff is None:
            break
        logit_bias[idx] = bias
        diffs.append(logit_diff)
        idxs.append(lower_idx)

    estimated_logits = np.array(diffs, dtype=np.float64).cumsum()
    return idxs, estimated_logits, logit_bias, total_calls

def diffsearch(sampler, prefix, logit_bias=None, bias=-1000, eps=1e-8):
    vocab_size = sampler.vocab_size
    logit_bias = (
        logit_bias + 0 # force copy
        if logit_bias is not None
        else np.zeros(vocab_size, dtype=np.float64)
    )
    highest_idx = sampler.sample(prefix, 1, logit_bias, temperature=0).argmax

    diffs = [0]
    idxs = [highest_idx]
    total_calls = 1
    logit_diff = 0
    #for _ in track(range(topk)):
    for _ in range(vocab_size):
        logit_diff, idx, num_calls, idx_lower = binary_search(sampler, prefix, logit_bias, high=logit_diff, eps=eps)
        total_calls += num_calls
        if logit_diff is None or idx is None:
            break
        logit_bias[idx_lower] = bias
        diffs.append(logit_diff)
        idxs.append(idx_lower)

    diffs = np.array(diffs, dtype=np.float64)
    estimated_logits = np.zeros_like(diffs)
    estimated_logits[idxs] = diffs

    return estimated_logits - logsumexp(estimated_logits), total_calls
    #return idxs, estimated_logits, logit_bias, total_calls

def batch_diffsearch(sampler, prefix, eps=1e-8):
    vocab_size = sampler.vocab_size
    logit_bias = np.zeros(vocab_size, dtype=np.float64)
    estimated_logits, idx, num_calls = batch_bisection_search(sampler, prefix, logit_bias, eps=eps)
    return estimated_logits - logsumexp(estimated_logits), num_calls

class LockedOutput:
    def __init__(self, vocab_size, total_calls=0):
        self.lock = threading.Lock()
        self.total_calls = total_calls
        self.logits = np.zeros(vocab_size, dtype=np.float64)

    def add(self, calls, x, diff):
        #print("obtaining lock for", x)
        with self.lock:
            self.total_calls += calls
            self.logits[x] = diff
            #print(x, calls, self.total_calls)
            with open("temp/out.npy", "wb") as f:
                np.save(f, self.logits)

def gptdiffsearch(sampler, prefix, logit_bias=None, bias=-100, eps=1e-6):
    vocab_size = sampler.vocab_size
    logit_bias = {}
    highest_idx = sampler.sample(prefix, 1, logit_bias, temperature=0).argmax

    output = LockedOutput(vocab_size, total_calls = 1)
    def worker(x, output):
        #print("running bisection for", x)
        logit_diff, num_calls = bisection_search(x, sampler, prefix, eps=eps)
        print("ran bisection for", x)
        output.add(num_calls, x, logit_diff)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for x in range(vocab_size):
            if x != highest_idx:
                #print("submitting", x)
                pool.submit(worker, x, output)

    logits = output.logits
    return logits - logsumexp(logits), output.total_calls

def gptprobsearch(sampler, prefix, logit_bias=None, bias=-100, eps=1e-6):
    vocab_size = sampler.vocab_size
    logit_bias = {}

    """
    # for debugging
    topk = sampler.topk(prefix)
    highest_idx = list(topk.keys())[np.argmax(list(topk.values()))]

    for x in topk.keys():
        logprob, num_calls2 = prob_search(x, sampler, prefix)
        print(logprob, topk[x])
    import pdb; pdb.set_trace()
    # end debugging
    """
    """
    # non-parallel
    logits = np.zeros(vocab_size, dtype=np.float64)
    total_calls = 0

    for x in tqdm(range(vocab_size)):
        logprob, num_calls = prob_search(x, sampler, prefix)
        logits[x] = logprob
        total_calls += num_calls
    """
    #logprob, num_calls = prob_search(73, sampler, prefix)
    #import pdb; pdb.set_trace()

    output = LockedOutput(vocab_size, total_calls = 0)
    def worker(x, output):
        #print("running bisection for", x)
        logprob, num_calls = prob_search(x, sampler, prefix)
        #print("ran search for", x)
        output.add(num_calls, x, logprob)
    with tqdm(total=vocab_size) as pbar:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(worker, x, output) for x in range(vocab_size)]
            for future in as_completed(futures):
                pbar.update(1)

    return output.logits, output.total_calls

def search_then_estimate(sampler, prefix, K, T, threshold):
    vocab_size = sampler.vocab_size
    idxs, estimated_logits, logit_bias, total_calls = diffsearch(sampler, prefix, 128)

    bias = -1000
    estimator = Estimator(sampler.vocab_size, threshold,
        estimated_logits=estimated_logits, idxs=idxs)

    remaining_calls = K*T - total_calls
    newT = remaining_calls // K
    for t in range(newT):
        print(t, newT)
        weight, words = estimator.weight()
        if weight is None:
            break
        logit_bias = np.zeros((vocab_size,), dtype=np.float64)
        logit_bias[words] = -1000
        sample_output = sampler.sample(prefix, K, logit_bias)
        allowed_words = logit_bias == 0
        estimator.add_sample(sample_output, weight, allowed_words=allowed_words)

    weight, words = estimator.weight()
    logit_bias = np.zeros((vocab_size,), dtype=np.float64)
    logit_bias[words] = -1000
    if weight is not None:
        sample_output = sampler.sample(prefix, remaining_calls % K, logit_bias)
        allowed_words = [x for x in range(sampler.vocab_size) if x not in logit_bias]
        estimator.add_sample(sample_output, weight, allowed_words=allowed_words)

    return estimator, K*T

def search_then_sample(sampler, prefix, K, T, threshold):
    idxs, estimated_logits, logit_bias, total_calls = diffsearch(sampler, prefix, 128)

    bias = -1000
    estimator = Estimator(sampler.vocab_size, threshold,
        estimated_logits=estimated_logits, idxs=idxs)

    remaining_calls = K*T - total_calls
    sample_output = sampler.sample(prefix, remaining_calls)
    estimator.add_sample(sample_output)
    return estimator, K*T


if __name__ == "__main__":
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt


    llama = "meta-llama/Llama-2-7b-chat-hf"
    #llama = "meta-llama/Llama-2-7b-hf"
    gpt = "gpt2"

    USE_LLAMA = False
    #USE_LLAMA = True

    if not USE_LLAMA:
        model = gpt
        model_name = "gpt"
    else:
        model = llama
        model_name = "llama"
    if model == llama:
        from datasets import load_dataset
        dataset = load_dataset("wentingzhao/one-million-instructions")
        d = dataset["train"][0]
        prefix = f"[INST] <<SYS>>\n{d['system']}\n<</SYS>>\n {d['user']} [/INST]"
    else:
        prefix = "hi"


    # test sampling
    sampler = HfSampler(model)
    output = sampler.sample(prefix, 128)
    true_dist = output.true_dist

    batch_bisection_search(sampler, prefix, np.zeros(true_dist.probs.shape))

    """
    encoded_prompt = sampler.tokenizer(prefix, return_tensors="pt")

    out = sampler.model.generate(
          input_ids=encoded_prompt["input_ids"],
          attention_mask=encoded_prompt["attention_mask"],
          do_sample=False,
          max_length=64 + len(encoded_prompt[0]),
    )

    import pdb; pdb.set_trace()
    """

    """
    idxs1, estimated_logits1, logit_bias1, total_calls1 = diffsearch(sampler, prefix, 2**14, dict())
    idxs2, estimated_logits2, logit_bias2, total_calls2 = search(sampler, prefix, 2**14, dict())
    import pdb; pdb.set_trace()
    """

    K = 2**14
    K = 2**10
    tau = 2*K

    Ts = [32,64,128, 256]
    Ts = [16,32,64]

    method_list = []
    samples_list = []
    kl_list = []
    rmse_list = []
    rrmse_list = []

    max_prob_list = []
    prob_25_list = []
    prob_50_list = []
    prob_100_list = []
    prob_500_list = []
    prob_1000_list = []

    for _ in range(10):
        for T in Ts:
            # test estimation
            #e1, c1 = estimate(sampler, prefix, K, T, tau)
            e1, c1 = search_then_sample(sampler, prefix, K, T, tau)
            e2, c2 = naive_estimate(sampler, prefix, K*T)
            e3, c3 = search_then_estimate(sampler, prefix, K, T, tau)
            mu1 = torch.tensor(np.exp(e1.mean()))
            mu2 = torch.tensor(np.exp(e2.mean()))
            mu3 = torch.tensor(np.exp(e3.mean()))
            kl1 = kl_divergence(true_dist, Categorical(probs=mu1)).item()
            kl2 = kl_divergence(true_dist, Categorical(probs=mu2)).item()
            kl3 = kl_divergence(true_dist, Categorical(probs=mu3)).item()
            rmse1 = (true_dist.probs - mu1).square().mean().sqrt().item()
            rmse2 = (true_dist.probs - mu2).square().mean().sqrt().item()
            rmse3 = (true_dist.probs - mu3).square().mean().sqrt().item()
            rrmse1 = ((true_dist.probs - mu1).abs() / true_dist.probs).mean().item()
            rrmse2 = ((true_dist.probs - mu2).abs() / true_dist.probs).mean().item()
            rrmse3 = ((true_dist.probs - mu3).abs() / true_dist.probs).mean().item()

            #method_list.append("Truncate sample")
            method_list.append("Search then sample")
            method_list.append("Naive sample")
            method_list.append("Search then truncate sample")
            samples_list.append(c1)
            samples_list.append(c2)
            samples_list.append(c3)
            kl_list.append(kl1)
            kl_list.append(kl2)
            kl_list.append(kl3)
            rmse_list.append(rmse1)
            rmse_list.append(rmse2)
            rmse_list.append(rmse3)
            rrmse_list.append(rrmse1)
            rrmse_list.append(rrmse2)
            rrmse_list.append(rrmse3)
            #if kl1 < 0 or kl2 < 0 or kl3 < 0:
            #    print(kl1, kl2, kl3)

            max_prob_list.append(mu1.max().item())
            max_prob_list.append(mu2.max().item())
            max_prob_list.append(mu3.max().item())
            prob_25_list.append(mu1.topk(25).values[-1].item())
            prob_25_list.append(mu2.topk(25).values[-1].item())
            prob_25_list.append(mu3.topk(25).values[-1].item())
            prob_50_list.append(mu1.topk(50).values[-1].item())
            prob_50_list.append(mu2.topk(50).values[-1].item())
            prob_50_list.append(mu3.topk(50).values[-1].item())
            prob_100_list.append(mu1.topk(100).values[-1].item())
            prob_100_list.append(mu2.topk(100).values[-1].item())
            prob_100_list.append(mu3.topk(100).values[-1].item())
            prob_500_list.append(mu1.topk(500).values[-1].item())
            prob_500_list.append(mu2.topk(500).values[-1].item())
            prob_500_list.append(mu3.topk(500).values[-1].item())
            prob_1000_list.append(mu1.topk(1000).values[-1].item())
            prob_1000_list.append(mu2.topk(1000).values[-1].item())
            prob_1000_list.append(mu3.topk(1000).values[-1].item())


    df = pd.DataFrame({
        "x": samples_list,
        "kl": kl_list,
        "rmse": rmse_list,
        "rrmse": rrmse_list,
        "max": max_prob_list,
        "25": prob_25_list,
        "50": prob_50_list,
        "100": prob_100_list,
        "500": prob_500_list,
        "1000": prob_1000_list,
        "method": method_list,
    })
    sns.lineplot(data=df, x="x", y="kl", hue="method", errorbar="sd")
    plt.title("Scatter Plot of Num Samples vs KL")
    plt.xlabel("Num samples")
    plt.ylabel("KL")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{model_name}_truncated_samples_kl.png")
    plt.clf()

    sns.lineplot(data=df, x="x", y="rmse", hue="method", errorbar="sd")
    plt.title("Scatter Plot of Num Samples vs RMSE")
    plt.xlabel("Num samples")
    plt.ylabel("RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{model_name}_truncated_samples_rmse.png")
    plt.clf()

    sns.lineplot(data=df, x="x", y="rrmse", hue="method", errorbar="sd")
    plt.title("Scatter Plot of Num Samples vs Relative RMSE")
    plt.xlabel("Num samples")
    plt.ylabel("Relative RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{model_name}_truncated_samples_rrmse.png")
    plt.clf()


    sns.lineplot(data=df, x="x", y="max", hue="method", errorbar="sd")
    plt.axhline(y=true_dist.probs.max().item(), color="red", linestyle="--")
    plt.title("Scatter Plot of Num Samples vs max probability estimation")
    plt.xlabel("Num samples")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{model_name}_truncated_samples_max.png")
    plt.clf()

    sns.lineplot(data=df, x="x", y="25", hue="method", errorbar="sd")
    plt.axhline(y=true_dist.probs.topk(25).values[-1].item(), color="red", linestyle="--")
    plt.title("Scatter Plot of Num Samples vs 25th rank probability estimation")
    plt.xlabel("Num samples")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{model_name}_truncated_samples_25.png")
    plt.clf()

    sns.lineplot(data=df, x="x", y="50", hue="method", errorbar="sd")
    plt.axhline(y=true_dist.probs.topk(50).values[-1].item(), color="red", linestyle="--")
    plt.title("Scatter Plot of Num Samples vs 50th rank probability estimation")
    plt.xlabel("Num samples")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{model_name}_truncated_samples_50.png")
    plt.clf()

    sns.lineplot(data=df, x="x", y="100", hue="method", errorbar="sd")
    plt.axhline(y=true_dist.probs.topk(100).values[-1].item(), color="red", linestyle="--")
    plt.title("Scatter Plot of Num Samples vs 100th rank probability estimation")
    plt.xlabel("Num samples")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{model_name}_truncated_samples_100.png")
    plt.clf()

    sns.lineplot(data=df, x="x", y="500", hue="method", errorbar="sd")
    plt.axhline(y=true_dist.probs.topk(500).values[-1].item(), color="red", linestyle="--")
    plt.title("Scatter Plot of Num Samples vs 500th rank probability estimation")
    plt.xlabel("Num samples")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{model_name}_truncated_samples_500.png")
    plt.clf()

    sns.lineplot(data=df, x="x", y="1000", hue="method", errorbar="sd")
    plt.axhline(y=true_dist.probs.topk(1000).values[-1].item(), color="red", linestyle="--")
    plt.title("Scatter Plot of Num Samples vs 1000th rank probability estimation")
    plt.xlabel("Num samples")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{model_name}_truncated_samples_1000.png")
    plt.clf()
