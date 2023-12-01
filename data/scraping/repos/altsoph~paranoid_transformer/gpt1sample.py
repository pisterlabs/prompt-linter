#!/usr/bin/env python3

import os
import random
import json

import numpy as np
import nltk
import torch
import torch.nn.functional as F

# from apex import amp
from tqdm import tqdm, trange
from pytorch_pretrained_bert import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, OpenAIAdam


SAMPLES = 16384
BATCH_SIZE = 32

MAX_LEN = 500
MODEL_DIR = "/home/altsoph/current"
SEED = 0xDEADFEED


def top_k_logits(logits, k):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)

def sample_sequence(model, length, segments=None, batch_size=None, context=None, temperature=1, top_k=0, device='cuda', sample=True, text_tag=0):
    context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    segments = torch.tensor(segments, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    text_tag_tpl = torch.tensor([text_tag,], device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)

    prev = context
    output = context
    prev_segments = segments
    past = None
    with torch.no_grad():
        for i in trange(length):
            # model(input_ids.to(device), lm_labels=lm_labels.to(device), token_type_ids=token_type_ids.to(device))
            logits = model(output, token_type_ids=prev_segments)
            logits = logits[:, -1, :] / temperature
            logits = top_k_logits(logits, k=top_k)
            log_probs = F.softmax(logits, dim=-1)
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1)
            output = torch.cat((output, prev), dim=1)
            prev_segments = torch.cat((prev_segments, text_tag_tpl), dim=1)
    return output

random.seed(SEED)
torch.random.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
model = OpenAIGPTLMHeadModel.from_pretrained(MODEL_DIR)
tokenizer = OpenAIGPTTokenizer.from_pretrained(MODEL_DIR)

model.to(device)

TAG_QUOTES, TAG_CYBER, TAG_TEXT, TAG_META1, TAG_META2, TAG_PAD = tokenizer.convert_tokens_to_ids(
                            ("<quotes>", "<cyberpunk>", "<text>", "<meta1>", "<meta2>", "<pad>"))

context_tokens   = [TAG_QUOTES, TAG_CYBER]
context_segments = [TAG_META1, TAG_META2]

generated = 0

for _ in range(SAMPLES // BATCH_SIZE):
    out = sample_sequence(
        model=model, length=MAX_LEN,
        context=context_tokens,
        segments=context_segments,
        batch_size=BATCH_SIZE,
        temperature=1, top_k=0, device=device,
        text_tag = TAG_TEXT
    )
    out = out[:, len(context_tokens):].tolist()
    for i in range(BATCH_SIZE):
        generated += 1
        text = tokenizer.decode(out[i])
        print("=" * 35 + " SAMPLE " + str(generated) + " " + "=" * (36-len(str(generated))) )
        print(text)

