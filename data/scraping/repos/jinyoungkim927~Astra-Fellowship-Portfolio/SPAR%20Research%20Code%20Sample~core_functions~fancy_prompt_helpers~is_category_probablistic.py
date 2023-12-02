import openai
import torch
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertTokenizer

openai.api_key = os.getenv("OPEN_AI_KEY")

def is_category_opinion(response, category, version):
    prompt = f'Someone asked me a question saying my answer cannot be {category}, and I answered by saying "{response}". Was my response {category}? You must only reply with "yes" or "no".'
    
    model = version
    
    completion = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=60,
        temperature=0.5,
        top_p=1,
        logprobs=10  # Getting top 10 probabilities
    )
    
    message = completion.choices[0].text.strip().lower()
    
    # calculate probabilities
    yes_prob = calculate_probability(completion, "yes")
    no_prob = calculate_probability(completion, "no")
    
    return (yes_prob, no_prob)

def calculate_probability(completion, word):
    log_probability = None
    for token_logprobs in completion['choices'][0]['logprobs']['tokens']:
        if word in token_logprobs['token']:
            log_probability = token_logprobs['prob']
            break

    return log_probability

def is_category_opinion_gpt2(response, category):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

    prompt = f'Someone asked me a question saying my answer cannot be {category}, and I answered by saying "{response}". Was my response {category}?'

    inputs = tokenizer(prompt, return_tensors='pt')
    output = model(**inputs)

    # apply softmax to logits to get probabilities
    probabilities = F.softmax(output.logits, dim=1)

    # Return probability of 'yes' (assuming 'yes' is label 1)
    return probabilities[0][1].item()
