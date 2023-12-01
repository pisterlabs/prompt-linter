# Copyright (c) 2023 Cohere Inc. and its affiliates.
#
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
#
# You may obtain a copy of the License in the LICENSE file at the top
# level of this repository.

import os
import re

import cohere
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

api_key = os.environ["CO_KEY"]
co = cohere.Client(api_key)


# Get text embeddings via the Embed endpoint
def embed_text(texts):
    embeddings = co.embed(model="large", texts=texts)
    embeddings = np.array(embeddings.embeddings)
    return embeddings


# Calculate cosine similarity between the target and candidate articles
def get_similarity(target, candidates):
    # Turn list into array
    candidates = np.array(candidates)
    target = np.expand_dims(np.array(target), axis=0)

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(target, candidates)
    similarity_scores = np.squeeze(similarity_scores).tolist()

    # Sort by descending order in similarity
    similarity_scores = list(enumerate(similarity_scores))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Return similarity scores
    return similarity_scores


# Truncate text to a max of 512 tokens (Classify finetuned model's limit)
def truncate_text(input_text):
    tokenized = co.tokenize(text=input_text).tokens
    truncated = co.detokenize(tokens=tokenized[:500]).text
    return truncated


# Perform classification with a finetuned model
def classify_text(input_text):
    text = truncate_text(input_text)
    classifications = co.classify(
        model="504b1b30-4927-464d-9d4c-412f9771775b-ft", inputs=[text]  # replace with your finetune model ID
    )
    return classifications.classifications[0].prediction


# Create the prompt for tags extraction
def create_prompt(text):
    return f"""Given a news article, this program returns the list tags containing keywords of that article.
Article: japanese banking battle at an end japan s sumitomo mitsui financial has withdrawn its takeover offer for \
rival bank ufj holdings  enabling the latter to merge with mitsubishi tokyo.  sumitomo bosses told counterparts at \
ufj of its decision on friday  clearing the way for it to conclude a 3 trillion
Tags: sumitomo mitsui financial, ufj holdings, mitsubishi tokyo, japanese banking
--
Article: france starts digital terrestrial france has become the last big european country to launch a digital \
terrestrial tv (dtt) service.  initially  more than a third of the population will be able to receive 14 free-to-air \
channels. despite the long wait for a french dtt roll-out
Tags: france, digital terrestrial
--
Article: apple laptop is  greatest gadget  the apple powerbook 100 has been chosen as the greatest gadget of all time  \
by us magazine mobile pc.  the 1991 laptop was chosen because it was one of the first  lightweight  portable computers \
and helped define the layout of all future notebook pcs.
Tags: apple, apple powerbook 100, laptop
--
Article: {text}
Tags:"""


# Get extractions via the Generate endpoint
def extract_tags(complete_prompt):
    # Truncate the complete prompt if too long
    token_check = co.tokenize(text=complete_prompt)
    if len(token_check.tokens) > 2000:
        complete_prompt = co.detokenize(token_check.tokens[:2000]).text

    prediction = co.generate(
        model="xlarge",
        prompt=complete_prompt,
        max_tokens=10,
        temperature=0.3,
        k=0,
        p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop_sequences=["--"],
        return_likelihoods="NONE",
    )
    tags_raw = prediction.generations[0].text

    if "\n" in tags_raw:
        tags_clean = re.search(".+?(?=\\n)", tags_raw).group()
    else:
        tags_clean = tags_raw

    if tags_clean:
        tags = tags_clean.split(",")
        tags = list(dict.fromkeys(tags))  # remove duplicates
        tags = [tag.strip() for tag in tags]  # remove empty spaces
        tags = [tag for tag in tags if tag]  # remove none ones
        tags = [tag for tag in tags if len(tag) > 3]  # remove short ones
        tags = [f"`{tag}`" for tag in tags]  # format tag string
        tags = ",".join(tags)
    else:
        tags = "None"

    return tags
