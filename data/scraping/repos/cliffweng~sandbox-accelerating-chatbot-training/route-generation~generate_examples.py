# Copyright (c) 2022 Cohere Inc. and its affiliates.
#
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
#
# You may obtain a copy of the License in the LICENSE file at the top
# level of this repository.

import cohere
import re
import streamlit as st

co = cohere.Client(st.secrets["cohere_api_token"])

def co_request(prompt, gens, description):
    """
    This is a call to Cohere's generation endpoint to create the example phrases for a given description

    param prompt: a string that serves as the pattern for the generation model to learn from
    param gens: integer for how many times the model should generate examples
    param description: a string describing the route to create

    returns: cohere generation response for each description
    """
    response = co.generate(
      model='xlarge',
      prompt= prompt + ' ' + description + '. Some example phrases are:',
      max_tokens=80,
      temperature=.9,
      k=0,
      p=0.7,
      frequency_penalty=0.04,
      presence_penalty=0,
      stop_sequences=["--"],
      return_likelihoods='NONE',
      num_generations=gens)
    return response

def generate_examples(prompt, num_examples, description, timeout=10, route=None):
    """
    This is a wrapper function around co_request that will handle generating enough examples to satisfy the user-provided example count.
    It will also remove duplicates and store responses even in the event of an error

    param prompt: a string that serves as the pattern for the generation model to learn from
    param num_examples: integer for how many total examples should be created for each description
    param description: a string describing the route to create
    param timeout: integer for the limit of how many calls to make per description

    returns: cohere generation response for each description
    """
    output = []
    calls = 0
    # this will loop until the number of examples is met or the timeout is hit
    while len(output) < num_examples:
        if calls >= timeout:
            return {'message': "timeout error: too many calls made. Returned results up until failure", 'results': output}
            break
        response = co_request(prompt, 1, description)
        calls += 1
        for gen in response.generations:
            #re.sub('\d[.]', '', g) - if using numbers
            examples = [g for g in gen.text.split('\n') if len(g) > 5]
            for e in examples:
                if len(output) == num_examples:
                    break
                if not e.strip() in output and not e == '--' and not e == '' and len(e.split()) > 2:
                    output.append(e.strip())
    print('generated', len(output), 'for', route)
    return {'message': "Results successfully created", 'results': output}