# Originally from: https://gist.github.com/kennethleungty/41d3eb19141c161c5f920bb1659521fe#file-llm-py
# Original author: Kenneth Leung
# Snapshot date: 2023-08-01

from langchain.llms import CTransformers

# https://github.com/marella/ctransformers#config
llm = CTransformers(model='models/llama-2-13b-chat.ggmlv3.q4_0.bin',
                model_type='llama', # Model type Llama
                config={'max_new_tokens': 256, 'temperature': 0.01})