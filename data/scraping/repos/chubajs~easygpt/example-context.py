"""
Effective Long Conversation Example using EasyGPT

This example demonstrates how EasyGPT can handle long conversation context effectively.

Author: Sergey Bulaev
License: MIT
"""

import os
from easygpt import EasyGPT
import openai

# Initialize OpenAI API key from an environment variable
openai.api_key = os.environ.get('OPENAI_API_KEY')

# Initialize EasyGPT instance
easy_gpt = EasyGPT(openai, model_name="gpt-3.5-turbo")

# Prepare the initial context
context = [
    {"role": "system", "content": "You are chatting with a human."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help you?"},
    {"role": "user", "content": "Tell me about quantum mechanics."},
    {"role": "assistant", "content": "Sure! It's the physics of really small stuff like atoms."},
    {"role": "user", "content": "That's too vague. Can you be more specific?"},
    {"role": "assistant", "content": "My apologies for the vague answer."},
    {"role": "user", "content": "Let's try this again. Explain quantum mechanics."},
]

# Query with the initial context
print(f"User says: {context[-1]['content']}")
response, input_price, output_price = easy_gpt.ask_with_context(context)
print(f"Assistant says: {response}")
print(f"Input Price: {input_price}, Output Price: {output_price}")

# Add the assistant's response to the context and continue
context.append({"role": "assistant", "content": response})
context.append({"role": "user", "content": "That was really detailed. What are its applications?"})

# Query with the updated context
print(f"User says: {context[-1]['content']}")
response, input_price, output_price = easy_gpt.ask_with_context(context)
print(f"Assistant says: {response}")
print(f"Input Price: {input_price}, Output Price: {output_price}")

# Add the assistant's response to the context and continue
context.append({"role": "assistant", "content": response})
context.append({"role": "user", "content": "You've regained my trust. That's quite comprehensive."})

# Query with the final context
print(f"User says: {context[-1]['content']}")
response, input_price, output_price = easy_gpt.ask_with_context(context)
print(f"Assistant says: {response}")
print(f"Input Price: {input_price}, Output Price: {output_price}")
