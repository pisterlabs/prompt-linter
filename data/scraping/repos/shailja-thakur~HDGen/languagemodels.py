from abc import ABC, abstractmethod

import openai
from anthropic import Anthropic
from anthropic import AsyncAnthropic, HUMAN_PROMPT, AI_PROMPT

import os
from conversation import Conversation

# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch



# Abstract Large Language Model
# Defines an interface for using different LLMs so we can easily swap them out
class AbstractLLM(ABC):
    """Abstract Large Language Model."""

    def __init__(self):
        pass

    @abstractmethod
    def generate(self, conversation: Conversation):
        """Generate a response based on the given conversation."""
        pass


class ChatGPT3p5(AbstractLLM):
    """ChatGPT Large Language Model."""

    def __init__(self):
        super().__init__()
        openai.api_key=os.environ['OPENAI_API_KEY']

    def generate(self, conversation: Conversation):
        messages = [{'role' : msg['role'], 'content' : msg['content']} for msg in conversation.get_messages()]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages = messages,
        )

        return response['choices'][0]['message']['content']

class ChatGPT4(AbstractLLM):
    """ChatGPT Large Language Model."""

    def __init__(self):
        super().__init__()
        openai.api_key=os.environ['OPENAI_API_KEY']

    def generate(self, conversation: Conversation):
        messages = [{'role' : msg['role'], 'content' : msg['content']} for msg in conversation.get_messages()]

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages = messages,
        )

        return response['choices'][0]['message']['content']

class Claude(AbstractLLM):
    """Claude Large Language Model."""

    def __init__(self):
        super().__init__()
        self.anthropic = Anthropic(
            api_key=os.environ['ANTHROPIC_API_KEY'],
        )

    def generate(self, conversation: Conversation):
        prompt = ""
        for message in conversation.get_messages():
            if message['role'] == 'system' or message['role'] == 'user':
                prompt += f"\n\nHuman: {message['content']}"
            elif message['role'] == 'assistant':
                prompt += f"\n\nAssistant: {message['content']}"
        prompt += "\n\nAssistant:"


        completion = self.anthropic.completions.create(
            model="claude-2",
            max_tokens_to_sample=3000,
            prompt=prompt,
        )

        #print(prompt)
        #print(completion.completion)
        return completion.completion



# class CodeLlama(AbstractLLM):
#     """CodeLlama Large Language Model."""

#     def __init__(self, model_id="codellama/CodeLlama-13b-hf"):
#         super().__init__()
        
#         self.model_id = model_id

#         self.tokenizer = CodeLlamaTokenizer.from_pretrained(self.model_id)
#         self.model = LlamaForCausalLM.from_pretrained(self.model_id, device_map="auto",torch_dtype = "auto")

#     def _format_prompt(self, conversation: Conversation) -> str:
#         # Extract the system prompt, initial user prompt, and the most recent user prompt and answer.
#         messages = conversation.get_messages()

#         prompt = ""

#         for message in messages:
#             # Append system messages with the "<<SYS>>" tags
#             if message['role'] == 'system':
#                 prompt = f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
#             # Append user messages with the "Human" prefix
#             elif message['role'] == 'user':
#                 prompt += f"{most_recent_user_prompt.strip()}"
#             # Append assistant messages with the "Assistant" prefix wrapped with [INST] tags
#             elif message['role'] == 'assistant':
#                 prompt += f"<s>[INST] {message['content']} [/INST]"



#         return prompt

#     def generate(self, conversation: Conversation):

#         # Prepare the prompt using the method we created
#         prompt = self._format_prompt(conversation)

#         inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
#         output = self.model.generate(
#             inputs["input_ids"],
#             max_new_tokens=200,
#             do_sample=True,
#             top_p=0.9,
#             temperature=0.1,
#         )

#         # Move the output tensor to the CPU
#         output = output[0].to("cpu")

#         # Decode the output to get the generated text
#         generated_output = self.tokenizer.decode(output, skip_special_tokens=True)
        
#         # Extract only the generated response
#         response = generated_output.split("</s>")[-1].strip()
        
#         return response
