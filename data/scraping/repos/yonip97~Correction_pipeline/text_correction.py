import openai
import tiktoken
import os


class LLM_model():
    def __init__(self, prompt_path=None, prompt_text=None,past_text_prompt = '', model='chatgpt-turbo-3.5', API_KEY=None, **kwargs):
        openai.api_key = API_KEY
        openai.api_type = "azure"
        openai.api_version = "2023-05-15"
        openai.api_base = "https://researchopenai2023.openai.azure.com/"
        if prompt_path is not None or prompt_text is not None:
            if prompt_path is not None and prompt_text is None:
                with open(prompt_path, "r") as file:
                    prompt = file.read()
            elif prompt_text is not None and prompt_path is None:
                prompt = prompt_text
            else:
                raise ValueError("prompt_path and prompt can't be both not None")
        else:
            raise ValueError("prompt_path and prompt can't be both None")
        self.past_text_prompt = past_text_prompt
        self.prompt = prompt
        self.model = model
        self.estimation_tokenizer = tiktoken.encoding_for_model(model)

    def call_llm(self, text, max_length, **kwargs):
        input = self.prompt + '\n' + text +'\n' + self.past_text_prompt
        try:
            message = [{
                "role": "user",
                "content": input,
            }]
            response = openai.ChatCompletion.create(
                engine=self.model,
                messages=message,
                temperature=0,
                max_tokens=max_length
            )
            return response['choices'][0]['message']['content']
        except openai.OpenAIError as e:
            print(f"Error occurred: {e}")
            return None


class Summerization_correction_model(LLM_model):
    def __call__(self, document, summary, max_length=None):
        text_for_revision = f"Document: \n {document} \n summary: \n {summary} \n"
        if max_length is None:
            max_length = len(self.estimation_tokenizer.encode(summary)) + 10
        revised_summary = self.call_llm(text_for_revision, max_length=max_length)
        return revised_summary

    def process_dataset(self, dataset):
        revised_summaries = []
        for i in range(len(dataset)):
            text = dataset[i]['text']
            summary = dataset[i]['summary']
            revised_summary = self(text, summary)
            revised_summaries.append(revised_summary)
        return dataset
