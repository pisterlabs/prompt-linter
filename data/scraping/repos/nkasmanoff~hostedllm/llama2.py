import torch
import transformers
from langchain.llms import HuggingFacePipeline



class HuggingFaceLLM():
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf", verbose=False):
        self.model_name = model_name
        self.verbose = verbose
        self.pipeline = self.load_pipeline(
            model_id=self.model_name, verbose=self.verbose
        )
        self.chat_template = """{query_str}"""

    def __call__(self, prompt):
        prompt = self.chat_template.format(query_str=prompt)
        response = self.pipeline(prompt) 
        return response

    def load_pipeline(self, model_id="meta-llama/Llama-2-7b-chat-hf", verbose=False):
        tokenizer = transformers.LlamaTokenizer.from_pretrained(model_id)
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id, 
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_new_tokens=1024,
            temperature=0,
            top_p=0.95,
            repetition_penalty=1.15,
            num_return_sequences=1,
        )
        pipeline = HuggingFacePipeline(pipeline=pipeline, verbose=True)
        return pipeline


if __name__ == '__main__':
    llm_model = HuggingFaceLLM()

    response = llm_model("Any weather events in the next 24 hours you think we should be aware of? For example, do any cities look like they might be in risk of flooding?")
    print(response)