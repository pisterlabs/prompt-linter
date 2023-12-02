import os
import logging

FILE_PATH = os.path.dirname(os.path.realpath(__file__))

class LLMInterface:
    model_name: str
    def __init__(self) -> None:
        pass

    def query(self, system: str, user_message: str):
        raise NotImplementedError()


def get_llm_interface(llm_type):
    if llm_type == "openai":
        import openai
        class OpenAIInterface(LLMInterface):
            def __init__(self) -> None:
                super().__init__()
                self.model_name = os.getenv("LLM_NAME",default="gpt-3.5-turbo")
                openai.api_key = os.getenv("OPENAI_API_KEY")
                if openai.api_key is None:
                    logging.warning("openai.api_key is None")

            def query(self, system, user_message):
                completion = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_message}
                    ]
                )
                return completion.choices[0].message.content
        return OpenAIInterface()
    elif llm_type == "huggingface":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
        class HuggingFaceInterface(LLMInterface):
            def __init__(self) -> None:
                model_name = os.getenv("LLM_NAME")
                cache_dir = os.getenv("CACHE_DIR")

                # Check if cache_Dir is a real directory
                if not os.path.isdir(cache_dir):
                    raise Exception(f"cache_dir {cache_dir} specified in is not a directory")

                self.model_name = model_name
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    load_in_8bit=True,
                    rope_scaling={"type": "dynamic", "factor": 2}, # allows handling of longer inputs
                    cache_dir=cache_dir,
                )
                self.streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

            def query(self, instruction, user_message):
                inputs = self.format_input(instruction, user_message)
                output = self.model.generate(**inputs, streamer=self.streamer, use_cache=True, max_new_tokens=float('inf'))
                output_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                # The output text includes the input text, so we need to remove it.
                response = output_text.split("Response")[1]
                return response

            def format_input(self, instruction, user_input):
                # Note: one issue seems to be the llama prompt format is juat a bit different...
                input_string = f"### Instruction:\n{instruction}\n### Input:\n{user_input}\n### Response:\n"
                inputs = self.tokenizer(input_string, return_tensors="pt").to(self.model.device)
                del inputs["token_type_ids"]
                return inputs
        return HuggingFaceInterface()
    else:
        raise NotImplementedError(f"llm_type {llm_type} not implemented")
