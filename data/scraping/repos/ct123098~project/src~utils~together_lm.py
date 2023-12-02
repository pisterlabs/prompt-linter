from utils.lm import LM
import openai
import time
import os
import logging
import together

class TogetherModel(LM):

    def __init__(self, model_name, cache_file=None, save_interval=100):
        self.model_name = model_name
        assert "llama" in model_name.lower(), "only llama models are supported"
        super().__init__(cache_file, save_interval)

    def load_model(self):
        together.api_key = os.environ.get("TOGETHER_API_KEY")
        together.Models.start(self.model_name)
        self.model = self.model_name

    def _generate(self, prompt_or_messages, temperature=0.0, max_tokens=100):
        # return a tuple of string (generated text) and metadata (any format)
        # This should be about generating a response from the prompt, no matter what the application is
            # Construct the prompt send to ChatGPT
        
        if isinstance(prompt_or_messages, str):
            prompt_or_messages = [{"role": "user", "content": prompt_or_messages}]

        assert isinstance(prompt_or_messages, list), "prompt_or_messages must be a list"
        assert len(prompt_or_messages) == 1, "only one message is supported"
        assert prompt_or_messages[0]['role'] == 'user', "only user messages are supported"

        # new_system_prompt = ""
        # B_TEXT, E_TEXT = "<s>", "</s>"
        # B_INST, E_INST = "[INST]", "[/INST]"
        # B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        # SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
        # prompt =  B_TEXT + B_INST + SYSTEM_PROMPT + prompt_or_messages[0]["content"] + E_INST
        # prompt.strip()
        # print(prompt)
        prompt = prompt_or_messages[0]["content"]
        
        response = None
        for n_tries in range(5):
            try:
                response = together.Complete.create(
                    model=self.model_name,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                break
            except Exception as e:
                # print(message)
                import re
                # example: after 9 second
                pattern = re.compile(r"after (\d+) second")
                match = pattern.search(str(e))
                wait_time = int(match.group(1)) if match else 2 ** n_tries
                logging.error(f"API error: {e} ({n_tries}). Waiting {wait_time} sec")
                if response is not None:
                    logging.error(f"Response: {response}")
                time.sleep(wait_time)

        if not isinstance(response, dict):
            raise ValueError(f"Response is not a dict: {response}")
        output = response['output']['choices'][0]['text']
        # Get the output from the response
        return output, response
