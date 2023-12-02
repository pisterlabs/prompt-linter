# from huggingface_hub import hf_hub_download

from typing import Callable
from .generic import ModelClass

from langchain.tools import hf_hub_download
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import GPT4All


class Model(ModelClass):
    def __init__(self, name: str, prompt_template: str):
        self._message_history = []
        self.name = name
        self._prompt_template = prompt_template
        self._patch_prompt_template()
        # Make sure we have the model we want to use
        print("Downloading / ensuring model exists...", flush=True)
        hf_hub_download(
            repo_id="LLukas22/gpt4all-lora-quantized-ggjt", 
            filename="ggjt-model.bin", 
            local_dir=".")
        print("Model download complete", flush=True)
        # Load the model
        from pyllamacpp.model import Model
        self.model = Model(ggml_model='ggjt-model.bin', n_ctx=2000)
        print("Model loaded", flush=True)
        
    def _patch_prompt_template(self):
        self._prompt_template = self._prompt_template + """
You are to provide Echo's responses to the conversation, using the below exchanges as an example and as history of the conversation so far.
You are not to provide the User's responses. Only provide one response per request."""
        self._prompt_template = self._prompt_template + '{history_of_questions}\n'
        
    def update_prompt_template(self, new_template: str):
        self._prompt_template = new_template
        self._patch_prompt_template()
        
    async def prompt_with_callback(self, prompt: str, callback: Callable[[str], None]) -> None:
        """
        Write a prompt to the bot and callback with the response.
        """
        self._message_history.append({'user': prompt})
        def cb(text):
            self._callback_wrapper(text, callback)
        self.model.generate(
                self._wrapped_prompt(prompt), 
                n_predict=512, 
                new_text_callback=cb, 
                n_threads=8)
        print("Prompt all done", flush=True)
        
    def prompt(self, prompt: str) -> str:
        """
        Write a prompt to the bot and return the response.
        """
        self._message_history.append({'user': prompt})
        return self.model.generate(
                self._wrapped_prompt(prompt), 
                n_predict=512, 
                n_threads=8)
        
    def _callback_wrapper(self, text: str, callback: Callable[[str], None]):
        " Record the response after stripping out the initial prompt "
        print(f">>> {text}", flush=True)
        current_item = self._message_history[-1]
        # Record all past plus this token
        response_so_far = current_item.setdefault('bot', '')
        bot = response_so_far + text
        self._message_history[-1]['bot'] = bot
        # Only respond once we've finished with the prompt
        if len(bot) > len(current_item['full_prompt']):
            callback(text)
        
    def _translate(self, history: dict[str, str]):
        user = history.get('user', '')
        bot = history.get('bot', '').split('\n')[-1]
        if bot:
            if bot.startswith(f'{self.name}: '):
                return f'User: {user}\n{bot}'
            return f'User: {user}\n{self.name}: {bot}'
        return f'User: {user}'
        
    def _wrapped_prompt(self, prompt: str):
        " Take user input, wrap it in a conversational prompt, and return the result "
        history_of_questions = '\n\n'.join(
            [ self._translate(item) for item in self._message_history ][-20:]) # Note, this includes the current prompt
        full_prompt = self._prompt_template.format(history_of_questions=history_of_questions, name=self.name)
        self._message_history[-1]['full_prompt'] = full_prompt
        return full_prompt
    
def get_model_for_chain():
    hf_hub_download(
            repo_id="LLukas22/gpt4all-lora-quantized-ggjt", 
            filename="ggjt-model.bin", 
            local_dir=".")
    print("Model download complete", flush=True)
    # Callbacks support token-wise streaming
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    # Verbose is required to pass to the callback manager
    llm = GPT4All(model='./models/ggjt-model.bin', callback_manager=callback_manager, verbose=True)
    return llm