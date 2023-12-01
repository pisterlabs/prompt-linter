
import backoff
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

class Inference_Model():
    def __init__(self, default_model:str, api_key:str, System_Prompt:str, SC_num:int) -> None:
        self.model_name = default_model
        self.claude_model_list = ["claude-1", "claude-2"]
        assert self.model_name in self.claude_model_list
        self.api_key = api_key
        self.System_Prompt = System_Prompt
        self.SC_num = SC_num

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(100))
    def get_info(self, Prompt_question:str, System_Prompt:str = "", model_name:str = "", api_key:str = "", SC_num:int = 0):
        if System_Prompt == "": System_Prompt = self.System_Prompt
        if model_name == "": model_name = self.model_name
        if api_key == "": api_key = self.api_key
        if SC_num == 0: SC_num = self.SC_num
        assert model_name in self.openai_model_list + self.claude_model_list
        assert SC_num > 0

        if model_name in self.claude_model_list:
            # defaults to os.environ.get("ANTHROPIC_API_KEY")
            anthropic = Anthropic(api_key=api_key)
            completion = anthropic.completions.create(
                model="claude-instant-1.2",
                max_tokens_to_sample=1024,
                prompt=f"{System_Prompt} {HUMAN_PROMPT} {Prompt_question} {AI_PROMPT}",
            )
            import ipdb; ipdb.set_trace()
            return completion.completion, None
        else:
            raise Exception("model_name error")

class Inference_Claude_Model():
    def __init__(self, model_name:str, claude_key:str, System_Prompt:str) -> None:
        self.model_name = model_name
        assert self.model_name in ["claude-1", "claude-2"]
        self.System_Prompt = System_Prompt

        # defaults to os.environ.get("ANTHROPIC_API_KEY")
        self.anthropic = Anthropic(api_key=claude_key)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def get_info(self, Prompt_question:str, System_Prompt:str = ""):
        if System_Prompt == "":
            System_Prompt = self.System_Prompt
        completion = self.anthropic.completions.create(
            model="claude-2",
            max_tokens_to_sample=300,
            prompt=f"{System_Prompt} {HUMAN_PROMPT} {Prompt_question} {AI_PROMPT}",
        )
        return completion.completion, None