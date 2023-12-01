from .openai_client import OpenAiClient
from .settings import load_settings


class CyberdolphinOpenAIAdvanced:
    the_settings = None

    @classmethod
    def INPUT_TYPES(s):
        openai_model_list = OpenAiClient.model_list()
        the_settings = load_settings()
        gpt_prompt = the_settings['prompt_templates']['gpt-3.5-turbo']
        example_system_prompt = gpt_prompt['system']
        example_user_prompt = f"{gpt_prompt['prefix']}{the_settings['example_user_prompt']}{gpt_prompt['suffix']}"

        return {
            "required": {
                "model": (openai_model_list, {
                    "default": "gpt-3.5-turbo"}),
                "system_prompt": ('STRING', {
                    "multiline": True,
                    "default": example_system_prompt
                }),
                "user_prompt": ("STRING", {
                    "multiline": True,
                    "default": example_user_prompt
                }),
                "temperature": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01,
                    "help": """
                            What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the
                            output more random, while lower values like 0.2 will make it more focused and deterministic.
                            
                            We generally recommend altering this or top_p but not both.
                    """

                }),
            },
            "optional": {
                "top_p": ("FLOAT", {
                    "default": 1.0, "min": 0.001, "max": 1.0, "step": 0.01,
                    "help": """
                            An alternative to sampling with temperature, called nucleus sampling, where the model
                            considers the results of the tokens with top_p probability mass.
                            So 0.1 means only the tokens comprising the top 10% probability mass are considered.
                            
                            We generally recommend altering this or `temperature` but not both.
                    """
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("gpt_response",)
    FUNCTION = "generate"
    CATEGORY = "🐬 CyberDolphin"

    def generate(self, model: str, system_prompt: str, user_prompt="",
                 temperature: float | None = None, top_p: float | None = None):
        system_content = system_prompt
        user_content = user_prompt

        response = OpenAiClient.complete(
            key="openai",
            model=model,
            temperature=temperature,
            top_p=top_p,
            system_content=system_content,
            user_content=user_content)

        return (f'{response.choices[0].message.content}',)
