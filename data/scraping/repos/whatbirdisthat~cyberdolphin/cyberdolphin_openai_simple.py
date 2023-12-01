from .openai_client import OpenAiClient
from .settings import load_settings


class CyberdolphinOpenAISimple:

    @classmethod
    def INPUT_TYPES(s):
        settings = load_settings()
        return {
            'required': {
                'user_prompt': ('STRING', {
                    'multiline': True,
                    'default': settings['example_user_prompt']
                }),
                'temperature': ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.75, "step": 0.01}),
                'model': (['gpt-3.5-turbo', 'gpt-4'], {
                    'default': settings['openai_compatible']['openai']['model']
                }),
            },
        }

    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('gpt_response',)
    FUNCTION = 'generate'
    CATEGORY = '🐬 CyberDolphin'

    def generate(self, user_prompt="", temperature: float = 1.0, model: str = "gpt-3.5-turbo"):
        settings = load_settings()
        gpt_prompt = settings['prompt_templates']['gpt-3.5-turbo']
        system_content = gpt_prompt['system']
        user_content = f"{gpt_prompt['prefix']} {user_prompt} {gpt_prompt['suffix']}"
        response = OpenAiClient.complete(
            key='openai',
            model=model,
            temperature=temperature,
            top_p=1.0,
            system_content=system_content,
            user_content=user_content)

        return (f'{response.choices[0].message.content}',)
