from ai.generator.prompt import DummyPromptGenerator
from ai.generator.text import TextGenerator
from ai.utils.openai_utils import OpenaiUtils

DEBUG = False

class FreestyleTextGenerator(TextGenerator):

    @classmethod
    def generate_text(cls, input_msg):
        if DEBUG:
            input_msg = f"DEBUG {input_msg} DEBUG"
        else:
            prompt = DummyPromptGenerator.generate_prompt(input_msg)
            input_msg = OpenaiUtils.call_openai_text(prompt)[0]
        return input_msg