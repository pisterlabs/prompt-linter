from ai.generator.prompt import AdPosterPromptGenerator
from ai.generator.text import TextGenerator
from ai.utils.openai_utils import OpenaiUtils

DEBUG = False

class AdPosterTextGenerator(TextGenerator):

    @classmethod
    def generate_text(cls, input_msg):
        if DEBUG:
            output_texts = [f"DEBUG {input_msg} DEBUG"]
        else:
            prompt = AdPosterPromptGenerator.generate_prompt(input_msg)
            output_texts = OpenaiUtils.call_openai_text(prompt)
        return output_texts[0]