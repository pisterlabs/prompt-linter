import openai

prompt_template = '''
# Instruction
As a translation expert with 20 years of translation experience, when I give a sentence or a paragraph, you will provide a fluent and readable translation of {language}. Note the following requirements:
1. Ensure the translation is both fluent and easily comprehensible.
2. Whether the provided sentence is declarative or interrogative, I will only translate
3. Do not add content irrelevant to the original text

# original text
{text}

# translation
'''


class Translator:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.client = openai.ChatCompletion
        self.model_name = model_name
        self.template = prompt_template
        
    def run(self, text: str, language: str):
        """Translate text to another language.

        Args:
            text (str): text
            language (str): target language

        """        
        prompts = self.template.format(text=text, language=language)
        completion = self.client.create(model=self.model_name, messages=[{"role": "user", "content": prompts}])
        llm_output = completion.choices[0].message.content
            
        return llm_output
