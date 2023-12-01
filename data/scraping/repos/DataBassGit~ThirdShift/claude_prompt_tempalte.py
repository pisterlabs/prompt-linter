import anthropic

class PromptTemplate():
    def __init__(self):
        pass

    def prompt_tempalte(self, user_input=None):
        print("-- Prompt Template")
        userQuestion = user_input
        prompt = f"{anthropic.HUMAN_PROMPT}{userQuestion}{anthropic.AI_PROMPT}"
        print("-- Prompt: ", prompt)
        return prompt
