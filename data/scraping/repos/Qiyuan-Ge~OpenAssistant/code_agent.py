import openai

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant with a deep knowledge of code and software design. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""


def get_prompt(message: str, chat_history: list[tuple[str, str]], system_prompt: str) -> str:
    texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
    # The first user input is _not_ stripped
    do_strip = False
    for user_input, response in chat_history:
        user_input = user_input.strip() if do_strip else user_input
        do_strip = True
        texts.append(f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')
    message = message.strip() if do_strip else message
    texts.append(f'{message} [/INST]')
    return ''.join(texts)


class CodeAgent:
    def __init__(self, model_name="code-llama"):
        self.client = openai.Completion
        self.model_name = model_name
        
    def run(self, question: str):
        """_summary_

        Args:
            question (str): question

        Returns:
            _type_: _description_
        """        
        
        prompt = get_prompt(question, chat_history=[], system_prompt=DEFAULT_SYSTEM_PROMPT)
        completion = self.client.create(
            model=self.model_name, 
            prompt=prompt, 
            max_tokens=2048,
            temperature=0.1,
            top_p=0.9,
            top_k=50,
        )
        llm_output = completion.choices[0].text
            
        return llm_output
