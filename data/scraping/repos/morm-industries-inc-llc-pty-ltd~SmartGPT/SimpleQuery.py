import openai
import tiktoken
from src.Cache.FileCache import FileCache

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0301" or "gpt-4":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            # every message follows <im_start>{role/name}\n{content}<im_end>\n
            num_tokens += 4
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
  See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")


def get_cents(tokens, model="gpt-3.5-turbo-0301"):
    """Returns the cost of a number of tokens in dollars."""
    if model == "gpt-3.5-turbo-0301":
        return 0.000002 * tokens
    elif model == "gpt-4":
        return 0.00003 * tokens
    elif model == "gpt-4-32k":
        return 0.00006 * tokens

class SimpleQuery:
    def __init__(self, messages = None, quiet = False, model = "gpt-3.5-turbo-0301"):
        self.quiet = quiet
        self.model = model
        if messages == None:
            self.messages = []
        else: 
            self.messages = messages

    def append(self, role, content):
        if role != "user" and role != "system" and role != "assistant":
            raise ValueError("role must be either 'user' or 'system' or 'assistant'")

        self.messages.append({"role": role, "content": content})

    def tokens(self, model = "gpt-3.5-turbo-0301"):
        return num_tokens_from_messages(self.messages, model)
    
    def cost(self, model = "gpt-3.5-turbo-0301"):
        return get_cents(self.tokens(model), model)

    #@FileCache("chatgpt")
    def make_purchase(self, messages, model):
        if not self.quiet:
            purchase = input("Make purchase? y/n")
        else:
            purchase = "y"

        if purchase == "y":
            return openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=0.5,
            )
        else: 
            return None

    def run(self):
        toks = self.tokens(self.model)
        cost = self.cost(self.model)
        if not self.quiet:
            print(f"Tokens: {toks}")
            print(f"Cost: {cost}")
        response = self.make_purchase(self.messages, self.model)
        if response is not None:
            return response["choices"][0]["message"]["content"].strip()
        else: 
            return None

    def __str__(self):
        return self.__repr__()
    
    def __repr__(self) -> str:
        return f"SimpleQuery({self.messages})"
