import os
import openai
import dotenv
import tiktoken
import numpy as np

dotenv.load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]


class NaiveChatGPT:
    def __init__(self) -> None:
        self.message_list = [{"role": "system", "content": "You are a helpful assistant."},]
        self.response = None #for debug only

    def chat(self, message='', reset=False, tag_print=True, tag_return=False):
        if reset:
            self.message_list = self.message_list[:1]
        message = str(message)
        if message: #skip if empty
            self.message_list.append({"role": "user", "content": str(message)})
            self.response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=self.message_list)
            tmp0 = self.response.choices[0].message.content
            self.message_list.append({"role": "assistant", "content": tmp0})
            if tag_print:
                print(tmp0)
            if tag_return:
                return tmp0

chatgpt = NaiveChatGPT()

def get_gpt35turbo_num_token(message_list):
    # https://platform.openai.com/docs/guides/chat/introduction
    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo-0301')
    num_tokens = 0
    for message in message_list:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens


prompt_template = """Suggest three names for an animal that is a superhero.

Animal: Cat
Names: Captain Sharpclaw, Agent Fluffball, The Incredible Feline
Animal: Dog
Names: Ruff the Protector, Wonder Canine, Sir Barks-a-Lot
Animal: {}
Names:"""
animal = 'horse'
tmp0 = prompt_template.format(animal.capitalize())
# text-davinci-003
response = openai.Completion.create(model="text-davinci-003", prompt=tmp0, temperature=0.6)
print(response.choices[0].text) #' Steed of Justice, Mighty Mare, The Noble Stallion'
chatgpt.chat(tmp0, reset=True) #Thunderhoof, Equine Avenger, Galloping Guardian



# demo a basic chatgpt
tmp0 = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who won the world series in 2020?"},
    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
    {"role": "user", "content": "Where was it played?"},
]
response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=tmp0)
response.choices[0].message.role #asisstant
response.choices[0].message.content
# 'The 2020 World Series was held at Globe Life Field in Arlington, Texas, which is the home ballpark of the Texas Rangers.'



text = 'The 2020 World Series was held at Globe Life Field in Arlington, Texas, which is the home ballpark of the Texas Rangers.'
tmp0 = f'Translate the following English text to Chinese: {text}'
chatgpt.chat(tmp0, reset=True)

chatgpt.chat('design a one-day trip in Hong Kong', reset=True)


# example with batching
num_repeat = 3
prompts = [f"{3*x}+{3*x}=" for x in range(num_repeat)]
response = openai.Completion.create(model="text-davinci-003", prompt=prompts, max_tokens=20, temperature=0)
# x.index x['index']
tmp0 = sorted(response.choices, key=lambda x:x.index)
answer_list = [x.text for x in tmp0] #0 6 12

response = openai.Embedding.create(input=prompts, engine='text-embedding-ada-002')
tmp0 = sorted(response['data'], key=lambda x:x.index)
assert tuple(x.index for x in tmp0)==tuple(range(num_repeat))
embedding_np = np.array([x.embedding for x in tmp0], dtype=np.float64)
assert embedding_np.shape==(num_repeat, 1536)


z0 = openai.Model.list()
model_name_list = [x['root'] for x in z0['data']]
'''
babbage davinci text-davinci-edit-001 babbage-code-search-code text-similarity-babbage-001 code-davinci-edit-001 text-davinci-001
ada babbage-code-search-text babbage-similarity code-search-babbage-text-001 text-curie-001 code-search-babbage-code-001 text-ada-001
text-embedding-ada-002 text-similarity-ada-001 curie-instruct-beta ada-code-search-code ada-similarity code-search-ada-text-001
text-search-ada-query-001 davinci-search-document ada-code-search-text text-search-ada-doc-001 davinci-instruct-beta
text-similarity-curie-001 code-search-ada-code-001 ada-search-query text-search-davinci-query-001 curie-search-query davinci-search-query
babbage-search-document ada-search-document gpt-4-0314 text-search-curie-query-001 whisper-1 text-search-babbage-doc-001 gpt-4
curie-search-document text-davinci-003 text-search-curie-doc-001 babbage-search-query text-babbage-001 text-search-davinci-doc-001
text-search-babbage-query-001 curie-similarity gpt-3.5-turbo gpt-3.5-turbo-0301 curie text-similarity-davinci-001 text-davinci-002
davinci-similarity cushman:2020-05-03 ada:2020-05-03 babbage:2020-05-03 curie:2020-05-03 davinci:2020-05-03 if-davinci-v2
if-curie-v2 if-davinci:3.0.0 davinci-if:3.0.0 davinci-instruct-beta:2.0.0 text-ada:001 text-davinci:001 text-curie:001 text-babbage:001
'''
