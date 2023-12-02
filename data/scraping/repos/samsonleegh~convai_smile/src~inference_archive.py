from simpletransformers.conv_ai import ConvAIModel, ConvAIArgs
from transformers import cached_path
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer

# model trained on https://colab.research.google.com/drive/1M6bjD4zbn8FHY83ZOap6d7vMqXKCLuPi?authuser=2#scrollTo=1SEeIDdsDmlJ
interact_args = {
    "cache_dir": "./cache_dir/",
    "max_length":50,
    "do_sample":True, #sampling, False will set to greedy encoding 
    "temperature":0.7, 
    "top_k":0, 
    "top_p":0.9,
    "max_history":5
}
tuned_model = ConvAIModel("gpt", "./saved_model",
                          use_cuda=False,
                          args=interact_args)

def generate_reply(personality, history, user_input):
    response, history = tuned_model.interact_single(user_input, 
                                                    history,
                                                    personality=personality)
    return response, history

# USER_INPUT = "I am suffering from anxiety and depression. What should I do?"

# generate_reply(personality=[], history=[], user_input=USER_INPUT)

if __name__ == '__main__':
    PERSONALITY = []
    HISTORY = []
    while True:
        USER_INPUT = input()
        response, history = generate_reply(PERSONALITY, history=HISTORY, user_input=USER_INPUT)
        print(response, history)
        HISTORY = HISTORY + history
