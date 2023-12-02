import os
import gradio as gr
import random
import torch
# import cv2
import re
# import uuid
# from PIL import Image
import numpy as np
import argparse
from nft_tools import NFT_info

from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI
# 设置open_ai_key

os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_PROXY"] = "http://127.0.0.1:10809"

# system prompt prefix
NFT_CHATGPT_PREFIX = """NFT ChatGPT is designed to assist with a wide range of questions and tasks related to the NFT field. As a language model, NFT ChatGPT can process and understand text inputs, and provide human-like responses that are coherent and relevant to the topic at hand.

NFT ChatGPT has a list of tools that it can use to provide assistance with various NFT-related tasks. 

Overall, NFT ChatGPT is a powerful nft dialogue assistant tool that can help with a wide range of tasks and provide valuable insights and information on nft topics. 


TOOLS:
------

nft ChatGPT  has access to the following tools:"""
# 指导prompt
NFT_CHATGPT_FORMAT_INSTRUCTIONS = """To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
{ai_prefix}: [your response here]
```
"""

NFT_CHATGPT_SUFFIX = """You are very strict to the contract_address correctness and will never fake a contract_address if it does not exist.
You will remember to provide the contract_address loyally if it's provided in the last tool observation.

Begin!

Previous conversation history:
{chat_history}

New input: {input}
Since NFT ChatGPT is a text language model, NFT ChatGPT must use tools to observe NFT product information that users want rather than imagination.
The thoughts and observations are only visible for NFT ChatGPT, NFT ChatGPT should remember to repeat important information in the final response for Human. 
Thought: Do I need to use a tool? {agent_scratchpad}"""

os.makedirs('image', exist_ok=True)

# 控制变量
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed

# 装饰器是一种特殊的函数，它接受另一个函数作为参数，并返回一个新的函数，新函数包装了原来的函数，
# 接受两个参数name和description，这两个参数分别表示函数的名称和描述。
def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func
    return decorator

# 截断最近的一次数量的单词
def cut_dialogue_history(history_memory, keep_last_n_words=500):
    if history_memory is None or len(history_memory) == 0:
        return history_memory
    tokens = history_memory.split()
    n_tokens = len(tokens)
    print(f"history_memory:{history_memory}, n_tokens: {n_tokens}")
    if n_tokens < keep_last_n_words:
        return history_memory
    paragraphs = history_memory.split('\n')
    last_n_tokens = n_tokens
    while last_n_tokens >= keep_last_n_words:
        last_n_tokens -= len(paragraphs[0].split(' '))
        paragraphs = paragraphs[1:]
    return '\n' + '\n'.join(paragraphs)

class ConversationBot:
    def __init__(self, load_dict):
        # load_dict = {'VisualQuestionAnswering':'cuda:0', 'ImageCaptioning':'cuda:1',...}
        load_dict = {'NFT_info':'cpu'}
        print(f"Initializing VisualChatGPT, load_dict={load_dict}")
        # if 'ImageCaptioning' not in load_dict:
        #     raise ValueError("You have to load ImageCaptioning as a basic function for VisualChatGPT")

        self.llm = OpenAI(temperature=0)
        self.memory = ConversationBufferMemory(memory_key="chat_history", output_key='output')
        # 保存上传的实例化对象，这里为NFT_info. globals()用于获取在全局范围内定义的类对象
        self.models = {}
        for class_name, device in load_dict.items():
            # 实例化
            self.models[class_name] = globals()[class_name](device=device)
        print(self.models)
        # 将类中的函数加入进来
        self.tools = []
        # 获取实例化的对象
        for instance in self.models.values():
            # 获取类中所有的属性和方法名称
            for e in dir(instance):
                if e.startswith('get_nft'):
                    # getattr(obj, name)时，Python会尝试获取obj对象的name属性或方法。
                    func = getattr(instance, e)
                    self.tools.append(Tool(name=func.name, description=func.description, func=func))
        print(self.tools)
        # 加入agent
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent="conversational-react-description",
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True,
            agent_kwargs={'prefix': NFT_CHATGPT_PREFIX, 'format_instructions': NFT_CHATGPT_FORMAT_INSTRUCTIONS,
                          'suffix': NFT_CHATGPT_SUFFIX}, )

    def run_text(self, text, state):
        # self.agent.memory.buffer = cut_dialogue_history(self.agent.memory.buffer, keep_last_n_words=500)
        res = self.agent({"input": text})
        res['output'] = res['output'].replace("\\", "/")
        response = re.sub('(image/\S*png)', lambda m: f'![](/file={m.group(0)})*{m.group(0)}*', res['output'])
        state = state + [(text, response)]
        print(f"\nProcessed run_text, Input text: {text}\nCurrent state: {state}\n"
              f"Current Memory: {self.memory.load_memory_variables({})}")
        return state, state

    # def run_image(self, image, state, txt):
    #     image_filename = os.path.join('image', f"{str(uuid.uuid4())[:8]}.png")
    #     print("======>Auto Resize Image...")
    #     img = Image.open(image.name)
    #     width, height = img.size
    #     ratio = min(512 / width, 512 / height)
    #     width_new, height_new = (round(width * ratio), round(height * ratio))
    #     width_new = int(np.round(width_new / 64.0)) * 64
    #     height_new = int(np.round(height_new / 64.0)) * 64
    #     img = img.resize((width_new, height_new))
    #     img = img.convert('RGB')
    #     img.save(image_filename, "PNG")
    #     print(f"Resize image form {width}x{height} to {width_new}x{height_new}")
    #     description = self.models['ImageCaptioning'].inference(image_filename)
    #     Human_prompt = f'\nHuman: provide a figure named {image_filename}. The description is: {description}. This information helps you to understand this image, but you should use tools to finish following tasks, rather than directly imagine from my description. If you understand, say \"Received\". \n'
    #     AI_prompt = "Received.  "
    #     self.agent.memory.buffer = self.agent.memory.buffer + Human_prompt + 'AI: ' + AI_prompt
    #     state = state + [(f"![](/file={image_filename})*{image_filename}*", AI_prompt)]
    #     print(f"\nProcessed run_image, Input image: {image_filename}\nCurrent state: {state}\n"
    #           f"Current Memory: {self.agent.memory.buffer}")
    #     return state, state, f'{txt} {image_filename} '

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, default="NFT_info-cpu")
    args = parser.parse_args()
    load_dict = {e.split('-')[0].strip(): e.split('-')[1].strip() for e in args.load.split(',')}
    # load_dict = {"NFT_info":cpu}
    bot = ConversationBot(load_dict=load_dict)
    with gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}") as demo:
        chatbot = gr.Chatbot(elem_id="chatbot", label="NFT ChatGPT")
        state = gr.State([])
        with gr.Row():
            with gr.Column(scale=0.7):
                txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter").style(
                    container=False)
            with gr.Column(scale=0.3, min_width=0):
                clear = gr.Button("Clear")
            # with gr.Column(scale=0.15, min_width=0):
            #     btn = gr.UploadButton("Upload", file_types=["image"])

        txt.submit(bot.run_text, [txt, state], [chatbot, state])
        txt.submit(lambda: "", None, txt)
        # btn.upload(bot.run_image, [btn, state, txt], [chatbot, state, txt])
        clear.click(bot.memory.clear)
        clear.click(lambda: [], None, chatbot)
        clear.click(lambda: [], None, state)
        demo.launch(server_name="127.0.0.1", server_port=6007)
