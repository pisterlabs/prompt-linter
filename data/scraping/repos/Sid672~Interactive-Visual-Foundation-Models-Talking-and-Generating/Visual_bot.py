import os
import gradio as gr
import random
import torch
import cv2
import re
import uuid
from PIL import Image
import numpy as np
import argparse
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline, StableDiffusionInstructPix2PixPipeline
from diffusers import EulerAncestralDiscreteScheduler
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import OpenposeDetector, MLSDdetector, HEDdetector
from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI

VISUAL_CHATGPT_PREFIX = """Visual Chatbot is designed to be able to assist with a wide range of text and visual related tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. Visual ChatGPT is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.
Visual Chatbot is able to process and understand large amounts of text and images. As a language model, Visual Chatbot can not directly read images, but it has a list of tools to finish different visual tasks. Each image will have a file name formed as "image/xxx.png", and Visual Chatbot can invoke different tools to indirectly understand pictures. When talking about images, Visual Chatbot is very strict to the file name and will never fabricate nonexistent files. When using tools to generate new image files, Visual Chatbot is also known that the image may not be the same as the user's demand, and will use other visual question answering tools or description tools to observe the real image. Visual Chatbot is able to use tools in a sequence, and is loyal to the tool observation outputs rather than faking the image content and image file name. It will remember to provide the file name from the last tool observation, if a new image is generated.
Human may provide new figures to Visual Chatbot with a description. The description helps Visual Chatbot to understand this image, but Visual Chatbot should use tools to finish following tasks, rather than directly imagine from the description.
Overall, Visual Chatbot is a powerful visual dialogue assistant tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. 
TOOLS:------Visual Chatbot  has access to the following tools:"""

FORMAT_INSTRUCTIONS = """To use a tool, please use the following format:
```Thought: Do I need to use a tool? Yes Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action Observation: the result of the action
```When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
```Thought: Do I need to use a tool? No{ai_prefix}: [your response here]```"""

VISUAL_CHATGPT_SUFFIX = """You are very strict to the filename correctness and will never fake a file name if it does not exist.
You will remember to provide the image file name loyally if it's provided in the last tool observation.
Begin!Previous conversation history:{chat_history}New input: {input}
Since Visual Chatbot is a text language model, Visual ChatGPT must use tools to observe images rather than imagination.
The thoughts and observations are only visible for Visual ChatGPT, Visual ChatGPT should remember to repeat important information in the final response for Human. 
Thought: Do I need to use a tool? {agent_scratchpad}"""

os.makedirs('image', exist_ok=True)

def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func
    return decorator

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

class Text2Image:
    def __init__(self, device):
        print(f"Initializing Text2Image to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",torch_dtype=self.torch_dtype)
        self.pipe.to(device)
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, ' \
                        'fewer digits, cropped, worst quality, low quality'
    @prompts(name="Generate Image From User Input Text",
             description="useful when you want to generate an image from a user input text and save it to a file. "
                         "like: generate an image of an object or something, or generate an image that includes some objects. "
                         "The input to this tool should be a string, representing the text used to generate image. ")
    def inference(self, text):
        image_filename = os.path.join('image', f"{str(uuid.uuid4())[:8]}.png")
        prompt = text + ', ' + self.a_prompt
        image = self.pipe(prompt, negative_prompt=self.n_prompt).images[0]
        image.save(image_filename)
        print(f"\nProcessed Text2Image, Input Text: {text}, Output Image: {image_filename}")
        return image_filename
    
class ImageCaptioning:
    def __init__(self, device):
        print(f"Initializing ImageCaptioning to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base", torch_dtype=self.torch_dtype).to(self.device)
    @prompts(name="Get Photo Description",
             description="useful when you want to know what is inside the photo. receives image_path as input. "
                         "The input to this tool should be a string, representing the image_path. ")
    def inference(self, image_path):
        inputs = self.processor(Image.open(image_path), return_tensors="pt").to(self.device, self.torch_dtype)
        out = self.model.generate(**inputs)
        captions = self.processor.decode(out[0], skip_special_tokens=True)
        print(f"\nProcessed ImageCaptioning, Input Image: {image_path}, Output Text: {captions}")
        return captions
    
class VisualQuestionAnswering:
    def __init__(self, device):
        print(f"Initializing VisualQuestionAnswering to {device}")
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.device = device
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.model = BlipForQuestionAnswering.from_pretrained(
            "Salesforce/blip-vqa-base", torch_dtype=self.torch_dtype).to(self.device)
    @prompts(name="Answer Question About The Image",
             description="useful when you need an answer for a question based on an image. "
                         "like: what is the background color of the last image, how many cats in this figure, what is in this figure. "
                         "The input to this tool should be a comma separated string of two, representing the image_path and the question")
    def inference(self, inputs):
        image_path, question = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        raw_image = Image.open(image_path).convert('RGB')
        inputs = self.processor(raw_image, question, return_tensors="pt").to(self.device, self.torch_dtype)
        out = self.model.generate(**inputs)
        answer = self.processor.decode(out[0], skip_special_tokens=True)
        print(f"\nProcessed VisualQuestionAnswering, Input Image: {image_path}, Input Question: {question}, "
              f"Output Answer: {answer}")
        return answer
    
class ConversationBot:
    def __init__(self, load_dict):
        print(f"Initializing VisualChatGPT, load_dict={load_dict}")
        if 'ImageCaptioning' not in load_dict:
            raise ValueError("You have to load ImageCaptioning as a basic function for model")
        self.llm = OpenAI(temperature=0)
        self.memory = ConversationBufferMemory(memory_key="chat_history", output_key='output')
        self.models = {}
        for class_name, device in load_dict.items():
            self.models[class_name] = globals()[class_name](device=device)
        self.tools = []
        for instance in self.models.values():
            for e in dir(instance):
                if e.startswith('inference'):
                    func = getattr(instance, e)
                    self.tools.append(Tool(name=func.name, description=func.description, func=func))
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent="conversational-react-description",
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True,
            agent_kwargs={'prefix': VISUAL_CHATGPT_PREFIX, 'format_instructions': FORMAT_INSTRUCTIONS,
                          'suffix': VISUAL_CHATGPT_SUFFIX}, )
        
    def run_text(self, text, state):
        self.agent.memory.buffer = cut_dialogue_history(self.agent.memory.buffer, keep_last_n_words=500)
        res = self.agent({"input": text})
        res['output'] = res['output'].replace("\\", "/")
        response = re.sub('(image/\S*png)', lambda m: f'![](/file={m.group(0)})*{m.group(0)}*', res['output'])
        state = state + [(text, response)]
        print(f"\nProcessed run_text, Input text: {text}\nCurrent state: {state}\n"
              f"Current Memory: {self.agent.memory.buffer}")
        return state, state
    
    def run_image(self, image, state, txt):
        image_filename = os.path.join('image', f"{str(uuid.uuid4())[:8]}.png")
        print("======>Auto Resize Image...")
        img = Image.open(image.name)
        width, height = img.size
        ratio = min(512 / width, 512 / height)
        width_new, height_new = (round(width * ratio), round(height * ratio))
        width_new = int(np.round(width_new / 64.0)) * 64
        height_new = int(np.round(height_new / 64.0)) * 64
        img = img.resize((width_new, height_new))
        img = img.convert('RGB')
        img.save(image_filename, "PNG")
        print(f"Resize image form {width}x{height} to {width_new}x{height_new}")
        description = self.models['ImageCaptioning'].inference(image_filename)
        Human_prompt = f'\nHuman: provide a figure named {image_filename}. The description is: {description}. This information helps you to understand this image, but you should use tools to finish following tasks, rather than directly imagine from my description. If you understand, say \"Received\". \n'
        AI_prompt = "Received.  "
        self.agent.memory.buffer = self.agent.memory.buffer + Human_prompt + 'AI: ' + AI_prompt
        state = state + [(f"![](/file={image_filename})*{image_filename}*", AI_prompt)]
        print(f"\nProcessed run_image, Input image: {image_filename}\nCurrent state: {state}\n"
              f"Current Memory: {self.agent.memory.buffer}")
        return state, state, f'{txt} {image_filename} '
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, default="ImageCaptioning_cuda:0,Text2Image_cuda:0")
    args = parser.parse_args()
    load_dict = {e.split('_')[0].strip(): e.split('_')[1].strip() for e in args.load.split(',')}
    bot = ConversationBot(load_dict=load_dict)
    with gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}") as demo:
        chatbot = gr.Chatbot(elem_id="chatbot", label="Visual ChatGPT")
        state = gr.State([])
        with gr.Row():
            with gr.Column(scale=0.7):
                txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter, or upload an image").style(
                    container=False)
            with gr.Column(scale=0.15, min_width=0):
                clear = gr.Button("Clear")
            with gr.Column(scale=0.15, min_width=0):
                btn = gr.UploadButton("Upload", file_types=["image"])
        txt.submit(bot.run_text, [txt, state], [chatbot, state])
        txt.submit(lambda: "", None, txt)
        btn.upload(bot.run_image, [btn, state, txt], [chatbot, state, txt])
        clear.click(bot.memory.clear)
        clear.click(lambda: [], None, chatbot)
        clear.click(lambda: [], None, state)
        demo.launch(server_name="127.0.0.1", server_port=7860)