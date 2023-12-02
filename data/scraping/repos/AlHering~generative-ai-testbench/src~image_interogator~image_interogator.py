# -*- coding: utf-8 -*-
"""
****************************************************
*      generative_ai_testbench:image_interogator                 
*            (c) 2023 Alexander Hering             *
****************************************************
"""
# Taken from https://nayakpplaban.medium.com/ask-questions-to-your-images-using-langchain-and-python-1aeb30f38751 and ajusted
import torch
import os
from langchain.tools import BaseTool
from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection, AutoProcessor, Blip2ForConditionalGeneration, AutoModelForCausalLM
from PIL import Image
from langchain.agents import initialize_agent
from src.configuration import configuration as cfg
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from typing import List, Any
from langchain.agents import Tool, AgentExecutor, BaseMultiActionAgent
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, LLMMathChain


# Taken from https://github.com/pharmapsychotic/clip-interrogator
INTEROGATION_MODELS = {
    'blip-base': 'Salesforce/blip-image-captioning-base',   # 990MB
    'blip-large': 'Salesforce/blip-image-captioning-large',  # 1.9GB
    'blip2-2.7b': 'Salesforce/blip2-opt-2.7b',              # 15.5GB
    'blip2-flan-t5-xl': 'Salesforce/blip2-flan-t5-xl',      # 15.77GB
    'git-large-coco': 'microsoft/git-large-coco',           # 1.58GB
}


class ImageInterogationTool(BaseTool):
    """
    Image Interogation Tools class.
    """
    name = "Image interogator"
    description = "Use this tool when given the path to an image that you would like to be described. It will return a simple caption describing the image."

    def _run(self, img_path, model="blip-large"):
        """
        Runner method of tool for getting a caption for a given image.
        :param img_path: Path to image file.
        :param model: Prefered model for interogation. Defaults to 'blip-large'.
        """
        image = Image.open(img_path).convert("RGB")

        device = "cpu"  # cuda#
        self.dtype = torch.float16 if device == 'cuda' else torch.float32

        model_path = INTEROGATION_MODELS[model]
        if model.startswith('git-'):
            interogator = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float32).to(device)
        elif model.startswith('blip2-'):
            interogator = Blip2ForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=self.dtype).to(device)
        else:
            interogator = BlipForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=self.dtype).to(device)

        processor = AutoProcessor.from_pretrained(model_path)

        inputs = processor(image, return_tensors="pt").to(device)
        output = interogator.generate(**inputs, max_new_tokens=20)

        caption = processor.decode(output[0], skip_special_tokens=True)

        return caption

    def _arun(self, query: str) -> None:
        """
        Asynchronous runner method. Not Implemented.
        :param query: Base paramenter.
        :raises NotImplementedError: Async method is not supported.
        """
        raise NotImplementedError("This tool does not support async.")


class ObjectDetectionTool(BaseTool):
    """
    Object Detection Tool class
    """
    name = "Object detector"
    description = "Use this tool when given the path to an image that you would like to detect objects. It will return a list of all detected objects. Each element in the list in the format: [x1, y1, x2, y2] class_name confidence_score."

    def _run(self, img_path):
        """
        Runner method for object detection.
        :param img_path: Path to image file.
        """
        image = Image.open(img_path).convert("RGB")

        processor = DetrImageProcessor.from_pretrained(
            "facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50")

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        # let"s only keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.9)[0]

        detections = ""
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detections += "[{}, {}, {}, {}]".format(
                int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            detections += " {}".format(model.config.id2label[int(label)])
            detections += " {}\n".format(float(score))

        return detections

    def _arun(self, query: str):
        """
        Asynchronous runner method. Not Implemented.
        :param query: Base paramenter.
        :raises NotImplementedError: Async method is not supported.
        """
        raise NotImplementedError("This tool does not support async")


class ImageInterogationAgent(BaseMultiActionAgent):
    """
    Class, representing an Image Interogation Agent.
    """

    def __init__(self, general_llm: Any) -> None:
        """
        Initiation method.
        :param general_llm: General LLM.
        """
        self.general_llm = general_llm
        self.tools = self._initiate_tools()
        self.conversational_memory = ConversationBufferWindowMemory(
            memory_key='chat_history',
            k=5,
            return_messages=True
        )
        self.agent = initialize_agent(
            agent="zero-shot-react-description",
            tools=self.tools,
            llm=self.general_llm,
            max_iterations=5,
            verbose=True,
            memory=self.conversational_memory,
            early_stopping_method='generate'
        )
        self.run = self.agent.run

    """
    Tools
    """

    def _initiate_tools(self) -> List[Tool]:
        """
        Internal method for initiating tools.
        """
        return [
            self._initiate_general_llm_tool,
            ImageInterogationTool(),
            ObjectDetectionTool()
        ]

    def _initiate_general_llm_tool(self) -> Tool:
        """
        Internal method for initiating general LLM tool.
        """
        promt_template = PromptTemplate(
            input_variables=["input"],
            template="{input}"
        )
        llm_chain = LLMChain(llm=self.general_llm, prompt=promt_template)

        return Tool(
            name="General Language Model",
            func=llm_chain.run,
            description="Use this tool for general purpose question answering and logic."
        )


def run_example_process(self) -> None:
    """
    Method for running example process.
    """
    llm = LlamaCpp(
        model_path=os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                "orca_mini_7B-GGML/orca-mini-7b.ggmlv3.q4_1.bin"),
        temperature=0
    )
    agent_wrapper = ImageInterogationAgent(llm)

    print("STARTING PROCESS")
    img_path = os.path.join(cfg.PATHS.DATA_PATH, "assets", "Parsons_PR.jpg")
    print("="*50)
    print("STEP 1")
    user_question = "generate a caption for this image?"
    response = agent_wrapper.run(
        f"{user_question}, this is the image path: {img_path}")
    print(response)

    print("="*50)
    print("STEP 2")

    user_question = "Please tell me what are the items present in the image."
    response = agent_wrapper.run(
        f'{user_question}, this is the image path: {img_path}')
    print(response)

    print("="*50)
    print("STEP 3")
    user_question = "Please tell me the bounding boxes of all detected objects in the image."
    response = agent_wrapper.run(
        f'{user_question}, this is the image path: {img_path}')
    print(response)
