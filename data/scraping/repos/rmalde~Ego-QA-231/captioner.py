# This file takes in a question and the clip results, and fetches the most plausible action

import langchain
import os
import io
from api import *
from datasets import load_dataset
from transformers import (
    CLIPProcessor,
    CLIPModel,
    GPT2TokenizerFast,
    ViTImageProcessor,
    VisionEncoderDecoderModel,
    MBart50TokenizerFast,
)
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from promptcap import PromptCap
from promptcap import PromptCap_VQA
from langchain.llms import OpenAI

from params import CaptionerParams


class Captioner:
    def __init__(self, captioner_name, captioner_params):
        self.captioner_name = captioner_name
        self.captioner_params = captioner_params
        if captioner_name == "nlpconnect/vit-gpt2-image-captioning":
            self.model = VisionEncoderDecoderModel.from_pretrained(
                "nlpconnect/vit-gpt2-image-captioning"
            )
            self.image_processor = ViTImageProcessor.from_pretrained(
                "nlpconnect/vit-gpt2-image-captioning"
            )
            self.tokenizer = GPT2TokenizerFast.from_pretrained(
                "nlpconnect/vit-gpt2-image-captioning"
            )
        elif captioner_name == "promptcap":
            self.model = PromptCap("vqascore/promptcap-coco-vqa")
            if torch.cuda.is_available():
                self.model.cuda()
        elif captioner_name == "blip":
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")
        else:
            raise RuntimeError(f"Unsupported Captioner: {captioner_name}")

    def caption(self, image, question=None, choices=None):
        """
        image is a PIL Image
        """

        generated_text = None

        if self.captioner_name == "nlpconnect/vit-gpt2-image-captioning":
            pixel_values = self.image_processor(image, return_tensors="pt").pixel_values
            generated_ids = self.model.generate(
                pixel_values, do_sample=True, max_new_tokens=50, top_k=5
            )
            generated_text = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            query = question
        
        elif self.captioner_name == "blip":
            inputs = self.processor(image, return_tensors="pt").to("cuda")
            out = self.model.generate(**inputs, max_new_tokens=50)
            generated_text = self.processor.decode(out[0], skip_special_tokens=True)

            query = question

        elif self.captioner_name == "promptcap":
            # promptcap needs the image to be a file, not a PIL image
            # TODO: have the dataloader give filenames rather than PIL objects
            file_object = io.BytesIO()
            image.save(file_object, format='PNG')
            file_object.seek(0)

            if self.captioner_params.question_type == CaptionerParams.Configs.Caption or question is None:
                query = "what does the image describe?"

                generated_text = self.model.caption(
                    query, file_object
                ) 
            elif self.captioner_params.question_type == CaptionerParams.Configs.Q_Caption:
                query = (
                    "Describe the image in a way that helps ChatGPT answer the question: \""
                    + question + "?\""
                )

                generated_text = self.model.caption(
                    query, file_object
                ) 
            elif self.captioner_params.question_type == CaptionerParams.Configs.Q:
                query = (
                    question + "?"
                )
                generated_text = self.model.caption(
                    query, file_object
                )

            elif self.captioner_params.question_type == CaptionerParams.Configs.Q_Answer:
                query = (
                    question + "? Your choices are: " + ", ".join(choices) + "."
                )

                generated_text = self.model.caption(
                    query, file_object
                )

            elif self.captioner_params.question_type == CaptionerParams.Configs.Caption_Q_Answer:
                query = (
                    "Describe the image in a way that helps ChatGPT answer the question: \"" +
                    question + "?\" Where the answer choices for the question are: " + ", ".join(choices) + "."
                )

                generated_text = self.model.caption(
                    query, file_object
                )

            elif self.captioner_params.question_type == CaptionerParams.Configs.Q_Cracked:

                llm = OpenAI(model_name="gpt-3.5-turbo")
                improved_question = llm("A person is trying to ask questions about an image to an AI model, that will then reply with the answer. However, their questions may be unclear in terms of what they are looking for. Given the following question: \"" + question + "\" and the following answers the user is expecting: " + ", ".join(choices) + ". come up with a better and more detailed question to ask the AI model")

                query = (
                    improved_question
                )

                generated_text = self.model.caption(
                    query, file_object
                )

            elif self.captioner_params.question_type == CaptionerParams.Configs.VQA:

                vqa_model = PromptCap_VQA(promptcap_model="vqascore/promptcap-coco-vqa", qa_model="allenai/unifiedqa-t5-base")

                query = (
                    question
                )

                generated_text = vqa_model.vqa(question, file_object)
            
            elif self.captioner_params.question_type == CaptionerParams.Configs.VQA_Answer:

                vqa_model = PromptCap_VQA(promptcap_model="vqascore/promptcap-coco-vqa", qa_model="allenai/unifiedqa-t5-base")

                query = (
                    question
                )

                generated_text = vqa_model.vqa_multiple_choice(question, file_object, choices)
            
        else:
            raise RuntimeError(f"Unsupported Captioner: {self.captioner_name}")

        # Plotting Code
        # plt.imshow(np.asarray(image))
        # plt.show()

        print("Query", query)
        print("Generated Text: ", generated_text)

        return generated_text


if __name__ == "__main__":
    # dummy image
    url = "https://raw.githubusercontent.com/michael-franke/npNLG/main/neural_pragmatic_nlg/pics/06-3DS-example.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    # # test ViT
    # print("Testing ViT Captioner...")
    # captioner = Captioner("nlpconnect/vit-gpt2-image-captioning", CaptionerParams)
    # caption = captioner.caption(image)
    # print("Caption: ", caption)

    # test PromptCap
    print("Testing PromptCap")
    captioner = Captioner("promptcap", CaptionerParams)
    caption = captioner.caption(image, question="What is the color of the sky?")
    print("Caption with query: ", caption)
    caption = captioner.caption(image)
    print("Caption no query: ", caption)
