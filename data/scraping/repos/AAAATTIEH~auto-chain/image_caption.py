
from langchain.tools import BaseTool
from transformers import BlipProcessor, BlipForConditionalGeneration 
from PIL import Image
import streamlit as st
@st.cache_resource
def load_model():
    model_name = "Salesforce/blip-image-captioning-base"
    device = "cpu"  # cuda
    return BlipForConditionalGeneration.from_pretrained(model_name).to(device)

@st.cache_resource
def load_processor():
    model_name = "Salesforce/blip-image-captioning-base"
    device = "cpu"  # cuda
    return BlipProcessor.from_pretrained(model_name)
class ImageCaptionTool(BaseTool):
    name = "Image captioner"
    description = "Image Path Finder Tool must be used before using this tool" \
                 "Use this tool when given the path to an image that you would like to be described. " \
                  "It will return a simple caption describing the image."

    def _run(self, img_path):
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            return "Image path must be found"
        
        processor = load_processor()
        model = load_model()

        inputs = processor(image, return_tensors='pt').to("cpu")
        output = model.generate(**inputs, max_new_tokens=20)

        caption = processor.decode(output[0], skip_special_tokens=True)

        return caption

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

