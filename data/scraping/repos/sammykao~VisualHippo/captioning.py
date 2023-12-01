import requests
# from PIL import Image
# from transformers import BlipProcessor, BlipForConditionalGeneration
# import os
# import openai
import replicate
import base64
import os
# processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")


def caption(img_file): 
    replicate.Client(api_token='r8_5tvVCPHOE4AzjaqPB0rjf8vkxnkZr3T0t0Eix')
    output = replicate.run(
        "andreasjansson/blip-2:4b32258c42e9efd4288bb9910bc532a69727f9acd26aa08e175713a0a857a608",
        input={"image": img_file, "question": "what food is in the picture"}
    )
    return(output)
    # raw_image = Image.open(img_file).convert('RGB')

    # # conditional image captioning
    # text = "a food picture of"
    # inputs = processor(raw_image, text, return_tensors="pt")
    # out = model.generate(**inputs)
    # return(processor.decode(out[0], skip_special_tokens=True)


