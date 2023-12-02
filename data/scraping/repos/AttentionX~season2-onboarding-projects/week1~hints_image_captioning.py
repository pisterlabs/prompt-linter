import torch
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import openai

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def run_on_cpu(img_url:str=None):
    if img_url is None:
        img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

    # conditional image captioning
    text = "a photography of"
    inputs = processor(raw_image, text, return_tensors="pt")

    out = model.generate(**inputs)
    print(processor.decode(out[0], skip_special_tokens=True))
    # >>> a photography of a woman and her dog

    # unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt")

    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    print(caption)
    # a woman sitting on the beach with her dog
    
    return caption

def run_on_gpu(f16=False):
    if f16:
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=torch.float16)
    model = model.to('cuda')
    
    img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

    # conditional image captioning
    text = "a photography of"
    
    inputs = processor(raw_image, text, return_tensors="pt")
    if f16:
        inputs = inputs.to("cuda", torch.float16)
    else:
        inputs.to("cuda")

    out = model.generate(**inputs)
    print(processor.decode(out[0], skip_special_tokens=True))
    # >>> a photography of a woman and her dog

    # unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt")
    
    if f16:
        inputs = inputs.to("cuda", torch.float16)
    else:
        inputs = inputs.to("cuda")

    out = model.generate(**inputs)

    caption = processor.decode(out[0], skip_special_tokens=True)
    print(caption)
    # a woman sitting on the beach with her dog

    return caption

def main():
    caption = run_on_cpu()

    print("Image Successfully Read")

    while True:
        query = input("Your question: ")
    
        prompt = f"""
        The following is the description of an image:
        {caption}

        Based on the descrion, answer the following question about the image:
        {query}
        """

        chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                    messages=[{"role": "user", "content": prompt}])
        answer = chat_completion.choices[0].message.content
        print(answer)

if __name__ == "__main__":
    main()
