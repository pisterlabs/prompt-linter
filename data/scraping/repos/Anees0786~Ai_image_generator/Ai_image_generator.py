#!/usr/bin/python3

from langchain.llms import OpenAI
from langchain.utilities.dalle_image_generator import DallEAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def generate_image(cmd):
    # Replace 'OPEN_AI_KEY' with your actual OpenAI API key
    openai_key = 'OPEN_AI_KEY'
    
    llm = OpenAI(api_key=openai_key, temperature=0.9)
    
    prompt = PromptTemplate(
        input_variables=["image_desc"],
        template="Generate a detailed prompt to generate an image based on the following description: {image_desc}",
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    image_url = DallEAPIWrapper().run(chain.run(cmd))
    
    return image_url

def main():
    cmd = input("Enter a command for image generation: ")
    
    if not cmd:
        print("Error: Command not provided.")
        return
    
    image_url = generate_image(cmd)
    
    print("Image URL:")
    print(image_url)

if __name__ == "__main__":
    main()
