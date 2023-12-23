import environment
import os



def getImage(imageURL):
    from PIL import Image
    import requests
    from io import BytesIO

    response = requests.get(imageURL)
    responseContent = BytesIO(response.content)
    img = Image.open(responseContent)
    print(img)
    # Write the stuff
    with open("replicate.png", "wb") as f:
        f.write(responseContent.getbuffer())

# !pip install replicate
## get a token: https://replicate.com/account
# from getpass import getpass
# REPLICATE_API_TOKEN = getpass()
# import os
# os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

from langchain.llms import Replicate
from langchain import PromptTemplate, LLMChain


llm = Replicate(model="replicate/dolly-v2-12b:ef0e1aefc61f8e096ebe4db6b2bacc297daf2ef6899f0f7e001ec445893500e5")
prompt = """
Answer the following yes/no question by reasoning step by step. 
Can a dog drive a car?
"""
# print(llm(prompt))


text2image = Replicate(model="stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf", 
                       input={'image_dimensions': '512x512'})
# image_output = text2image("A cat riding a motorcycle by Picasso")
# getImage(image_output)




from langchain.chains import SimpleSequentialChain

dolly_llm = Replicate(model="replicate/dolly-v2-12b:ef0e1aefc61f8e096ebe4db6b2bacc297daf2ef6899f0f7e001ec445893500e5")
text2image = Replicate(model="stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf")

prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

chain = LLMChain(llm=dolly_llm, prompt=prompt)
second_prompt = PromptTemplate(
    input_variables=["company_name"],
    template="Write a description of a logo for this company: {company_name}",
)
chain_two = LLMChain(llm=dolly_llm, prompt=second_prompt)

third_prompt = PromptTemplate(
    input_variables=["company_logo_description"],
    template="{company_logo_description}",
)
chain_three = LLMChain(llm=text2image, prompt=third_prompt)

# Run the chain specifying only the input variable for the first chain.
overall_chain = SimpleSequentialChain(chains=[chain, chain_two, chain_three], verbose=True)
catchphrase = overall_chain.run("colorful socks")
print(catchphrase)