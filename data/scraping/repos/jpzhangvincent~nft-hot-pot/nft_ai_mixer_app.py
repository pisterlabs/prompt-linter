import streamlit as st
import pandas as pd
import requests
import time
import json
import os
import re
from dotenv import load_dotenv
import openai
from PIL import Image
import random
load_dotenv() 

COVALENT_API = os.environ['COVALENT_API_KEY']
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
openai.api_key = OPENAI_API_KEY
COVALENT_BASE_URL = 'https://api.covalenthq.com/v1'
st.session_state['final_img_prompt'] = ''

@st.cache()
def get_nfts_from_walletaddress(address, save_json = True):
    """Function to extract relevant NFT metadata given a wallet address

    Args:
        address (_type_): _description_
        save_json (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    endpoint = f'/eth-mainnet/address/{address}/balances_nft/?key={COVALENT_API}'
    url = COVALENT_BASE_URL + endpoint
    result = requests.get(url).json()
    all_nft_data = result["data"].get('items', [])
    all_nft = []
    if all_nft_data:
        for item in all_nft_data:
            item_nft_data = item['nft_data'][0]['external_data']
            #print(item_nft_data)
            if item_nft_data is not None:
                nft_title = item_nft_data['name']
                nft_description = item_nft_data['description']
                nft_url = item_nft_data['asset_url']
                nft_attributes = item_nft_data['attributes']
                nft_attributes = [attrs.values() for attrs in nft_attributes]
                nft_attributes_str= '; '.join([f'{list(at)[0]}: {list(at)[1]}'  for at in nft_attributes])
                nft_item = {'nft_title': nft_title, 'nft_description': nft_description, 'nft_url': nft_url, 'nft_attributes': nft_attributes_str}
                all_nft.append(nft_item)
    if save_json:
        with open('connected_wallet_data.json', 'w', encoding='utf-8') as f:
            json.dump({'nft_data': all_nft}, f, ensure_ascii=False, indent=4)
    return all_nft

@st.cache()
def auto_prompt_generation(nft_metadata, random_select=2):
    if random_select > 0:
        nft_metadata = random.sample(nft_metadata, random_select)
    
    selected_img_urls = [nf['nft_url'] for nf in nft_metadata]
    context_str = ""
    for nd in nft_metadata:
        #print("NFT url:", nd['nft_url'])
        nd_str = "===============\n"
        if nd["nft_description"] is not None and nd["nft_description"]:
            nd_str += f"Background: {nd['nft_description']}\n"
        if nd["nft_attributes"] is not None and nd["nft_attributes"]:
            nd_str += f"Attributes: {nd['nft_attributes']}\n"
        context_str += nd_str

    img_gen_prompt_str = f"Given the objects with the following context:\n {context_str}\nCan you provide a description on the artwork you would like to create and convery a message based on the above context? Please make it as concise as possible."
    print("Text Generation Prompt: ", img_gen_prompt_str)
    # call the chatGPT API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a digital artist specialized in abstract art and interested in web3 and NFT. You want to create your own unique artist style with a vision to inspire creativity and positivity with AI.",
            },
            {"role": "user", "content": img_gen_prompt_str},
        ],
        temperature=0.2,
    )
    image_prompt = response["choices"][0]["message"]["content"]
    return image_prompt, selected_img_urls


@st.cache()
def auto_mixer_image_generation(image_prompt):
    # Generate image with DALL-E API
    response = openai.Image.create(
        prompt=image_prompt,
        n=1,  # Number of images to generate
        size="256x256",
        response_format="url"
    )

    # Get image URL from response
    image_url = response["data"][0]["url"]

    # Load image from URL and display it
    img = Image.open(requests.get(image_url, stream=True).raw)
    return img


# App logic
title = "NFT Mixer x Generative AI"
st.set_page_config(page_title=title, layout="wide")
st.title(title)

wallet_address = st.text_input('Wallet Address: ', '0x0097b9cFE64455EED479292671A1121F502bc954')
random_number = st.number_input('Randomly select the number for NFT Mixer: ', 2)
if wallet_address:
    nft_metadata = get_nfts_from_walletaddress(wallet_address)

run_generation_prompt = st.button('Generate the Image Prompt')

generated_img_prompt = ''
final_img_prompt = ''
if run_generation_prompt:
    generated_img_prompt, selected_img_urls = auto_prompt_generation(nft_metadata, random_number)
    
if generated_img_prompt:
    st.markdown('Selected input images:')
    for url in selected_img_urls:
        st.image(url)
    final_img_prompt = st.text_area("Output for the Image Prompt Generation: ", generated_img_prompt)
    st.session_state['final_img_prompt'] = final_img_prompt

    # run_img_generation = st.button('Generate the image')
    # print(run_img_generation)
    #print(st.session_state['final_img_prompt'])
    st.markdown("Final generated image:")
    if st.session_state['final_img_prompt']:
        final_generated_img_url = auto_mixer_image_generation(final_img_prompt)
        #print(final_generated_img_url)
        st.image(final_generated_img_url)