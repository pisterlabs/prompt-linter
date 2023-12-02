import streamlit as st 
import pandas as pd
import openai
import os
from dotenv import load_dotenv
import pandas as pd
import json

def load_product_data():
    with open('product.json', 'r') as file:
        product_data = json.load(file)
    return product_data

# ...

products = load_product_data()

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.title("Smart Search  Engine")

def is_valid_query(query):
    # Define keywords or patterns that indicate a valid query
    valid_keywords = ["find", "recommend", "suggest", "help with", "what to buy", "product", "can't decide what to buy"]
    
    for keyword in valid_keywords:
        if keyword in query.lower():
            return True
    return False
def get_reply(text):
    
    if not is_valid_query(text):
        return "I'm here to help you find products and make recommendations. Please ask a question related to finding products."
    
    message = [
        {"role": "system", "content": "you are an expert at finding out what people want you are an expert product suggesting machine who knows what a customer wants even with their vague explanation"},
        {"role": "system", "content": "only generate keywords for eg. black shoes, black dress, black heels, black dress, black heels, black dress, black heels, black dress, black heels, black dress, black heels, black dress, black heels, black dress, black heels, black dress, black heels, black dress, black heels, black dress"},
        {"role": "user", "content": "i am going on a date i want to look good and decent not too shabby i am a girl  i want to something elegant and good for first date "},
        {"role": "assistant", "content": " black dress, beautiful heels,or a gown, or a skirt they will look good  "},
        {"role": "user", "content": "I am expecting a baby soon i want to shop but i am a new mother i don't know what top buy "},
        {"role": "assistant", "content": " baby kirb, some baby clothes ,baby oil, baby powder , baby toys, diapers, lactose"},
        {"role": "user", "content": "i just brought a new house suggest me some furniture"},
        {"role": "assistant", "content": " bed, sofa, table, chair, dresser, closet, wardrobe, dresser, couch, bookshelf"},
        {"role": "user", "content": text}
    ]


    response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0.8,
        messages=message
    )
    return response["choices"][0]["message"]["content"]

def get_matching_products(keywords):
    matching_products = []
    for product in products:
        product_keywords = product["keywords"]  # Get the keywords for the current product
        if any(keyword in product_keywords for keyword in keywords):
            matching_products.append(product)
    return matching_products

text= st.text_input("Enter your message", key="unique_key")
s=[]


if st.button("Send"):
    s = get_reply(text)
    keywords_from_response = s.split()  # Extract keywords from GPT-4 response
    st.write(s)
    #st.write("Keywords from response:", keywords_from_response)  # Add this line to debug

    if keywords_from_response:
        matching_products = get_matching_products(keywords_from_response)

        if matching_products:
            st.write("Matching Products:")
            for product in matching_products:
                if os.path.exists(product["image_url"]):
                    st.image(product["image_url"], caption=product["name"], use_column_width=True)
                    st.write("Name:", product["name"])
                    st.write("Price:", product["price"])
                    st.write("Description:", product["description"])
                else:
                    st.write("Name:", product["name"])
                    st.write("Price:", product["price"])
                    st.write("Description:", product["description"])
                    st.write("---")
        else:
            st.write("No matching products found.")



