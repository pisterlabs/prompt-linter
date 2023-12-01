# Importing the main function from your backend script
from backend import main_function
import streamlit as st
from PIL import Image
import sys
# sys.path.append('../backend')

# st.title('Image Processor')

# uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# if uploaded_file is not None:
#     st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
#     st.write("Storing the embeddings for menu items...")
#     image = Image.open(uploaded_file)

#     result = main_function()  # Run your backend main function

#     st.write(result)
import cohere
from langchain import LLMChain
from langchain import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import requests
from langchain.docstore.document import Document
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent
from langchain.utilities import SerpAPIWrapper
from langchain.agents import AgentType
from langchain.agents import Tool
from langchain.chains import ConversationChain
from langchain.chat_models import ChatAnthropic
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import os

os.environ['ANTHROPIC_API_KEY'] = "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"
os.environ['LANGCHAIN_API_KEY'] = "ls__0aa97ffdedf342068430ab83273564fd"


# os.environ['ANTHROPIC_API_KEY'] = "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"
# os.environ['ANTHROPIC_API_KEY'] = "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"
# os.environ['ANTHROPIC_API_KEY'] = "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"
# os.environ['ANTHROPIC_API_KEY'] = "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"


openai_api_key = ""
ANTHROPIC_API_KEY = ""
SERPAPI_API_KEY = ""


LANGCHAIN_API_KEY = "ls__0aa97ffdedf342068430ab83273564fd"
LANGCHAIN_TRACING_V2 = True
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
LANGCHAIN_PROJECT = "Foodsmith"


cohere_api_key = "5GIQYhLSWrnXOprlPqJSwKu6l7awxtBfi26R9c7c"


def main_function():
    # from langchain.chat_models import ChatOpenAI

    co = cohere.Client(cohere_api_key)
    claude = ChatAnthropic(temperature=0)
    cohereX = ChatAnthropic(temperature=0)
    gpt = ChatAnthropic(temperature=0)

    claude = ChatAnthropic(temperature=0)
    # chain = load_qa_with_sources_chain(claude)

    # User preferences

    user_preferences_input = "I cannot eat beef and im allergic to peanuts"

    # MUst have

    # Extract user prefs

    # Nice to have

    # Another input we can try -> we have to extract what they cannot eat

    user_preferences_input_1 = "I am hindu, specifically brahmin"

    # taste profiles input
    taste_profiles = """[
        # National Cuisines
        "Chinese", "Thai", "Japanese", "Italian", "Korean", "Mexican", "Indian",
        "French", "Spanish", "Mediterranean", "Greek", "American", "Vietnamese",
        "Caribbean", "Brazilian",

        # Food Types
        "Soup", "Ramen", "Ice Cream", "Barbecue", "Udon", "Sandwiches", "Pizza",
        "Pasta", "Seafood", "Steakhouse", "Bakery", "Vegan", "Gluten-Free",
        "Tapas", "Hotpot", "Sushi", "Fried Chicken"
    ] """

    flavor = """
        [
            {
                "Name": "Indian",
                "Flavor": ["Spicy", "Creamy", "Vegetarian", "Stir-fry", "Aromatic", "Curry leaves", "Turmeric", "Paneer", "Lentils", "Ghee", "Masala", "Card
            },
            {
                "Name": "Chinese",
                "Flavor": ["Umami", "Sweet-and-sour", "Spicy", "Stir-fry", "Steamed", "Soy sauce", "Rice", "Ginger", "Bok choy", "Duck", "Oyster sauce", "To
            },
            {
                "Name": "Japanese",
                "Flavor": ["Umami", "Delicate", "Seaweed", "Rice-based", "Fresh", "Sushi rice", "Fish", "Seaweed", "Soy sauce", "Wasabi", "Matcha", "Miso",
            },
            {
                "Name": "Thai",
                "Flavor": ["Spicy", "Aromatic", "Coconut-rich", "Lemongrass", "Citrusy", "Coconut milk", "Thai basil", "Lemongrass", "Chilies", "Fish sauce"
            },
            {
                "Name": "Korean",
                "Flavor": ["Spicy", "Fermented", "Sesame", "Grilled", "Garlicky", "Kimchi", "Gochujang", "Bulgogi meat", "Sesame oil", "Dried seaweed", "Soj
            }
        ]
    }
    """

    cuisineArray = """
        [
            {
                "Name": "Indian",
                "Flavor": ["Spicy", "Creamy", "Vegetarian", "Stir-fry", "Aromatic", "Curry leaves", "Turmeric", "Paneer", "Lentils", "Ghee", "Masala", "Card
            },
            {
                "Name": "Chinese",
                "Flavor": ["Umami", "Sweet-and-sour", "Spicy", "Stir-fry", "Steamed", "Soy sauce", "Rice", "Ginger", "Bok choy", "Duck", "Oyster sauce", "To
            },
            {
                "Name": "Japanese",
                "Flavor": ["Umami", "Delicate", "Seaweed", "Rice-based", "Fresh", "Sushi rice", "Fish", "Seaweed", "Soy sauce", "Wasabi", "Matcha", "Miso",
            },
            {
                "Name": "Thai",
                "Flavor": ["Spicy", "Aromatic", "Coconut-rich", "Lemongrass", "Citrusy", "Coconut milk", "Thai basil", "Lemongrass", "Chilies", "Fish sauce"
            },
            {
                "Name": "Korean",
                "Flavor": ["Spicy", "Fermented", "Sesame", "Grilled", "Garlicky", "Kimchi", "Gochujang", "Bulgogi meat", "Sesame oil", "Dried seaweed", "Soj
            }
        ]
    }
    """

    # restaurant menu input
    restaurant_menu_input_dict = {
        "restaurant": "Up Thai",
        "location": "1411 2nd Ave, New York, NY 10021",
        "contact": {
            "phone": "(212) 256-1188",
            "email": "info@upthai.com",
            "website": "http://www.upthai.com"
        },
        "menu": {
            "appetizers": [
                {
                    "id": "a1",
                    "name": "Chicken Satay",
                    "description": "Grilled chicken skewers served with peanut sauce",
                    "price": 8.95
                },
                {
                    "id": "a2",
                    "name": "Spring Rolls",
                    "description": "Crispy vegetable rolls served with sweet chili sauce",
                    "price": 7.95
                },
                {
                    "id": "a3",
                    "name": "Tom Yum Soup",
                    "description": "Spicy shrimp soup with mushrooms, lemongrass, and kaffir lime",
                    "price": 9.95
                },
                {
                    "id": "a4",
                    "name": "Vegetable Tempura",
                    "description": "Deep-fried mixed vegetables served with dipping sauce",
                    "price": 6.95
                },
                {
                    "id": "a5",
                    "name": "Calamari",
                    "description": "Fried calamari rings served with garlic mayo",
                    "price": 9.95
                }
            ],
            "mains": [
                {
                    "id": "m1",
                    "name": "Pad Thai",
                    "description": "Stir-fried rice noodles with shrimp, peanuts, and lime",
                    "price": 14.95
                },
                {
                    "id": "m2",
                    "name": "Green Curry",
                    "description": "Spicy green curry with chicken, bamboo shoots, and basil",
                    "price": 15.95
                },
                {
                    "id": "m3",
                    "name": "Massaman Curry",
                    "description": "Creamy curry with beef, potatoes, and peanuts",
                    "price": 16.95
                },
                {
                    "id": "m4",
                    "name": "Fried Rice",
                    "description": "Thai-style fried rice with your choice of meat and vegetables",
                    "price": 13.95
                },
                {
                    "id": "m5",
                    "name": "Spicy Basil Chicken",
                    "description": "Stir-fried chicken with basil, bell peppers, and onions",
                    "price": 14.95
                }
            ],
            "desserts": [
                {
                    "id": "d1",
                    "name": "Mango Sticky Rice",
                    "description": "Fresh mango served with sweet sticky rice",
                    "price": 6.95
                },
                {
                    "id": "d2",
                    "name": "Coconut Ice Cream",
                    "description": "Homemade coconut ice cream",
                    "price": 5.95
                },
                {
                    "id": "d3",
                    "name": "Banana Fritters",
                    "description": "Deep-fried banana slices served with honey",
                    "price": 5.95
                },
                {
                    "id": "d4",
                    "name": "Chocolate Mousse",
                    "description": "Rich chocolate mousse with a hint of chili",
                    "price": 7.95
                },
                {
                    "id": "d5",
                    "name": "Tapioca Pudding",
                    "description": "Tapioca pearls cooked in coconut milk",
                    "price": 5.95
                }
            ],
            "beverages": [
                {
                    "id": "b1",
                    "name": "Thai Iced Tea",
                    "description": "Sweet and creamy iced tea",
                    "price": 3.95
                },
                {
                    "id": "b2",
                    "name": "Coconut Water",
                    "description": "Fresh coconut water served in a coconut shell",
                    "price": 4.95
                },
                {
                    "id": "b3",
                    "name": "Mojito",
                    "description": "Classic mojito with a Thai twist",
                    "price": 7.95
                },
                {
                    "id": "b4",
                    "name": "Lemonade",
                    "description": "Homemade lemonade with a splash of lychee syrup",
                    "price": 3.95
                },
                {
                    "id": "b5",
                    "name": "Hot Tea",
                    "description": "Selection of herbal teas",
                    "price": 2.95
                }
            ]
        }
    }

    restaurant_menu_input = str(restaurant_menu_input_dict)

    # sources = [
    #   create_embedding(user_preferences_input),
    #   input_user_tasteprofile_embedding("tasteprofile_input"),
    #   input_restaurant_menu_embedding("restaurant_menu_input")
    # ]

    # def qa(chain, question):
    #   inputs = {"input_documents": sources, "question": question}
    #   outputs = chain(inputs, return_only_outputs=True)["output_text"]
    #   return outputs

    def query_tasteprofile_embedding():

        # You can write your tasteprofile embedding querying
        return True

        # Based on how you want it to work, returns all the embeddings

    def create_embedding(input=["Buttery", "Creamy", "Herbaceous", "Wine-based", "Rich", "Savory", "Garlicky", "Earthy", "Fruity", "Nutty", "Tangy", "Mildly Spiced", "Seafood-flavored", "Floral", "Sweet"]
                         ):

        # Input must be an array

        # Use cohere embedding api here

        response = co.embed(
            texts=input,
            model='embed-english-light-v2.0',
        )

        return response.embeddings

    print(create_embedding())

    def input_user_tasteprofile_embedding(input):
        # Hard code this to return the embeddings of the tasteprofile
        # Embedding we have

        embedding = create_embedding(input)

        return True

    def cuisine_profile_generator_for_user_taste_profile(description):

        template = """Answer the question based on the context below. If the
    question cannot be answered using the information provided answer
    with "I don't know".

    Context: You are a food cuisine flavor classifer. Using {flavor} as a reference, from {cuisineArray} create the flavor profile for 
    the user and return it as a array

    

    Answer: """

        prompt_template = PromptTemplate(
            input_variables=["flavor", "cuisineArray"],
            template=template
        )

        claude_chain = LLMChain(prompt=prompt_template, llm=claude)

        item_flavor_profile = claude_chain.run(flavor=flavor,
                                               cuisineArray=cuisineArray,
                                               )

        embedding_for_item_flavor_profile = create_embedding(
            item_flavor_profile)

        # Store this embedding in the collection

        # Use the embedding created from user profile to check how similar these embeddings ar

        return embedding_for_item_flavor_profile

    def cuisine_profile_generator_for_each_menu_item(description):

        template = """Answer the question based on the context below. If the
    question cannot be answered using the information provided answer
    with "I don't know".

    Context: You are a food cuisine flavor classifer. Using {flavor} as a reference, from {description} create the flavor profile for 
    the dish and return it as a array

    

    Answer: """

        prompt_template = PromptTemplate(
            input_variables=["flavor", "description"],
            template=template
        )

        claude_chain = LLMChain(prompt=prompt_template, llm=claude)

        item_flavor_profile = list(claude_chain.run(flavor=flavor,
                                                    description=description,
                                                    ))

        # print(type(item_flavor_profile))

        embedding_for_item_flavor_profile = create_embedding(
            item_flavor_profile)

        print(str(embedding_for_item_flavor_profile))

        # Store this embedding in the collection

        # Use the embedding created from user profile to check how similar these embeddings ar

        return embedding_for_item_flavor_profile

    # cuisine_profile_generator_for_each_menu_item("Grilled chicken skewers served with peanut sauce")

    def generate_menu_embeddings(restaurant_data):
        # Dictionary to hold the embeddings
        menu_embeddings = {}

        # Looping through each category in the menu
        for category, items in restaurant_data['menu'].items():
            # Initialize category list
            menu_embeddings[category] = []

            # Looping through each item in the category
            for item in items:
                # Getting the description
                description = item['description']

                # Generating the embedding for the description
                embedding = cuisine_profile_generator_for_each_menu_item(
                    description)

                # Adding the embedding to the corresponding category
                menu_embeddings[category].append({item['name']: embedding})

                # Store this in the vector databaes of mongoDB
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        return menu_embeddings

    template = """Answer the question based on the context below. If the
    question cannot be answered using the information provided answer
    with "I don't know".

    Context: You are a food recommend expert. Given the user preferences, allergies and the user's taste profile, you will recommend 3 items from the inputed
    restaurant's menu that you would think the user would like, while also right below every recommendation, you will give your reasoning on why you think the user
    might like it.

    You will do this analysis from the description of the item in the menu but if it is not available, then you shall generate the description from your knowledge what the food might contain

    User Preferences: {user_preferences}

    User Taste Profile: {taste_profile}

    Restaurant menu : {restaurant_menu_input}

    Answer: """

    prompt_template = PromptTemplate(
        input_variables=["user_preferences",
                         "taste_profile", "restaurant_menu_input"],

        template=template
    )

    claude_chain = LLMChain(prompt=prompt_template, llm=claude)

    cohere_chain = LLMChain(prompt=prompt_template, llm=cohereX)

    gpt_chain = LLMChain(prompt=prompt_template, llm=gpt)

    print(claude_chain.run(user_preferences=user_preferences_input,
                           taste_profile=taste_profiles,
                           restaurant_menu_input=restaurant_menu_input))

    claude_output = claude_chain.run(user_preferences=user_preferences_input,
                                     taste_profile=taste_profiles,
                                     restaurant_menu_input=restaurant_menu_input)

    gpt_output = gpt_chain.run(user_preferences=user_preferences_input,
                               taste_profile=taste_profiles,
                               restaurant_menu_input=restaurant_menu_input)

    cohere_output = cohere_chain.run(user_preferences=user_preferences_input,
                                     taste_profile=taste_profiles,
                                     restaurant_menu_input=restaurant_menu_input)

    def most_similar_sentence(sentences):
        """
        This function takes a list of three sentences and returns the one that has the highest average cosine similarity 
        with the other two sentences.

        Parameters:
            sentences (list): A list of three sentences.

        Returns:
            str: The sentence that has the highest average cosine similarity with the other two sentences.
        """

        if len(sentences) != 3:
            return "The function requires exactly three sentences."

        # Convert the sentences into vectors using CountVectorizer
        vectorizer = CountVectorizer().fit_transform(sentences)
        vectors = vectorizer.toarray()

        # Compute the cosine similarity between the sentence vectors
        cosine_matrix = cosine_similarity(vectors)

        # Calculate the average cosine similarity for each sentence with the other two
        avg_cosine_similarities = np.mean(cosine_matrix, axis=1)

        # Find the index of the sentence with the highest average cosine similarity
        most_similar_index = np.argmax(avg_cosine_similarities)

        # Return the sentence with the highest average cosine similarity
        return sentences[most_similar_index]

    # Test the function
    test_sentences = [
        cohere_output,
        gpt_output,
        claude_output
    ]

    life_saving_rec_without_hallucinating = most_similar_sentence(
        test_sentences)

    # return life_saving_rec_without_hallucinating, cohere_output, gpt_output, claude_output

    return claude_output


# Page Config
st.set_page_config(
    page_title="Image Upload App",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded",

)

# App Title
st.title("FoodSmith 📷")

# Columns
col1, col2 = st.columns(2)

# Custom CSS for Image Display
custom_css = """
    <style>
        .uploaded-image {
            max-width: 100%;
            height: auto;
        }
    </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Image Upload
with col1:
    st.header("Upload an Image")
    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "png", "jpeg"])

# Image Display
with col2:
    st.header("Preview")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)


# # Additional Markdown and HTML Styling
# st.markdown("## **Image Analysis Options**")
# st.markdown("<hr/>", unsafe_allow_html=True)


# Loading Indicator for Analysis
if uploaded_file is not None:
    # result = main_function()
    # st.write(result)
    with st.spinner("Analyzing image..."):
        result = main_function()  # Run your backend main function

        # Simulate image analysis here
    st.success("Analysis complete!")

    st.write(result)
