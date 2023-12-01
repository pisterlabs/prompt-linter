# Testing multiple api calls to gpt and serpapi
from serpapi import GoogleSearch
import os
import ast
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

#NOTE TO SELF.  Past this command in the CLI to display UI: streamlit run DIY.py [ARGUMENTS]

# Declare variables to be used later
searchPhrase = ""
acknowledgement = ""
list_of_tools = ""
arr = []
selected_item = []
first_video_link = ""
zipcode = ""

# Declare state variables to control the flow of the app
topicComplete = False
searchPhraseComplete = False
acknowledgementComplete = False
first_video_linkComplete = False



# Set the sidebar to take in the API keys.  This will be used to call the API's
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password", placeholder="Enter your API key here")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    serp_api_key = st.text_input("SerpApi API Key", key="serp_api_key", type="password")
    "[Get an Serpapi API key](https://serpapi.com/manage-api-key)"
    st.header('Helpful Links:')

#Main page title and input box to start the app
st.title('Generative AI Planning Tool for DIYers')
st.write("This app will help you plan your DIY project.  It will help you find the best resources to help you complete your project.  It will also help you find the best prices on the tools and supplies you will need to complete your project.")
st.write("")
st.write("2 API keys are required. OpenAI and SerpAPI.  You can get a free API key from both places.  Just click on the links in the sidebar to get your API keys.")   
topic = st.text_input("**What are you trying to DIY?**")



if 'topicComplete' not in st.session_state:
    st.session_state.topicComplete = False  # If not, initialize it
if 'searchPhraseComplete' not in st.session_state:
    st.session_state.searchPhraseComplete = False  # If not, initialize it
if 'acknowledgementComplete' not in st.session_state:
    st.session_state.acknowledgementComplete = False  # If not, initialize it
if 'first_video_linkComplete' not in st.session_state:
    st.session_state.first_video_linkComplete = False  # If not, initialize it



# Initiate API keys only if the user has entered a topic
if topic and not st.session_state.topicComplete:
    # API Keys
    llm = OpenAI(openai_api_key=openai_api_key)
    llm2 = OpenAI(openai_api_key=openai_api_key)
    llm3 = OpenAI(openai_api_key=openai_api_key)


# Summarize the topic into a keyword phrase video search for Youtube
if topic and not st.session_state.topicComplete:
#if topic and topicComplete == False:
    # Prompt Template summarize_youtube_template  
    # Simplify the topic into a keyword phrase
    # API Keys
    llm = OpenAI(openai_api_key=openai_api_key)
    summarize_youtube_template = """Summarize this topic into the most optimal Youtube search phrase. Context: {topic}  Youtube Search Phrase:"""

    prompt = PromptTemplate(
        input_variables = ['topic'],
        template=summarize_youtube_template
    )

    formatted_prompt = prompt.format(topic=topic)

    searchPhrase = llm.predict(formatted_prompt)


# Acknowledge the user's reason for visiting.  Let them know you will be helping them with the project as an assistant.
if searchPhrase and not st.session_state.searchPhraseComplete:
    llm2 = OpenAI(openai_api_key=openai_api_key)
#if searchPhrase and searchPhraseComplete == False:
 # Prompt Templates
    # Simplify the topic into a keyword
    summarize_youtube_template = """Acknowledge the user's {topic} with interest.  Be excited to help the user accomplish their goal. Briefly explain that you are gathering resources that will help the user with their project. This will include a couple of how to videos from youtube.  And also building a short list of supplies they will need for their task. Acknowledgement:"""

    prompt = PromptTemplate(
        input_variables = ['topic'],
        template=summarize_youtube_template
    )

    formatted_prompt = prompt.format(topic=topic)

    acknowledgement = llm2.predict(formatted_prompt)

    st.write(acknowledgement)
    #add a space between the acknowledgement and the youtube search results
    st.write("")

# Search Youtube for relevant videos based on the topic.  I'm selecting the first index of the array
if acknowledgement and not st.session_state.acknowledgementComplete:
#if acknowledgement and acknowledgementComplete == False:
    # search youtube for the top video result based on main topic
    params = {
    "api_key": serp_api_key,
    "engine": "youtube",
    "search_query": searchPhrase
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    first_video_link = results['video_results'][0]['link']
    first_video_title = results['video_results'][0]['title']
    first_video_thumbnail = results['video_results'][0]['thumbnail']['static']
    second_video_link = results['video_results'][1]['link']
    second_video_title = results['video_results'][1]['title']
    second_video_thumbnail = results['video_results'][1]['thumbnail']['static']

    # Print the first video title and link
    with st.sidebar:
        st.write("")
        st.write("Here are the top 2 video results on Youtube that could be helpful:")
        st.write(first_video_title)
        st.markdown(f'<a href="{first_video_link}" target="_blank"><img src="{first_video_thumbnail}" alt="Thumbnail" style="width:200px;height:150px;"></a>', unsafe_allow_html=True)
        st.write(second_video_title)
        st.markdown(f'<a href="{second_video_link}" target="_blank"><img src="{second_video_thumbnail}" alt="Thumbnail" style="width:200px;height:150px;"></a>', unsafe_allow_html=True)


# Create a list of tools and supplies needed to complete the project
if first_video_link and not st.session_state.first_video_linkComplete:
    llm3 = OpenAI(openai_api_key=openai_api_key)
#if first_video_link and first_video_linkComplete == False:
     # Prompt Templates
    # Simplify the topic into a keyword
    tools_template = """Generate a list of parts, supplies and necessary tools required to complete the following topic. Only reply with an array. No need to add titles or numbers. Limit it to a maximum list of 9. List them highest priority.  Meaning, if they are installing equipment list the equipment as top of the list. People are more likely to have a screw driver, so list that at the bottom. {topic}.  array:"""

    prompt = PromptTemplate(
        input_variables = ['topic'],
        template=tools_template
    )

    formatted_prompt = prompt.format(topic=topic)

    list_of_tools = llm3.predict(formatted_prompt)

    arr = [item.strip() for item in list_of_tools.split(',')]
    st.write("")
    #print out a nice numbered list of tools and supplies
    st.write("**Here is a list of tools and supplies you will need to complete this project:**")

    # display radio buttons for each item in the list and store the selected item in a variable
    selected_item = st.multiselect("Select the item you want to buy", arr)
    st.write("You selected: ", selected_item)


    # add a radio button to each output in the list to select the item
    #for i in range(len(arr)):
    #    st.write(i+1, arr[i])
st.write("")
# Ask the user for their zip code so we can search for the best prices on the tools and supplies
if (selected_item):
    zipcode = st.text_input("**I can shop around for the best prices for these supplies.  What is your zip code?**")


# Search for the best prices on the tools and supplies

             
if zipcode:
    total = 0
    products_list = []

    for item in selected_item:
        params = {
            "engine": "home_depot",
            "q": item,
            "api_key": serp_api_key,
            "country": "us",
            "delivery_zip": zipcode
        }

        search = GoogleSearch(params)
        results = search.get_dict()

        products = results.get("products", [])[0:1]

        for product in products:
            thumbnails = product.get("thumbnails", [])
            thumbnail = thumbnails[0][0] if thumbnails and thumbnails[0] else "url-to-a-default-thumbnail-image" 
            title = product.get("title", "No title available")
            price = product.get("price", 0)
            link = product.get("link", "")
            
            total += round(price)
            products_list.append({"thumbnail": thumbnail, "title": title, "price": price, "link": link})

    for i in range(0, len(products_list), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(products_list):
                product = products_list[i + j]
                with cols[j]:
                    st.image(product['thumbnail'])
                    st.markdown(f"[{product['title']}]({product['link']})")
                    st.write(f"${product['price']}")
    
    st.write(f"Grand total for all the supplies and parts will cost: ${total}")
