import os
import streamlit as st
from typing import Literal
from dataclasses import dataclass

from langchain.chains import RetrievalQAWithSourcesChain, ConversationalRetrievalChain, ConversationChain
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.conversation.memory import ConversationSummaryMemory

from langchain.vectorstores import Qdrant
from langchain.document_loaders import WebBaseLoader
from qdrant_client import models, QdrantClient



os.environ["COHERE_API_KEY"] = st.secrets["COHERE_API_KEY"]
QDRANT_HOST = st.secrets["QDRANT_HOST"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]

# "qRMUuF5mmpHClOpnpk5BXsxUvzcnMFmQTUc6Gwh5"
st.set_page_config(page_title="Co:Chat - An LLM-powered chat bot")
st.title("Sheraton-Bot")

#Creating a Class for message
@dataclass
class Message :
    """Class for keepiong track of chat Message."""
    origin : Literal["Customer","elsa"]
    Message : "str"


#Funcion to load css
def load_css():
    with open("static/styles.css", "r")  as f:
        css = f"<style>{f.read()} </style>"
        # st.write(css)
        st.markdown(css, unsafe_allow_html = True)


# save the embeddings in a DB that is persistent
# def manage_pdf() :
# loader = TextLoader("./facts.txt")

#Weblinks for Knowledge Base
# web_links = ["https://in.hotels.com/ho1068250336/four-points-by-sheraton-kochi-infopark-kakkanad-india",
#              "https://www.expedia.co.in/Kochi-Hotels-Four-Points-By-Sheraton-Kochi-Infopark.h33351573.Hotel-Information",
#              ] 


embeddings = CohereEmbeddings(model = "embed-english-v2.0")
print(" embedding docs !")

# creating a client to connect to qdrant
qdrant_client = QdrantClient(
    url = QDRANT_HOST,
    api_key= QDRANT_API_KEY,
)

# creating vector store
collection_name = "hotelDataCollection"
vector_store = Qdrant(
    client=qdrant_client,
    collection_name = collection_name,
    embeddings=embeddings
)
print("connection established !")

# vector_store = Qdrant.from_documents(texts, embeddings, location=":memory:",collection_name="summaries-po", distance_func="Dot")

#initializing Session State
def initialize_session_state() :

    # Initialize a session state to track whether the initial message has been sent
    if "initial_message_sent" not in st.session_state:
        st.session_state.initial_message_sent = False

    # Initialize a session state to store the input field value
    if "input_value" not in st.session_state:
        st.session_state.input_value = ""


    if "history" not in st.session_state:
        st.session_state.history = []

    if "chain" not in st.session_state :  

        #create custom prompt for your use case
        prompt_template = """
        You are a Hotel Receptionist at "Four Points by Sheraton" hotel.

         Here is the information about the hotel in JSON format:: 
        "name":"Four Points by Sheraton Kochi Infopark","address":"Four Points by Sheraton Kochi Infopark 
        Infopark Kochi Phase 1 Campus, Infopark P.O, Kakkanad, Kochi, Kerala, India, 682042","phone":"4847160000",
        "email":"sheratton@gmail.com","website":"https://www.marriott.com/en-us/hotels/COKFP-four-points-kochi-infopark/overview/",
        "description":"Welcome to the Marriott Hotel Discover a world of comfort and luxury at the Marriott Hotel, 
        where your every need is our top priority. Conveniently nestled in the heart of the city, our hotel offers a 
        perfect blend of modern elegance and timeless sophistication.  ,Exceptional Accommodations  ,
        Indulge in our spacious and beautifully appointed rooms and suites, designed to provide the 
        ultimate in relaxation and convenience. Whether you're traveling for business or leisure, you'll 
        find everything you need to make your stay unforgettable. Enjoy plush bedding, stunning city views, and a 
        range of modern amenities to enhance your comfort.  ,Exquisite Dining  ,Savor a culinary journey at our on-site 
        restaurants, where our talented chefs craft a diverse menu of delectable dishes. From international flavors to 
        local delicacies, there's something to satisfy every palate. Don't forget to explore our bar and lounge for 
        handcrafted cocktails and a vibrant atmosphere.  ,Unparalleled Services Our commitment to excellence extends
        to our services. Experience the warm hospitality of our dedicated staff, ready to assist you with any request. 
        From 24-hour room service to concierge assistance, we're here to ensure your stay is seamless and enjoyable.  
        World-Class Amenities, Stay active and energized with our fitness center, equipped with state-of-the-art equipment. For business travelers, our full-service business center provides all the tools you need to stay productive. Relax and unwind in our spa or take a dip in our refreshing pool.  ,Explore the City  ,Our central location puts you in the heart of the action. Explore nearby attractions, museums, shopping districts, and entertainment venues. Whether you're here for work or leisure, you'll find endless opportunities to make the most of your visit.  ,Book Your Stay  ,Discover why the Marriott Hotel is the preferred choice for discerning travelers. Experience luxury, convenience, and unmatched hospitality during your stay with us. Book your reservation today and embark on a memorable journey in our world of elegance and comfort.  ,","rooms":[["_id":"64ff544ce83c1395a075ef82","name":"Twin/Twin Deluxe Guest Room","description":"2 Twin/Single Beds, Guest Room","rate":["$numberDecimal":"7400"],"capacity":3,"amenities":["Air-conditioned  ,Non-smoking  ,Connecting rooms are available  ,Lighted makeup mirror  ,Hair dryer  ,Robe  ,Slippers  ,Chair, oversized  ,Alarm clock  ,Safe, in room  ,Desk, writing/work, electrical outlet  ,Iron and ironing board  ,Room service, 24-hour  ,Bottled water, complimentary  ,Coffee/tea maker  ,Instant hot water  ,Mini fridge  ,Phones  ,Phone features: speaker phone, and phone lines (2)  ,High-speed internet, complimentary  ,Wireless internet, complimentary  ,Newspaper delivered to room, on request  ,49in/124cm LED TV  ,Premium movie channels  ,Cable/satellite  ,International cable/satellite  ,Radio"],"__v":0],["_id":"64ff544ce83c1395a075ef83","name":"1-Bedroom Suite with Executive Lounge Access","description":"1 King Bed, Executive Lounge Access, Shower Only, High Floor","rate":["$numberDecimal":"11600"],"capacity":3,"amenities":["Air-conditioned  ,Non-smoking  ,Connecting rooms are available  ,Lighted makeup mirror  ,Hair dryer  ,Robe  ,Slippers  ,Sofa  ,Chair  ,Alarm clock  ,Safe, in room  ,Desk, writing/work, electrical outlet  ,Iron and ironing board  ,Room service, 24-hour  ,Bottled water, complimentary  ,Coffee/tea maker  ,Instant hot water  ,Mini fridge  ,Phones  ,Phone features: speaker phone, and phone lines (2)  ,High-speed internet, complimentary  ,Wireless internet, complimentary  ,Newspaper delivered to room, on request  ,2 TVs  ,Premium movie channels  ,Cable/satellite  ,International cable/satellite  ,Radio"],"__v":0],["_id":"64ff544ce83c1395a075ef84","name":"1-Bedroom Suite with Executive Lounge Access, Shower and Tub Combination","description":"1 King Bed, Executive Lounge Access, Shower and Tub Combination, High Floor","rate":["$numberDecimal":"12600"],"capacity":3,"amenities":["Air-conditioned  ,Non-smoking  ,Connecting rooms are available  ,Lighted makeup mirror  ,Hair dryer  ,Robe  ,Slippers  ,Sofa  ,Chair, oversized  ,Alarm clock  ,Safe, in room  ,Desk, writing/work, electrical outlet  ,Iron and ironing board  ,Room service, 24-hour  ,Bottled water, complimentary  ,Coffee/tea maker  ,Instant hot water  ,Mini fridge  ,Phones  ,Phone features: speaker phone, and phone lines (2)  ,High-speed internet, complimentary  ,Wireless internet, complimentary  ,Newspaper delivered to room, on request  ,2 TVs  ,Premium movie channels  ,Cable/satellite  ,International cable/satellite  ,Radio"],"__v":0]],"ammenities":[],"nearbyAttractions":


        You will be given a context of the conversation made so far followed by a customer's question, 
        give the answer to the question using the context. 
        The answer should be short, straight and to the point.
        If you don't know the answer, reply that the answer is not available.
        Never Hallucinate.
        
        Context: {context}

        Question: {question}
        Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        chain_type_kwargs = { "prompt" : PROMPT }
        llm = Cohere(model = "command-light", temperature=0.5)

        #build your chain for RAG+C
        st.session_state.chain = ConversationalRetrievalChain.from_llm(     
            llm = llm,
            chain_type = "stuff",
            memory = ConversationSummaryMemory(llm = llm, memory_key='chat_history', input_key='question', output_key= 'answer', return_messages=True),
            retriever = vector_store.as_retriever(),
           return_source_documents=False,
           combine_docs_chain_kwargs=chain_type_kwargs,
           
        )
        print(st.session_state.chain.memory.buffer)

#Callblack function which when activated calls all the other
#functions 
def on_click_callback():

    load_css()
    # print(st.session_state.customer_prompt)
    customer_prompt = st.session_state.customer_prompt

    if customer_prompt:
        
        st.session_state.input_value = ""
        st.session_state.initial_message_sent = True

        with st.spinner('Generating response...'):

            llm_response = st.session_state.chain(
                {"question": customer_prompt,"summaries": st.session_state.chain.memory.buffer}, return_only_outputs=True)
            
           
            # answer = llm_response["answer"]

    st.session_state.history.append(
        Message("customer", customer_prompt)
    )
    st.session_state.history.append(
        Message("AI", llm_response)
    )



initialize_session_state()
chat_placeholder = st.container()
prompt_placeholder = st.form("chat-form")
creditcard_placeholder = st.empty()

# query = st.chat_input()
# if query!= None :
#     st.session_state.customer_prompt = query
#     # st.write(st.session_state.customer_prompt)
#     result = st.session_state.chain(query)
#     st.write(print_result(result))
#     # st.write(result)
#     # display(Markdown(print_result(result)))

with chat_placeholder:
    for chat in st.session_state.history:
        if type(chat.Message) is dict:
            msg = chat.Message['answer']
        else:
            msg = chat.Message 
        div = f"""
        <div class = "chatRow 
        {'' if chat.origin == 'AI' else 'rowReverse'}">
            <img class="chatIcon" src = "app/static/{'elsa.png' if chat.origin == 'AI' else 'admin.png'}" width=32 height=32>
            <div class = "chatBubble {'adminBubble' if chat.origin == 'AI' else 'humanBubble'}">&#8203; {msg}</div>
        </div>"""
        st.markdown(div, unsafe_allow_html=True)


# Streamlit UI Input field
with st.form(key="chat_form"):
    cols = st.columns((6, 1))
    
    # Display the initial message if it hasn't been sent yet
    if not st.session_state.initial_message_sent:
        cols[0].text_input(
            "Chat",
            placeholder="Hello, how can I assist you?",
            label_visibility="collapsed",
            key="customer_prompt",
        )  
    else:
        cols[0].text_input(
            "Chat",
            value=st.session_state.input_value,
            label_visibility="collapsed",
            key="customer_prompt",
        )

    cols[1].form_submit_button(
        "Ask",
        type="secondary",
        on_click=on_click_callback,
    )

# Update the session state variable when the input field changes
st.session_state.input_value = cols[0].text_input