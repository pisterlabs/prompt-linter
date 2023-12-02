import streamlit as st
from langchain.chat_models import ChatOpenAI
from run_chains import get_args, find_flights
from search_flights import pull_flights
from utils import SingletonToken, query_template

st.markdown(
    """
    #### Prototype Built by [Data-Centric Solutions](https://www.data-centric-solutions.com/)
    """,
    unsafe_allow_html=True,
)

# Side panel for OpenAI token input
st.sidebar.title("Configuration")
OPENAI_KEY = st.sidebar.text_input("Enter OpenAI Key", type="password")

# Initialize an empty placeholder
placeholder = st.empty()

if OPENAI_KEY:
    SingletonToken.set_token(OPENAI_KEY)
    OPENAI_KEY = SingletonToken.get_token()
    
    # If OpenAI key and data_url are set, enable the chat interface
    st.title("Find my flights🛫 ")
    query_user = placeholder.text_input("Search for flights...")
    
    if st.button("Submit"):
        try:
            num_adults, departureDate, returnDate, destinationLocationCode, originLocationCode, TypeofflightReuqest = get_args(query_user, OPENAI_KEY)
        except Exception:
            st.write("Please make sure you tell us the origin, destination, departure and return dates, and number of adults") 
        
        db = pull_flights(originLocationCode, destinationLocationCode, departureDate, returnDate, num_adults)
        llm = ChatOpenAI(temperature=0, model="gpt-4-0613", openai_api_key=OPENAI_KEY)
        query = query_template(num_adults, departureDate, returnDate, destinationLocationCode, originLocationCode, TypeofflightReuqest)
        response = find_flights(query, llm, db)
        st.markdown(f"Here's your suggested Journey: : {response}")

else:
    # If OpenAI key and data_url are not set, show a message
    placeholder.markdown(
        """
        **Please enter your OpenAI key and data URL in the sidebar.**
        
        Follow this [link](https://www.howtogeek.com/885918/how-to-get-an-openai-api-key/) to get your OpenAI API key.
        """,
        unsafe_allow_html=True,
    )