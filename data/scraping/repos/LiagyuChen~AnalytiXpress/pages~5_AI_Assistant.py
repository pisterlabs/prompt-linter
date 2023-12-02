import streamlit as st
from pandasai import SmartDataframe
from pandasai.llm import OpenAI

st.set_page_config(page_title = "AnalytiXpress", page_icon = "./assets/logo.png")


st.markdown("# AI Assistant")
st.sidebar.header("AI Assistant")


# Get the data from session_state
if 'df' in st.session_state:
    df = st.session_state.df
    st.header("Current Dataframe: ")
    st.write(df)
    
    setted = False
    
    # Get the Open-AI API
    if "apiKey" not in st.session_state or "llm" not in st.session_state:
        apiKey = st.sidebar.text_input("Enter your OpenAI API Key:")
        if apiKey != None and len(apiKey) == 52:
            llm = OpenAI(api_token = apiKey)
            st.session_state.apiKey = apiKey
            st.session_state.llm = llm
            setted = True
        else:
            st.sidebar.warning("Invalid, please re-enter a valid API key")
    else:
        apiKey = st.session_state.apiKey
        llm = st.session_state.llm
        setted = True
        
    
    if setted == True:
        # Convert the dataframe into smart dataframe
        df = SmartDataframe(df, config = {"llm": llm})

        # Initialize chat history
        if "chatHistory" not in st.session_state:
            st.session_state.chatHistory = []

        # Display chat history
        for message in st.session_state.chatHistory:
            if message['sender'] == 'user':
                st.chat_message(message['content'], 'You', 'right')
            else:
                st.chat_message(message['content'], 'AI', 'left')

        # User input
        userInput = st.chat_input(placeholder = "Enter yourr prompt")

        # When user sends a message
        if userInput:
            st.session_state.chatHistory.append({'sender': 'user', 'content': userInput})
            with st.spinner("Please wait while the AI processes your request..."):
                try:
                    response = df.chat(userInput)
                except Exception as e:
                    st.warning(f"An error occurred: {e}")
                    st.warning("Please adjust your prompt.")
                st.session_state.chatHistory.append({'sender': 'ai', 'content': str(response)})
            
# Show Warning
else:
    st.warning("Please upload and edit data first!")
    

