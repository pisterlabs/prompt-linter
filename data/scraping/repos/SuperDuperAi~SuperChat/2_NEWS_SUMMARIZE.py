import time
import streamlit as st
from langchain.document_loaders import NewsURLLoader

from runtime import model

st.title("SuperChat with NEWS (any articles)")
st.markdown(
    "**Chat with Claude v2 on Bedrock (100k context)")

if 'doc_news' not in st.session_state:
    st.session_state['doc_news'] = ""


url = ''
if st.experimental_get_query_params():
    url = st.experimental_get_query_params()['url'][0]

input_url = st.text_input("Enter a URL:", value=url)


if "messages" not in st.session_state:
    st.session_state.messages = []

if input_url and st.session_state['doc_news'] == "":
    with st.spinner('Processing'):
        loader = NewsURLLoader(urls=[input_url])
        docs = loader.load()

        page_content = str(docs[0].page_content)
        metadata = docs[0].metadata

        # Define prompt
        prompt_template = f"""
        I'm going to give you a document from web url (news or article, post blog).
        
        Generate a summarized and episodic narrative from the following text. 
        Divide the summary into three acts, 10 slides. 
        Identify key theses, important figures, and locations. 
        Make sure each episode fits into a 5-10 second slide for short-form videos like shorts, reels, or TikTok. 
        Conclude with a main takeaway. 
        Retain the essence of the original text.

        Here is the document:
        <document>
        Title: {metadata['title']}
        Language: {metadata['language']}
        Description: {metadata['description']}
        Publish Date: {metadata['publish_date']}

        Page Content: {page_content}
        </document>
               
        Thus, the format of your overall response should look like example what's shown between the <example></example> tags.  
        For generating a script that dynamically adapts to the length of the input text,The aim would be to maintain the integrity of the essential points while condensing information if the text is too long.
        Make sure to follow the formatting and spacing exactly. 
        
        <example>
        # title
        ## subtitle
        
        ### Summary:  
                
        ---
        ### Scenes (describe the scenes in the video):
        [including, if possible, descriptions, quotes and characters]
        
        ---
        ### Analysis:
        (1) Identify the main themes and problems discussed.
        (2) List interesting theses and quotes.
        (3) Identify the main characters.
        (4) Suggest tags for linking with articles.
        (5) Sentiment Analysis. Is the sentiment expressed in the text positive, negative, or neutral? Please provide evidence from the text to support your assessment. Additionally, on a scale of -1 to 1, where -1 is extremely negative, 0 is neutral, and 1 is extremely positive, rate the sentiment of the text.
        (6) Political Orientation Analysis. 
        (7) Fake news detection or manipulation, critical thinking.
        
        ### Questions:
        Q1:
        Q2:
        Q3:
        
        [Provide three follow-up questions worded as if I'm asking you. 
        Format in bold as Q1, Q2, and Q3. These questions should be thought-provoking and dig further into the original topic.]

        </example>
        
        Answer the question immediately without preamble.
        
        """

        # st.info(prompt_template)
        with st.chat_message("assistant"):
            # st.info(prompt_template)

            st.warning(f"Page len: {len(page_content)}")
            st.experimental_set_query_params = {'url': input_url}

        news_summarise = model.predict(input=prompt_template)
        st.session_state['doc_news'] = news_summarise
        st.session_state.messages.append({"role": "assistant", "content": news_summarise})


        # with st.chat_message("assistant"):
        #     st.markdown(news_summarise)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize session state for chat input if it doesn't already exist

prompt_disabled = (st.session_state['doc_news'] == "")

if prompt := st.chat_input("What is up?", disabled=prompt_disabled):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        processed_prompt = prompt

        result = model.predict(input=prompt)

        for chunk in result:
            full_response += chunk
            time.sleep(0.01)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(result)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
