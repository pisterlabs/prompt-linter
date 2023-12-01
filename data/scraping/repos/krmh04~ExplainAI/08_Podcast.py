import asyncio
import os
from deepgram import Deepgram
import feedparser
import openai
import streamlit as st

async def main():
    st.markdown(""" 
    <style>
    .big-font {
    font-size:30px !important;
    font-weight:bold !important;
    }
    .small-font{
    font-size:16px !important;
    font-weight:bold !important;    
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<p class="big-font">Podcast Audio Transcription and Summarization</p>',unsafe_allow_html=True)
    st.markdown('<p class="small-font">You can get the RSS feed of a podcast by entering the Podcast URL here::<a>https://getrssfeed.com/</a></p>',unsafe_allow_html=True)
    openai.api_key = os.getenv('OPENAI_API_KEY')
    DEEPGRAM_API_KEY = os.getenv('DEEPGRAM_API_KEY')
    rssfeedurl=st.text_input("Please input the RSS feed.")
    deepgram = Deepgram(DEEPGRAM_API_KEY)
    rss = feedparser.parse(rssfeedurl)
    #Create a list of the fetched entries from the rss feed.
    entry_titles = [rss.entries[i].title for i in range(len(rss.entries))]
    selected_entry = st.selectbox('Select the episode', entry_titles)
    #Create a dropdown of the fetched entries.
    if selected_entry is not None:
        selected_index = entry_titles.index(selected_entry)
        st.session_state["selected_index"] = selected_index
        
        if st.button("Submit"):
            
            url = rss.entries[selected_index].enclosures[0].href
            source = { 'url': url }

            #Transcribe the content
            transcription_options = { 'punctuate': True, 'diarize': True, 'paragraphs': True }
            response =await deepgram.transcription.prerecorded(source, transcription_options)

            transcript = response['results']['channels'][0]['alternatives'][0]['paragraphs']['transcript']
            st.download_button("Download transcript", transcript)

            with st.expander("Expand to see transcript"):
                st.text_area('Transcript', transcript)

            
            # Summarize transcription using OpenAI GPT API
            with st.spinner("Summarising transcript..."):
                summary = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                            {
                                "role": "system", 
                                "content": (
                                    "You are a helpful assistant that"
                                    " summarises audio/podcast transcripts into 200 words or less."
                                    )
                            },
                            {"role": "user", "content": transcript}
                        ]
                )

            # Display summary
            st.write("Summary:", summary.choices[0]["message"]["content"])
if __name__ == '__main__':
    # LOGGED_IN key is defined by streamlit_login_auth_ui in the session state.
    if 'LOGGED_IN' in st.session_state and st.session_state.LOGGED_IN:
        asyncio.run(main())
    else:
        st.write("Please login first")