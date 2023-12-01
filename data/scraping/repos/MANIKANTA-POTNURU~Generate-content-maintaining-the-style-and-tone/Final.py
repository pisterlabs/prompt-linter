import streamlit as st
import cohere as ch
import time 
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi
import wikipedia
import requests 
from functions import convert_pdf_to_txt_file, displayPDF

def main():
    st.title("Generate content maintaining the style and tone")
    st.sidebar.title("Choose an Option")
    task = st.sidebar.radio("Select a task:", ["Using AI","From YouTube & AI", "From Wikipedia & AI","Chat with pdf using AI","Tone&Style Checker"])
    if task == "From YouTube & AI":
        st.title("Enter the youtube video Url")
        youtube_url = st.text_input("Enter YouTube video URL:")
        if youtube_url:
            video_id = youtube_url.split("=")[-1]
            if st.button("Summarize"):
                summarized_text = summarize_youtube(video_id)
                st.subheader("Summarized Transcript:")
                st.write(summarized_text)
                not_satisfied_checkbox = st.checkbox("Not Satisfied")
                if not_satisfied_checkbox:
                    not_satisfied(summarized_text)
                time.sleep(30000)
                

    elif task == "From Wikipedia & AI":
        st.title("Input The Text")
        article_title = st.text_input("Enter Wikipedia article title:")
        if article_title:
            if st.button("Summarize"):
                summarized_text =summarize_script(summarize_wikipedia(article_title))
                st.subheader("Summarized Article:")
                st.write(summarized_text)
                not_satisfied_checkbox = st.checkbox("Not Satisfied")
                if not_satisfied_checkbox:
                    not_satisfied(summarized_text)
                time.sleep(30000)
                
    elif task=="Using AI":
        st.title("Input The Prompt")
        user_input = st.text_input("Enter something:")
        if st.button("Submit"):
            output1 = generate_script(user_input)
            st.write(output1)
            not_satisfied_checkbox = st.checkbox("Not Satisfied")
            if not_satisfied_checkbox:
                not_satisfied(output1)
            time.sleep(30000)

    elif task=="Chat with pdf using AI":
        pdf_file = st.file_uploader("Load your PDF", type=['pdf'])
        if pdf_file:
            path = pdf_file.read()
            file_extension = pdf_file.name.split(".")[-1]
            if file_extension == "pdf":
                # display document
                textarea=st.text_input("Enter some more text:")
                with st.expander("Display document"):
                    displayPDF(path)
                
                text_data_f, nbPages = convert_pdf_to_txt_file(textarea,pdf_file)
                totalPages = "Pages: "+str(nbPages)+" in total"
                st.info(totalPages)
                st.download_button("Download txt file", text_data_f)
    elif task=="Tone&Style Checker":
        st.title("Text Tone Checker")
        tetx= st.text_input("Enter something:")
        st.write(tone_reco(tetx))

def not_satisfied(output1):
    user_input1 = st.text_input("Enter some more text:")
    tone=tone_reco(output1)
    context = summarize_script(output1)
    prompt1 =  ' Follow this '+tone+'tone for this prompt regarding' + user_input1 +'based on this '+context #Prompt If the user is not satisfied or if user want some more 
    if prompt1!="":
        
        st.write(generate_script(prompt1))

def tone_reco(tetx):
    
    response = requests.post(
        "https://api.sapling.ai/api/v1/tone",  # SaplingAi
        json={
            "key": "SAPLING_APIKEY",#Use SaplingAi Api For Recongnizing The tone and Style For an text
            "text": tetx
        }
    )
    data = response.json()

    overall_tones = data.get('overall', [])
    results = data.get('results', [])

    tone_results = overall_tones + (results[0] if results else [])
    tones_formatted = [f"{tone[1]} ({tone[2]})" for tone in tone_results]

    output_sentence = ', '.join(tones_formatted)
    return output_sentence



def generate_script(user_input):
    
    co = ch.Client('COHERE_APIKEY') # This is your trial API key (Paste the cohere API KEY)
    response = co.generate(
        model='command',
        prompt=user_input,
        max_tokens=300,
        temperature=0.9,
        k=0,
        stop_sequences=[],
        return_likelihoods='NONE')
    return response.generations[0].text

def summarize_script(output1):
    co = ch.Client('COHERE_APIKEY') # This is your trial API key(Paste the Api Key)
    response = co.summarize( 
        text=output1,
        length='auto',
        format='auto',
        model='summarize-xlarge',
        additional_command='',
        temperature=0.3,
    ) 
    return response.summary
def summarize_youtube(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    result = ""
    for i in transcript:
        result += ' ' + i['text']

    summarizer = pipeline('summarization')
    summarized_text = summarizer(result)
    return summarized_text[0]['summary_text']
def summarize_wikipedia(article_title):
    content = wikipedia.page(article_title).content
    summarizer = pipeline('summarization')
    max_length = 1024  # Adjust this value as needed
    chunks = [content[i:i + max_length] for i in range(0, len(content), max_length)]

    summarized_text = ""
    for chunk in chunks:
        summarized_chunk = summarizer(chunk)
        summarized_text += summarized_chunk[0]['summary_text'] + " "
    
    return summarized_text

if __name__ == "__main__":
    main()
