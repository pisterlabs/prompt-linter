from langchain.document_loaders import YoutubeLoader
import streamlit as st
import openai
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

openai.api_key = ''

def generate_summary(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=500,  # Adjust the value as per your requirements
        n=1,
        stop=None,
        temperature=0.7,  # Adjust the temperature as per your preference
    )
    summary = response.choices[0].text.strip()
    return summary

def generate_translation(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=500,  # Adjust the value as per your requirements
        n=1,
        stop=None,
        temperature=0.7,  # Adjust the temperature as per your preference
    )
    translation = response.choices[0].text.strip()
    return translation

def main():
    st.title("YouTube to Text Summary")

    # Home page
    st.subheader("Enter YouTube video URL")
    user_input = st.text_input("")

    if st.checkbox("Get Transcript"):
        if user_input:
            # Load transcript from YouTube
            loader = YoutubeLoader.from_youtube_url(user_input, add_video_info=True)
            transcript = loader.load()
            document = transcript[0]

            if document:
                page_content = document.page_content

                # Display transcript details
                st.subheader("Transcript Details")
                st.write("Title:", document.metadata['title'])
                st.write("Author:", document.metadata['author'])
                st.write("Content:", page_content)

    elif st.checkbox("Get Summary"):
        if user_input:
            # Load transcript from YouTube
            loader = YoutubeLoader.from_youtube_url(user_input, add_video_info=True)
            transcript = loader.load()
            document = transcript[0]

            if document:
                page_content = document.page_content

                def preprocess_text(text):
                    # Tokenization
                    tokens = word_tokenize(text)
                    
                    # Remove stop words
                    stop_words = set(stopwords.words('english'))
                    tokens = [token for token in tokens if token.lower() not in stop_words]
                    
                    # Lemmatization
                    lemmatizer = WordNetLemmatizer()
                    tokens = [lemmatizer.lemmatize(token) for token in tokens]
                    
                    # Remove punctuations and unwanted spaces
                    tokens = [token for token in tokens if token not in string.punctuation]
                    tokens = [token.strip() for token in tokens if token.strip()]
                    
                    # Join tokens with space
                    preprocessed_text = ' '.join(tokens)
                    
                    return preprocessed_text

                page_content = preprocess_text(page_content)

                # Generate summary
                summary_prompt = f"You are the best summarizer. Summarize this:\n\n{page_content}\n\nBriefly and concisely."
                summary = generate_summary(summary_prompt)

                # Display summary
                st.subheader("Summary")
                st.write(summary)

                # Translation
                st.subheader("Translation")
                language_opted = st.text_input("Enter the name of the language for translation")
                tt=st.checkbox("Get Translation")
                # language_opted = st.text_input("Enter the name of the language for translation")
                if language_opted:
                    # Generate translation prompt
                    translation_prompt = f"You are an excellent language translator. Translate this:\n\n{summary}\n\nTo:\n\n{language_opted}"
                    translation = generate_translation(translation_prompt)
                    st.write(translation)

if __name__ == '__main__':
    main()
