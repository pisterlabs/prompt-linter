import streamlit as st
import openai
import os
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from apikey import apikey
from prompts import spike_template, quentin_template, wes_template

def generate_image(image_description):
  img_response = openai.Image.create(
    prompt = image_description,
    n=1,
    size="512x512")
  img_url = img_response['data'][0]['url']
  return img_url

def main():
    os.environ['OPENAI_API_KEY'] = apikey
    st.set_page_config(page_title="Movie Concept Generation")
    st.title("AI Movie Concept Generation")
    st.subheader("Powered by OpenAI, LangChain, Streamlit")

    director = st.selectbox(
        label="AI Director", 
        options=(
            "Spike Lee",
            "Quentin Tarrentino",
            "Wes Anderson",
        )
    )

    if director=="Spike Lee":
        CONCEPT_TEMPLATE=PromptTemplate(
            input_variables=['user_input'],
            template=spike_template)
    elif director=="Quentin Tarrentino":
        CONCEPT_TEMPLATE=PromptTemplate(
            input_variables=['user_input'],
            template=quentin_template)
    elif director=="Wes Anderson":
        CONCEPT_TEMPLATE=PromptTemplate(
            input_variables=['user_input'],
            template=wes_template)
        
    ImageGenTemplate = PromptTemplate(
        input_variables=['concept'],
        template='''
            From this title, subtitle, and movie concept, generate an prompt for a relevant poster image utilizing the DALLE2 image generation.
            Keep your response to at most 2 sentences, this is very important that it is no longer than 25 words. 
            That visually encapsulates the title and story based on the movie concept
            MOVIE CONCEPT: {concept}
        '''
    )

    user_input = st.text_input("Enter Prompt:")

    generate_button = st.button("Generate")
    if generate_button and user_input:
        with st.spinner('Generating...'):
            try:
                concept_memory = ConversationBufferMemory(input_key='user_input', memory_key='chat_history')
                llm = OpenAI(temperature=0.9)
                concept_chain = LLMChain(llm=llm, prompt=CONCEPT_TEMPLATE, verbose=True, memory=concept_memory)
                imageprompt_chain = LLMChain(llm=llm, prompt=CONCEPT_TEMPLATE, verbose=True, memory=concept_memory)
                concept_response = concept_chain.run(user_input)
                imageprompt_response = f"In {director} Movie Poster Style and with no words: " + imageprompt_chain.run(concept_response)
                generated_img = generate_image(imageprompt_response)
                st.image(generated_img)
                st.write(concept_response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


if __name__ == '__main__':
    main()
