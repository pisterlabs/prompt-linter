import os
import openai
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 
import streamlit as st
from PIL import Image

os.environ['OPENAI_API_KEY']='Your OpenAI key'

img=Image.open("C:\\Create.ai\\Page_Icon.png")
st.set_page_config(page_title="Create.ai: Generate content with AI",page_icon=img)
st.title('Create:violet[.]ai 📷')
tab1,tab2=st.tabs(['Home','Create'])

with tab1:

    st.write('Create:blue[.]ai is an AI-powered content creation tool that can help you level up your YouTube channel. With Create.ai, you can generate high-quality content in minutes, including titles, descriptions, scripts, and even entire videos.Whether you\'re a beginner or a seasoned YouTuber, Create.ai can help you take your channel to the next level.')
    st.image('https://www.apa.org//images//2021-09-career-content-creation_tcm7-296397.jpg')

    st.write('If you\'re looking for a way to create engaging and informative YouTube videos quickly and easily, then Create.ai is the perfect tool for you. :violet[Sign up] for a free trial today and see how Create:violet[.]ai can help you grow your channel.')
    
    st.write('Here are some of the benefits of using Create:violet[.]ai:')
    
    st.success('''

 Save time and effort: Create.ai can help you generate content quickly and easily, so you can focus on other aspects of your YouTube channel.

Improve your content quality: Create.ai uses AI to understand your audience and create content that is both engaging and informative.

Stand out from the competition: Create.ai can help you create unique and original content that will help you stand out from the competition.

''')
    
with tab2:
    st.write('Try Create:violet[.]ai today and see how it can help you grow your channel.')
    st.image('https://assets.entrepreneur.com/content/3x2/2000/1629828633-GettyImages-1212772310.jpg')
    prompt=st.text_input('What are you looking to create?',placeholder='Enter a prompt here')
    
    # Prompt templates
    title_template = PromptTemplate(
        input_variables = ['topic'], 
        template='write me a youtube video title about {topic}'
    )
    script_template = PromptTemplate(
        input_variables = ['title','wikipedia_research'], 
        template='write me a youtube video script based on this title: {title} while leveraging this wikipedia research {wikipedia_research}'
        
    )
    description_template=PromptTemplate(
        input_variables=['script'],
        template='Write me a description for youtube video in three lines based on this content:{script}'
    )
    hashtags_template=PromptTemplate(
        input_variables=['script'],
        template='write me five best hashtags for youtube video based on this content:{script}'
    )
    thumbnail_tempalte=PromptTemplate(
        input_variables=['title'],
        template='write me an eye-catching text on thumbnail for youtube video on this title: {title}'
    )
        
    # Memory 
    title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
    script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')
    description_memory = ConversationBufferMemory(input_key='script', memory_key='chat_history')
    hashtags_memory = ConversationBufferMemory(input_key='script', memory_key='chat_history')
    thumbnail_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')
        
    # Llms
    llm = OpenAI(temperature=0.9) 
    title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
    script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)
    description_chain = LLMChain(llm=llm, prompt=description_template, verbose=True, output_key='description', memory=description_memory)
    hashtags_chain = LLMChain(llm=llm, prompt=hashtags_template, verbose=True, output_key='hashtags', memory=hashtags_memory)
    thumbnail_chain = LLMChain(llm=llm, prompt=thumbnail_tempalte, verbose=True, output_key='thumbnail', memory=thumbnail_memory)
    
    wiki = WikipediaAPIWrapper()

    if prompt: 
        title = title_chain.run(prompt)
        wiki_research = wiki.run(prompt)
        
        script = script_chain.run(title=title, wikipedia_research=wiki_research)
        description = description_chain.run(script=script)
        hashtags=hashtags_chain.run(script=script)
        thumbnail=thumbnail_chain.run(title)
        
        
        with st.expander('Title'):
            st.info(title)
        with st.expander('Script'):
            st.info(script)
        with st.expander('Description'):
            st.info(description)
        with st.expander('Hashtags'):
            st.info(hashtags)
        with st.expander('Thumbnail'):
            st.info(thumbnail)
        with st.expander('Wikipedia Research'): 
            st.info(wiki_research)
   
            
            
