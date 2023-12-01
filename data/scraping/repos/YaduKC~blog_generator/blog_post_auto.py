from ast import arg
from random import choice, random
import streamlit as st
import openai
from decouple import config
import requests

openai.api_key = st.secrets["OPENAI_KEY"]
UNSPLASH_KEY = st.secrets["UNSPLASH_KEY"]

if 'blog_data_' not in st.session_state:
    st.session_state.blog_data_ = {}

def unsplash_image(tag="", orientation="landscape"):
    url_list = []
    url_dict = {}

    response = openai.Completion.create(
                    engine="text-davinci-002",
                    prompt="Extract keywords from this text:\n\n" + tag,
                    temperature=0.3,
                    max_tokens=60,
                    top_p=1.0,
                    frequency_penalty=0.8,
                    presence_penalty=0.0,
                    )
    key = response.choices[0].get("text")
    topics = list(set(key.split()))
    for topic in topics:
        if len(topic)>3:
            topic=topic.replace(",","")
            images = requests.get("https://api.unsplash.com/search/photos?query={}&page=1&client_id={}&orientation={}".format(topic, UNSPLASH_KEY, orientation))
            try:
                data = images.json()
                result = data.get("results")
            except:
                continue
            if len(result) > 0:
                for u in result:
                    if orientation=='landscape':
                        url_dict['main_image'] = u["urls"]["regular"]
                        orientation = 'squarish'
                    else:
                        url_list.append(u["urls"]["regular"])
    url_dict['section_images'] = url_list
    return url_dict

def extract_blog_summary(reference_blog):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt="Summarize the text given below\n {}".format(reference_blog),
        temperature=0.7,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )
    return response['choices'][0]['text']

def extract_blog_title(summary):
    #print(summary)
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt="{} \nWrite a short creative blog title using the text given above.\n ".format(summary),
        temperature=0.7,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )
    return response['choices'][0]['text']

def generateBlogSections(prompt1):
    response = openai.Completion.create(
      engine="text-davinci-002",
      prompt="Expand the blog title into 5 high level blog sections: {} \n\n- Introduction: ".format(prompt1),
      temperature=1.0,
      max_tokens=100,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    return response['choices'][0]['text']

def blogIntro(title, c_name):
    response = openai.Completion.create(
    engine="text-davinci-002",
    prompt="Write a long and detailed blog introduction for the comany '{}' for the blog topic '{}'".format(c_name, title),
    temperature=0.7,
    max_tokens=200,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    intro = response['choices'][0]['text']
    return intro

def blogSectionExpander(heading):
    response = openai.Completion.create(
    engine="text-davinci-002",
    prompt="{}\nWrite three detailed professional paragraphs on the topic given above for a blog post.".format(heading),
    temperature=0.7,
    max_tokens=200,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    section = response['choices'][0]['text']
    return section

def extract_info(c_name, reference_blog):
    para_list = []
    para = ""
    for i in reference_blog:
        para += i
        if len(para) > 2000:
            para_list.append(para)
            para = ""
    summary = extract_blog_summary(para_list[0])
    title = extract_blog_title(summary)
    sections = generateBlogSections(title)
    sections = sections.split('\n')
    sections = [{'header':i, 'content':''} for i in sections if i]
    st.session_state.blog_data_['summary'] = summary
    st.session_state.blog_data_['title'] = title
    st.session_state.blog_data_['intro'] = blogIntro(title, c_name)
    section_list = []
    images = unsplash_image(tag=st.session_state.blog_data_['summary'])
    for section in sections:
        if section['header'][0] == '-':
            section['header'] = section['header'][1:].strip()
        section['content'] = blogSectionExpander(section['header'])
        section['image'] = choice(images['section_images'])
        section_list.append(section)
    st.session_state.blog_data_['sections'] = section_list
    st.session_state.blog_data_['main_image'] = images['main_image']

def load_blog():
    from blog_preset import preset
    st.session_state.blog_data_ = preset

if __name__ == "__main__":
    st.set_page_config(layout='wide')
    column = [1,3,1]
    with st.container():
        cols = st.columns(column)
        cols[1].title("Blog Post Generation(Reference Guided)")
        cols[1].write("---")
        c_name = cols[1].text_input('Company Name')
        reference_blog = cols[1].text_area('Reference Blog', height=500)
        cols[1].button("Submit", on_click=extract_info, args=(c_name, reference_blog,))
        cols[1].button("Load Generated Blog", on_click=load_blog)
        cols[1].write("---")

    if 'title' in st.session_state.blog_data_:
        with st.container():
            cols = st.columns(column)
            cols[1].title(st.session_state.blog_data_['title'])
            cols[1].write("---")
            cols[1].image(st.session_state.blog_data_['main_image'])
            cols[1].write("---")
            cols[1].header("Introduction")
            cols[1].write(st.session_state.blog_data_['intro'])
            cols[1].write("---")

            # cols[1].write(st.session_state.blog_data_)
            # Display sections
        section_placement = ['left', 'right', 'none']
        for section in st.session_state.blog_data_['sections']:
            selected_placement = choice(section_placement)
            with st.container():
                cols = st.columns(column)
                cols[1].header(section['header'])
            
            if selected_placement == 'left':
                with st.container():
                    cols = st.columns([1,1,2,1])
                    cols[2].write(section['content'])
                    cols[1].image(section['image'])

            if selected_placement == 'right':
                with st.container():
                    cols = st.columns([1,2,1,1])
                    cols[1].write(section['content'])
                    cols[2].image(section['image'])

            if selected_placement == 'none':
                with st.container():
                    cols = st.columns(column)
                    cols[1].write(section['content'])
            

        with st.container():
            cols = st.columns(column)
            cols[1].write("---")
