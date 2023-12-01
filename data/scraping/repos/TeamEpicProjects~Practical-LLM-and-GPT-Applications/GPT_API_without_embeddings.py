import streamlit as st
import os
import openai

openai.api_key = "Enter your OpenAI API Key here."

if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

# Sidebar Design
with st.sidebar:
    response=st.radio("Please choose an operation..",('Void',
                                                      'General Queries',
                                                      'Grammer and Spell Check',
                                                      'Summarize Text',
                                                      'Q&A',
                                                      'Language Translation',
                                                      'Language Detection',
                                                      'Detect and Translate',
                                                      'Code Explanation',
                                                      'Generate SQL Queries',
                                                      'Programming Language Conversion',
                                                      'Sentiment Analysis',
                                                      'Extract Keywords',
                                                      'Text Generator from keywords',
                                                      'Essay Outline Generator',
                                                      'Essay Generator'))
    match response:
        case 'Void':
            st.write('You have not selected any operation yet!!!')
        case 'General Queries':
            st.write('You have selected general queries.')
        case 'Grammer and Spell Check':
            st.write('You have selected grammer and spell check.')
        case 'Summarize Text':
            st.write('You have selected for summarizing text.')
        case 'Q&A':
            st.write('You have selected for questionnaire.')
        case 'Language Translation':
            st.write('You have selected language translation.')
        case 'Language Detection':
            st.write('You have selected language detection.')
        case 'Detect and Translate':
            st.write('You have selected for language detection and translation.')
        case 'Code Explanation':
            st.write('You have selected for code explanation.')
        case 'Generate SQL Queries':
            st.write('You have selected for generating SQL queries.')
        case 'Programming Language Conversion':
            st.write('You have selected for converting a code snippet to another programming language.')
        case 'Sentiment Analysis':
            st.write('You have selected for sentiment analysis.')
        case 'Extract Keywords':
            st.write('You have selected for extracting keywords from text.')
        case 'Text Generator from keywords':
            st.write('You have selected for generating text from keywords')
        case 'Essay Outline Generator':
            st.write('You have selected for generating outline for an essay.')
        case 'Essay Generator':
            st.write('You have selected for generating an essay.')

def general(text):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=text,
        temperature=0,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response['choices'][0]['text'].strip()

def grammer(text):
    response = openai.Completion.create(
        model="text-davinci-003",\
        prompt="Correct this to standard English:"+text,
        temperature=0,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response['choices'][0]['text'].strip()

def summary(text):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="Summarize this for a second-grade student:"+text,
        temperature=0.01,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response['choices'][0]['text'].strip()

def questionnaire(question):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="Answer the question: "+question,
        temperature=0,
        max_tokens=140,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response['choices'][0]['text'].strip()

def translation(target, text):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="Translate "+text+" to "+target,
        temperature=0,
        max_tokens=140,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response['choices'][0]['text'].strip()

def identify_language(text):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="Detect the language of "+text,
        temperature=0,
        max_tokens=140,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response['choices'][0]['text'].strip()    

def detect_translate(target, text):
    result=[]
    detected = identify_language(text)
    result.append(detected)
    trans = translation(target, text)
    result.append(trans)
    return result

def code_explain(code):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="Explain what the mentioned code is doing: "+code,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\"\"\""]
    )
    return response['choices'][0]['text'].strip()

def sql_queries(query,schema=""):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=schema+" An SQL query to "+query,
        temperature=0,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["#", ";"]
    )
    return response['choices'][0]['text'].strip()

def sentiment(text):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="Classify the sentiment of this text:"+text,
        temperature=0,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response['choices'][0]['text'].strip()

def keywords(text):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="Extract keywords from this text: "+text,
        temperature=0,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response['choices'][0]['text'].strip()

def text_generator(keywords, char):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="Generate a paragraph in "+char+" characters using keywords: "+keywords,
        temperature=0,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response['choices'][0]['text'].strip()

def essay_outline(topic):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="Create an outline for an essay about"+topic,
        temperature=0,
        max_tokens=3000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response['choices'][0]['text'].strip()

def essay_generator(topic,outline="",limit="0"):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="Write an essay in "+limit+" words about "+topic+"using the outline"+outline,
        temperature=0,
        max_tokens=3000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response['choices'][0]['text'].strip()

match response:
    case 'Void':    
        st.header('This application is a one-stop solution for your NLP needs and more....')

    case 'General Queries':
        st.header('General Queries')
        text = st.text_input(
            "Enter your query here 👇*",
            label_visibility=st.session_state.visibility,
            disabled=st.session_state.disabled,
            key="1")
        if text:
            result=general(text)
            st.subheader("Output:")
            st.write(result)

    case 'Grammer and Spell Check':
        st.header('Grammer and Spell Check')
        inputtext = st.text_input(
            "Enter your text here 👇*",
            label_visibility=st.session_state.visibility,
            disabled=st.session_state.disabled,
            key="2")
        if inputtext:
            result=grammer(inputtext)
            st.subheader("Output:")
            st.write(result)

    case 'Summarize Text':
        st.header('Summarize Text')
        article = st.text_input(
            "Enter your article here 👇*",
            label_visibility=st.session_state.visibility,
            disabled=st.session_state.disabled,
            key="3")
        if article:
            output=summary(article)
            st.subheader("Output:")
            st.write(output)

    case 'Q&A':
        st.header('Questionnaire')
        question = st.text_input(
            "Enter your question here 👇*",
            label_visibility=st.session_state.visibility,
            disabled=st.session_state.disabled,
            key="4")
        if question:
            result=questionnaire(question)
            st.subheader("Answer: ")
            st.write(result)            

    case 'Language Translation':
        st.header('Language Translation')
        target = st.text_input(
            "Enter your target language here 👇*",
            label_visibility=st.session_state.visibility,
            disabled=st.session_state.disabled,
            key="5")
        text = st.text_input(
            "Enter your text here 👇*",
            label_visibility=st.session_state.visibility,
            disabled=st.session_state.disabled,
            key="6")
        if text and target:
            output=translation(target, text)
            st.subheader('Translated Text:')
            st.write(output)

    case 'Language Detection':
        st.header('Language Detection')
        text = st.text_input(
            "Enter your text here 👇*",
            label_visibility=st.session_state.visibility,
            disabled=st.session_state.disabled,
            key="7")
        if text:
            output=identify_language(text)
            st.subheader("Output:")
            st.write(output)

    case 'Detect and Translate':
        st.header('Detect and Translate')
        target = st.text_input(
            "Enter your target language here 👇*",
            label_visibility=st.session_state.visibility,
            disabled=st.session_state.disabled,
            key="8")
        text = st.text_input(
            "Enter your text here 👇*",
            label_visibility=st.session_state.visibility,
            disabled=st.session_state.disabled,
            key="9")
        if text and target:
            st.subheader('Language: ')
            output=detect_translate(target, text)
            st.write(output[0])
            st.subheader('Translation: ')
            st.write(output[1])

    case 'Code Explanation':
        st.header('Code Explanation')
        code = st.text_input(
            "Enter your code snippet here 👇*",
            label_visibility=st.session_state.visibility,
            disabled=st.session_state.disabled,
            key="10")
        if code:
            result=code_explain(code)
            st.subheader("Code explanation:")
            st.write(result)
        
    case 'Generate SQL Queries':
        st.header('Generate SQL Queries')
        query = st.text_input(
            "Enter your query objective here 👇*",
            label_visibility=st.session_state.visibility,
            disabled=st.session_state.disabled,
            key="11")
        schema= st.text_input(
            "Enter your schema here 👇",
            label_visibility=st.session_state.visibility,
            disabled=st.session_state.disabled,
            key="12")
        if query and schema:
            output=sql_queries(query, schema)
            st.subheader('Schema provided: ')
            st.write(schema)
            st.subheader('Query Objective: ')
            st.write(query)
            st.subheader('Query Generated: ')
            st.write(output)
        elif query:
            output=sql_queries(query)
            st.subheader('Query Objective: ')
            st.write(query)
            st.subheader('Query Generated: ')
            st.write(output)

    case 'Programming Language Conversion':
        st.header('Convert Code Snippet from one Programming Language to another')
        target = st.text_input(
            "Enter your target here 👇*",
            label_visibility=st.session_state.visibility,
            disabled=st.session_state.disabled,
            key="13")
        code = st.text_input(
            "Enter your code here 👇*",
            label_visibility=st.session_state.visibility,
            disabled=st.session_state.disabled,
            key="14")
        if target and code:
            st.subheader('Generated Code: ')
            result=translation(target, code)
            st.write(result)

    case 'Sentiment Analysis':
        st.header('Sentiment Analysis')
        text = st.text_input(
            "Enter your text here 👇*",
            label_visibility=st.session_state.visibility,
            disabled=st.session_state.disabled,
            key="15")
        if text:
            output=sentiment(text)
            st.subheader("Sentiment of the text:")
            st.write(output)

    case 'Extract Keywords':
        st.header('Extract Keywords from text')
        text = st.text_input(
            "Enter your text here 👇*",
            label_visibility=st.session_state.visibility,
            disabled=st.session_state.disabled,
            key="16")
        if text:
            output=keywords(text)
            st.subheader("Output:")
            st.write(output)

    case 'Text Generator from keywords':
        st.header('Generate Text from keywords')
        words = st.text_input(
            "Enter your keywords here 👇*",
            label_visibility=st.session_state.visibility,
            disabled=st.session_state.disabled,
            key="17")
        limit = st.text_input(
            "Enter your limit here 👇*",
            label_visibility=st.session_state.visibility,
            disabled=st.session_state.disabled,
            key="18")
        if words and limit:
            output=text_generator(words, limit)
            st.subheader("Generated Text:")
            st.write(output)

    case 'Essay Outline Generator':
        st.header('Generate Outline for Essay')
        topic = st.text_input(
            "Enter your topic here 👇*",
            label_visibility=st.session_state.visibility,
            disabled=st.session_state.disabled,
            key="19")
        if topic:
            output=essay_outline(topic)
            st.subheader("Essay Outline:")
            st.write(output)

    case 'Essay Generator':
        st.header('Generate Essay')
        topic = st.text_input(
            "Enter your topic here 👇*",
            label_visibility=st.session_state.visibility,
            disabled=st.session_state.disabled,
            key="20")
        outline = st.text_input(
            "Enter your outline here 👇",
            label_visibility=st.session_state.visibility,
            disabled=st.session_state.disabled,
            key="21")
        limit = st.text_input(
            "Enter your limit here 👇",
            label_visibility=st.session_state.visibility,
            disabled=st.session_state.disabled,
            key="22")
        if topic and outline:
            if limit:
                output=essay_generator(topic, outline=outline,limit=limit)
                st.subheader('Generated Essay:')
                st.write(output)
            else:
                output=essay_generator(topic,outline=outline)
                st.subheader('Generated Essay:')
                st.write(output)
        elif topic:
            if limit:
                output=essay_generator(topic, limit=limit)
                st.subheader('Generated Essay:')
                st.write(output)
            else:
                output=essay_generator(topic)
                st.subheader('Generated Essay:')
                st.write(output)