import streamlit as st
import streamlit.components.v1 as html
from bokeh.models import CustomJS
from bokeh.models import Button
from bokeh.layouts import row
from streamlit_bokeh_events import streamlit_bokeh_events
import time
from gtts import gTTS
from io import BytesIO
import base64
import openai
import os
openai.api_key = os.environ.get('api_key')
openai.api_base = os.environ.get('api_base')
openai.api_type = os.environ.get('api_type')
openai.api_version = os.environ.get('api_version')
audio_byte_io = BytesIO()
def resize_svg(svg_path, size):
    with open(svg_path, 'r') as file:
        svg_content = file.read().replace('<svg ', f'<svg width="{size}" height="{size}" ')
    return svg_content
def generate_response(prompt):
    st.session_state['prompts'].append({"role": "user", "content":prompt})
    completion=openai.ChatCompletion.create(
        engine = "PvA",
        model="gpt-3.5-turbo",
        messages = st.session_state['prompts']
    )
    
    message=completion.choices[0].message.content
    return message
def audio_output(output):
    audio = st.empty()
    audio_byte_io = BytesIO()
    
    tts = gTTS(output, lang='en', tld='com')
    tts.write_to_fp(audio_byte_io)

    sound_b64 = base64.b64encode(audio_byte_io.getvalue()).decode("utf-8")
    audio_html = f'<audio id="audioElement" controls autoplay><source src="data:audio/mp3;base64,{sound_b64}"></audio>'
    audio.markdown(audio_html, unsafe_allow_html=True)


if 'input' not in st.session_state:
    st.session_state['input'] = dict(text='', session=0)
if 'prompts' not in st.session_state:
    st.session_state['prompts'] = [{"role": "system", "content": "Act like JARVIS voice assistant from in the MCU. Answer as concisely as possible with a lot of humor expression. And always refer to the user as sir"}]
title, placeholder, image_place  = st.columns([10,10,1])
tr = st.empty()
val = tr.text_area("**Your input**", value=st.session_state['input']['text'])
size = 32
mic_off = resize_svg("VoiceBot/mic-off.svg", size)
mic_on = resize_svg("VoiceBot/mic-on.svg", size)
    
title.title('Voice Bot')
placeholder.empty()
image_holder = image_place.image(mic_off)
speak_js = CustomJS(code="""
    var value = "";
    var rand = 0;
    var recognition = new webkitSpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = 'en';
    document.dispatchEvent(new CustomEvent("GET_ONREC", {detail: 'start'}));
    
    recognition.onspeechstart = function () {
        document.dispatchEvent(new CustomEvent("GET_ONREC", {detail: 'running'}));
    }
    recognition.onsoundend = function () {
        document.dispatchEvent(new CustomEvent("GET_ONREC", {detail: 'stop'}));
    }
    recognition.onresult = function (e) {
        var value2 = "";
        for (var i = e.resultIndex; i < e.results.length; ++i) {
            if (e.results[i].isFinal) {
                value += e.results[i][0].transcript;
                rand = Math.random();
                
            } else {
                value2 += e.results[i][0].transcript;
            }
        }
        document.dispatchEvent(new CustomEvent("GET_TEXT", {detail: {t:value, s:rand}}));
        document.dispatchEvent(new CustomEvent("GET_INTRM", {detail: value2}));
    }
    recognition.onerror = function(e) {
        document.dispatchEvent(new CustomEvent("GET_ONREC", {detail: 'stop'}));
    }
    recognition.start();
    """)
enter_js = CustomJS(code="""
    document.dispatchEvent(new CustomEvent("GET_ONREC", {detail: 'stop'}));
""")
button_size = 30
speak_button = Button(label="Speak", button_type="success", width=button_size, height=button_size, name = 'speak_button')
enter_button = Button(label="Enter", button_type="success", width=button_size, height=button_size)
speak_button.js_on_event("button_click", speak_js)
enter_button.js_on_event("button_click", enter_js)
r = row(speak_button,enter_button, sizing_mode='stretch_width')
result = streamlit_bokeh_events(
    bokeh_plot = r,
    events="GET_TEXT,GET_ONREC,GET_INTRM",
    key="listen",
    refresh_on_update=False,
    override_height=75,
    debounce_time=0)
if result:
    if "GET_TEXT" in result:
        if result.get("GET_TEXT")["t"] != '' and result.get("GET_TEXT")["s"] != st.session_state['input']['session'] :
            st.session_state['input']['text'] = result.get("GET_TEXT")["t"]
            val = tr.text_area("**Your input**", value= st.session_state['input']['text'])
            st.session_state['input']['session'] = result.get("GET_TEXT")["s"]
            
    if "GET_INTRM" in result:
        if result.get("GET_INTRM") != '':
            val = tr.text_area("**Your input**", value=st.session_state['input']['text']+' '+result.get("GET_INTRM"))
    if "GET_ONREC" in result:
        if result.get("GET_ONREC") == 'start':
            image_holder.image(mic_on)
            st.session_state['input']['text'] = ''
        elif result.get("GET_ONREC") == 'running':
            image_holder.image(mic_on)
        elif result.get("GET_ONREC") == 'stop':
            image_holder.image(mic_off)
            if val != '':
                input = val
                output = generate_response(input)
                st.write("**ChatBot:**")
                st.write(output)
                audio_output(output)
                st.session_state['prompts'].append({"role": "user", "content":input})
                st.session_state['prompts'].append({"role": "assistant", "content":output})