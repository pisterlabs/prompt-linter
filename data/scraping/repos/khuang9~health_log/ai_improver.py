import openai
from utils import * 
from .constants import *
openai.api_key = OPENAIKEY
import streamlit as st
from io import StringIO
from PIL import Image

FIXED_KEYS = ['NAME','SUMMARY','DOB','WORKING AS','CONTACTS']
VARIABLE_KEYS = ['EXPERIENCE','SCHOOL']
ORDERED_KEYS = ['NAME','DOB','WORKING AS','SUMMARY','EXPERIENCE','SCHOOL','CONTACTS']
DOB_PROMPT = 'Check that the following date is in YYYY/MM/DD. Correct it if its not in the same format, otherwise return the same: '
SUMMARY_PROMPT_CONVERT = 'Convert the text in a CV summary:'
TEMPERATURE_SUMMARY_PROMPT_CONVERT = 0.8
SUMMARY_PROMPT_IMPROVER = 'Improve the following text quality:'
TEMPERATURE_SUMMARY_PROMPT_IMPROVER = 0.3
OPENAIMODEL = 'text-davinci-003'
OPENAIKEY = "fake_key"
TEMPLATE_FILE = 'cv_template.txt'
EXPERIENCE_PROMPT_CONVERT = "Make the text more appealing for a recruiter:"
RESULT_FILE = 'cv_improved.txt'

def general_corrector(prompt, temperature,model = OPENAIMODEL,max_tokens = 20):
    openai.api_key = OPENAIKEY
    res = "blah blah"
    #openai.Completion.create(model=model,prompt=prompt,temperature=temperature,max_tokens=max_tokens)
    return res['choices'][0]['text']

def single_experience_corrector(experience_text):
    correct_text = general_corrector(prompt=EXPERIENCE_PROMPT_CONVERT+experience_text,temperature=0.4,max_tokens=200)
    st.markdown("<span style='color:lightblue'>"+experience_text+"</span>",
     unsafe_allow_html=True)
    st.text('The AI suggests the following summary instead: \n')
    #print(final_correction)
    st.markdown("<span style='color:red'>"+correct_text+"</span>",
             unsafe_allow_html=True)
    return correct_text

    
def summary_corrector(summary_text):
    print('The AI is rephrasing the text (if necessary): \n')
    st.text('The AI is rephrasing the text (if necessary):\n')
    first_correction = general_corrector(prompt=SUMMARY_PROMPT_CONVERT+summary_text,temperature=TEMPERATURE_SUMMARY_PROMPT_CONVERT,max_tokens=200)
    print('The AI is improving the rephrased summary \n')
    st.text('The AI is improving the rephrased summary \n')
    final_correction = general_corrector(prompt=SUMMARY_PROMPT_IMPROVER+first_correction,temperature =TEMPERATURE_SUMMARY_PROMPT_IMPROVER,max_tokens=200)
    print('The summary of your current CV is the following:\n')
    st.text('The AI is improving the rephrased summary \n')
    print(summary_text)
    #st.text(summary_text)
    st.text('The summary section of your CV is the following one: \n')
    st.markdown("<span style='color:lightblue'>"+summary_text+"</span>",
         unsafe_allow_html=True)
    st.text('The AI suggests the following summary instead: \n')
    print(final_correction)
    st.markdown("<span style='color:red'>"+final_correction+"</span>",
             unsafe_allow_html=True)
    return final_correction

def summary_corrector_main(summary_text):
    first_correction =   general_corrector(prompt=SUMMARY_PROMPT_CONVERT+summary_text,temperature=TEMPERATURE_SUMMARY_PROMPT_CONVERT,max_tokens=200)
    final_correction = general_corrector(prompt=SUMMARY_PROMPT_IMPROVER+first_correction,temperature =TEMPERATURE_SUMMARY_PROMPT_IMPROVER,max_tokens=200)
    return final_correction

def single_experience_corrector_main(experience_text):
    correct_text = general_corrector(prompt=EXPERIENCE_PROMPT_CONVERT+experience_text,temperature=0.4,max_tokens=200)
    return correct_text 