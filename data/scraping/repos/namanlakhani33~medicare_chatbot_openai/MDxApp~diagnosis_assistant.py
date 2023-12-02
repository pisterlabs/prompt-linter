import streamlit as st
import os
import json
import openai
from streamlit.components.v1 import html


path = os.path.dirname(__file__)

# Load translations from JSON file
with open(path+"/../Assets/translations.json", 'r', encoding='utf-8') as f:
    transl = json.load(f)

# Replace 'YOUR_OPENAI_API_KEY' with your actual OpenAI API key
# openai.api_key = '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'
openai.api_key = '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'

def openai_create(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages= [
    {"role": "system", "content": "You are a medical assistant."},
    {"role": "user", "content": "I need help identifying a possible disease. My patient has the following symptoms:"},
    {"role": "user", "content": "1. Severe headache\n2. Nausea\n3. Blurred vision"},
    {"role": "assistant", "content": "I'm here to assist. Let me analyze the symptoms."},
    {"role": "assistant", "content": "Based on the symptoms provided (severe headache, nausea, blurred vision), there are several possible conditions that could be causing these symptoms."},
    {"role": "assistant", "content": "Some potential conditions associated with these symptoms include migraines, hypertension, and certain eye conditions. However, please keep in mind that this is a general suggestion and not a specific diagnosis."},
    {"role": "user", "content": "Can you provide more details about these conditions?"}
],

        temperature=0.7,
        max_tokens=150,
        frequency_penalty=0.0,
        presence_penalty=0.6,
        stop=None
    )
    return response['choices'][0]['message']['content']

# Rest of your code...

st.set_page_config(
    page_title="Diagnosis_Assistant",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
  <style>
      .css-zck4sz p {
        font-weight: bold;
        font-size: 18px;
      }
  </style>""", unsafe_allow_html=True)

# Add the language selection dropdown    
if 'lang_tmp' not in st.session_state:
    st.session_state['lang_tmp'] = 'English'

if 'lang_changed' not in st.session_state:
    st.session_state['lang_changed'] = False

if 'lang_select' in st.session_state:
    #st.sidebar.markdown("<h3 style='text-align: center; color: black;'>{}</h3>".format(transl[st.session_state['lang_select']]["language_selection"]), unsafe_allow_html=True)
    lang = st.sidebar.selectbox(transl[st.session_state['lang_select']]["language_selection"], options=list(transl.keys()), key='lang_select')
else:
    #st.sidebar.markdown("<h3 style='text-align: center; color: black;'>{}</h3>".format(transl[st.session_state['lang_tmp']]["language_selection"]), unsafe_allow_html=True)
    lang = st.sidebar.selectbox(transl[st.session_state['lang_tmp']]["language_selection"], options=list(transl.keys()), key='lang_select')

if lang != st.session_state['lang_tmp']:
    st.session_state['lang_tmp'] = lang
    st.session_state['lang_changed'] = True
else:
    st.session_state['lang_changed'] = False

# Line separator for clarity
st.sidebar.markdown("""---""")

# Font size and weight for the sidebar
# Works with streamlit==1.17.0
# TODO: Review class names for future versions
st.markdown("""
  <style>
      ul[class="css-j7qwjs e1fqkh3o7"]{
        position: relative;
        padding-top: 2rem;
        display: flex;
        justify-content: center;
        flex-direction: column;
        align-items: center;
      }
      .css-17lntkn {
        font-weight: bold;
        font-size: 18px;
        color: grey;
      }
      .css-pkbazv {
        font-weight: bold;
        font-size: 18px;
      }
  </style>""", unsafe_allow_html=True)

# Use GitHub logo file
logo_name = path+"/../Assets/chatgpt-doctor-neurosceuncee.png"

# Define columns
t1, t2 = st.columns([1,3], gap="large")



# with t1: 
    # image_path = '../MDxApp/Assets/chatgpt-doctor-neurosceuncee.png'
    # st.image(image_path, caption='', width=256)
with t2:
    st.header("**{}**".format(transl[lang]['page1_header']))
    st.write("<p style=\"font-weight: bold; font-size:18px;\">{}</p>".format(transl[lang]['page1_subheader']), 
                 unsafe_allow_html=True)    

st.markdown("", unsafe_allow_html=True)
"""
---
"""
st.write("<p style=\"font-weight: bold; font-size:18px;\">{}</p>".format(transl[lang]['htu_0'])+ \
         "<p style=\"font-size:18px;\">1. {}<br/>".format(transl[lang]['htu_1'])+ \
         "2. {}<br/>".format(transl[lang]['htu_2'])+ \
         "3. {}</p>".format(transl[lang]['htu_3']), 
         unsafe_allow_html=True)

st.write(
    """<style>
    [data-testid="stHorizontalBlock"] {
        align-items: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

## Font size configs
st.markdown(
    """<style>
div[class*="stRadio"] > label > div[data-testid="stMarkdownContainer"] > p {
    font-size: 18px;
}
    </style>
    """, unsafe_allow_html=True)
st.markdown(
    """<style>
div[class*="stNumberInput"] > label > div[data-testid="stMarkdownContainer"] > p {
    font-size: 18px;
}
    </style>
    """, unsafe_allow_html=True)
st.markdown(
    """<style>
div[class*="stTextInput"] > label > div[data-testid="stMarkdownContainer"] > p {
    font-size: 18px;
}
    </style>
    """, unsafe_allow_html=True)
####

st.subheader(":black_nib: **{}**".format(transl[lang]['report_header']))
# Define columns
col1, col2, col3 = st.columns(3, gap="large")

# Store initial values in session state
if "disabled" not in st.session_state:
    st.session_state.disabled = False
# Declare lists
genders_list = ["{}".format(transl[lang]['male']), "{}".format(transl[lang]['female'])]
pregnant_list = ["{}".format(transl[lang]['no']), "{}".format(transl[lang]['yes'])]
##

# Gender selector 
with col1:   
    if st.session_state['lang_changed'] and 'gender' in st.session_state:     
        del st.session_state['gender']
    if 'gender' not in st.session_state:
        st.session_state['gender'] = genders_list[0]
    st.radio("**{}**".format(transl[lang]['gender']), genders_list, key='gender')
# Age selector 
with col2:
    st.number_input("**{}**".format(transl[lang]['age']), min_value= 0, max_value= 99, step=1, key="age")
# Pregnancy
if st.session_state.gender == '{}'.format(transl[lang]['male']):
    st.session_state.disabled = True
    if 'pregnant' in st.session_state:
        st.session_state['pregnant'] = "{}".format(transl[lang]['no'])
else: 
    st.session_state.disabled = False

with col3: 
    if st.session_state['lang_changed'] and 'pregnant' in st.session_state:
        del st.session_state['pregnant']
    if 'pregnant' not in st.session_state:
        st.session_state['pregnant'] = pregnant_list[0]
    st.radio("**{}**".format(transl[lang]['pregnant']), pregnant_list, disabled=st.session_state.disabled, key='pregnant')

# Context
st.text_input('**{}** *{}*'.format(transl[lang]['history'], transl[lang]['hist_example']), 
             placeholder="{}".format(transl[lang]['hist_ph']), key="context", max_chars=250, 
             help=":green[**{}**]".format(transl[lang]['hist_help']))

# List of symptoms
st.text_input("**{}** *{}*".format(transl[lang]['symptoms'], transl[lang]['symp_example']), 
             placeholder="{}".format(transl[lang]['symp_ph']), key="symptoms", max_chars=250, 
             help=":green[**{}**]".format(transl[lang]['symp_help']))

# List of observations at exam
st.text_input("**{}** *{}*".format(transl[lang]['exam'], transl[lang]['exam_example']), 
             placeholder="{}".format(transl[lang]['exam_ph']), key="exam", max_chars=250, 
             help=":green[**{}**]".format(transl[lang]['exam_help']))

# Laboratory test results
st.text_input("**{}** *{}*".format(transl[lang]['lab'], transl[lang]['lab_example']), 
             placeholder="{}".format(transl[lang]['lab_ph']), key="labresults", max_chars=250, 
             help=":green[**{}**]".format(transl[lang]['lab_help']))

st.subheader(":clipboard: **{}**".format(transl[lang]['summary']))
# Diagnostic

report_list = [st.session_state.context, st.session_state.symptoms, 
               st.session_state.exam, st.session_state.labresults]
corr_list = ["{}".format(transl[lang]['none']), 
             "{}".format(transl[lang]['none']), 
             "{}".format(transl[lang]['none']), 
             "{}".format(transl[lang]['none'])]
for ic in range(0,len(report_list)):
    if report_list[ic] == "":
        report_list[ic] = corr_list[ic]

vis_summary = "<p style=\"font-size:18px;\">" + \
              "<b>{}</b>".format(transl[lang]['vissum_patient']) + \
              st.session_state.gender + ", " + str(st.session_state.age) + "{}<br/>".format(transl[lang]['vissum_yrsold']) + \
              "<b>{}</b>".format(transl[lang]['vissum_pregnancy']) + st.session_state.pregnant + "<br/>" + \
              "<b>{}</b>".format(transl[lang]['vissum_history']) + report_list[0] + "<br/>" + \
              "<b>{}</b>".format(transl[lang]['vissum_symp']) + report_list[1] + "<br/>" + \
              "<b>{}</b>".format(transl[lang]['vissum_exam']) + report_list[2] + "<br/>" + \
              "<b>{}</b>".format(transl[lang]['vissum_lab']) + report_list[3] + "<br/> </p>"

st.write(vis_summary, unsafe_allow_html=True) 

prompt_words = {
    "prompt_words_0": "Your prompt word 0: ",
    "prompt_words_1": "Your prompt word 1: ",
    "prompt_words_2": "Your prompt word 2: ",
    "prompt_words_3": "Your prompt word 3: ",
    "prompt_words_4": "Your prompt word 4: ",
    "prompt_words_5": "Your prompt word 5: ",
    "prompt_words_6": "Your prompt word 6: ",
    "prompt_words_7": "Your prompt word 7: ",
    "prompt_words_8": "Your prompt word 8: ",
    "prompt_words_9": "Your prompt word 9: "
} 


# question_prompt = (
#     prompt_words["prompt_words_0"]
#     + st.session_state.gender
#     + ", "
#     + str(st.session_state.age)
#     + "{}. ".format(transl[lang]['vissum_yrsold'])
#     + prompt_words["prompt_words_1"]
#     + st.session_state.pregnant
#     + ". "
#     + prompt_words["prompt_words_2"]
#     + report_list[0]
#     + ". "
#     + prompt_words["prompt_words_3"]
#     + report_list[1]
#     + ". "
#     + prompt_words["prompt_words_4"]
#     + report_list[2]
#     + ". "
#     + prompt_words["prompt_words_5"]
#     + report_list[3]
#     + ". "
#     + prompt_words["prompt_words_6"]
#     + prompt_words["prompt_words_7"]
#     + prompt_words["prompt_words_8"]
#     + prompt_words["prompt_words_9"]
#     + lang
#     + ". "
# )

question_prompt = prompt_words["prompt_words_0"] + st.session_state.gender + ", " + \
                 str(st.session_state.age) + "{}. ".format(transl[lang]['vissum_yrsold']) + \
                 prompt_words["prompt_words_1"] + st.session_state.pregnant + ". " + \
                 prompt_words["prompt_words_2"] + report_list[0] + ". " + \
                 prompt_words["prompt_words_3"] + report_list[1] + ". " + \
                 prompt_words["prompt_words_4"] + report_list[2] + ". " + \
                 prompt_words["prompt_words_5"] + report_list[3] + ". " + \
                 prompt_words["prompt_words_6"] + prompt_words["prompt_words_7"] + \
                 prompt_words["prompt_words_8"] + prompt_words["prompt_words_9"] + lang + ". "


st.write('')
submit_button = st.button('**{}**'.format(transl[lang]['submit']), help=":green[**{}**]".format(transl[lang]['submit_help']))
st.write('')

st.subheader(":computer: :speech_balloon: :pill: **{}**".format(transl[lang]['diagnostic']))
if submit_button:
    if report_list[1] == "{}".format(transl[lang]['none']):
        st.write("<p style=\"font-weight: bold; font-size:18px;\">{}</p>".format(transl[lang]['submit_warning']), 
                 unsafe_allow_html=True)
    else:
        with st.spinner('{}'.format(transl[lang]['submit_wait'])):
            try:
                st.session_state.diagnostic = openai_create(prompt=question_prompt)
                st.write('')
                st.write(st.session_state.diagnostic.replace("<|im_end|>", ""), unsafe_allow_html=True)
                st.markdown(
                            """
                            ### :rotating_light: **{}** :rotating_light:
                            {}
                            """.format(transl[lang]['caution'], transl[lang]['caution_message']), 
                            unsafe_allow_html=True
                           )
                
                # Buy me a coffee - MDxApp support
                st.markdown("""---""")
                st.markdown("<h4 style='text-align: left; color: black;'>{}</h4>".format(transl[lang]['invest']), unsafe_allow_html=True)
                st.markdown("<h5 style='text-align: left; color: black;'>{}</h5>".format(transl[lang]['bmc_1']), unsafe_allow_html=True)
                html(height=70, width=220)
                st.markdown("<h5 style='text-align: left; color: black;'>{}</h5>".format(transl[lang]['bmc_2']), unsafe_allow_html=True)
                qr_name = path+"/../Materials/bmc_qr.png"
                st.image(qr_name, caption= '', width = 220)
            except Exception as e: 
                #st.write(e) 
                st.write("<p style=\"font-weight: bold; font-size:18px;\">{}</p>".format(transl[lang]['no_response']), 
                         unsafe_allow_html=True)
else: 
    if "diagnostic" in st.session_state:
        st.write(st.session_state.diagnostic.replace("<|im_end|>", ""), unsafe_allow_html=True)
    else: 
        st.write("<p style=\"font-weight: bold; font-size:18px;\">{}</p>".format(transl[lang]['no_diagnostic']), 
                 unsafe_allow_html=True)