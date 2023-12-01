#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
[env]
# Conda Environment
# NO CONDA ENV

conda create --name ai_chatgpt_prompts python=3.9.13
conda info --envs
source activate ai_chatgpt_prompts
conda deactivate
# if needed to remove
conda env remove -n [NAME_OF_THE_CONDA_ENVIRONMENT]


# update conda 
conda update -n base -c defaults conda

# to export requirements
pip freeze > ai_chatgpt_prompts.txt


# to install
pip install -r ai_chatgpt_prompts.txt

[path]
cd /Users/brunoflaven/Documents/03_git/BlogArticlesExamples/ai_chatgpt_prompts/project_3_streamlit/

[file]
streamlit run 003_project_3_python_streamlit_app_chatgpt_api.py

https://stackoverflow.com/questions/65602056/how-to-set-and-access-environment-variables-in-python-file-for-streamlit-app

https://techcommunity.microsoft.com/t5/healthcare-and-life-sciences/integrating-azure-openai-with-streamlit-with-example-source-code/ba-p/3809006

https://github.com/ajitdash/pview/blob/main/explaincode.py


pip install openai
pip install streamlit
pip install python-dotenv

"""

# os + dotenv to manage the OpenAI platform's licence
import os as os
from dotenv import load_dotenv

# streamlit
import streamlit as st

# OpenAI platform
import openai

# personal configuration
import config_values.values_conf as conf

# Load environment variables from .env file
load_dotenv()
# Get API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# load the key
openai.api_key = OPENAI_API_KEY


### 1. VALUES ###
TEXT_TITLE_APP = conf.TEXT_TITLE_APP
TEXT_SUBHEADER_APP = conf.TEXT_SUBHEADER_APP
TEXT_WARNING = conf.TEXT_WARNING
TEXT_OUTPUT = conf.TEXT_OUTPUT

LABEL_EXPANDER = conf.LABEL_EXPANDER

# Usecase Title
MENU_SIDEBAR_USECASE_TITLE_OPTIONS = conf.MENU_SIDEBAR_USECASE_TITLE_OPTIONS

# Usecase Description
MENU_SIDEBAR_USECASE_DESCRIPTION_OPTIONS = conf.MENU_SIDEBAR_USECASE_DESCRIPTION_OPTIONS


TEXT_HELP_2 = conf.TEXT_HELP_2
GREAT_EXPENDER_TEXT_1 = conf.GREAT_EXPENDER_TEXT_1
GREAT_EXPENDER_TEXT_2 = conf.GREAT_EXPENDER_TEXT_2
GREAT_EXPENDER_TEXT_3 = conf.GREAT_EXPENDER_TEXT_3
TEXT_WARNING_REPORT = conf.TEXT_WARNING_REPORT
TEXT_WARNING_REPORT_HELP = conf.TEXT_WARNING_REPORT_HELP

TEXT_HELP_1 = conf.TEXT_HELP_1
TEXT_HELP_2 = conf.TEXT_HELP_2

### 2. USECASES VALUES ###

# CASE_6 :: 6. English to other languages
CASE_6_LANGUAGES_SELECTION = conf.CASE_6_LANGUAGES_SELECTION

TEXT_HELP_CASE_6_1 = conf.TEXT_HELP_CASE_6_1
TEXT_HELP_CASE_6_2 = conf.TEXT_HELP_CASE_6_2

# help for parameters
TEXT_HELP_TEMPERATURE_PARAMETER = conf.TEXT_HELP_TEMPERATURE_PARAMETER
TEXT_HELP_TOKENS_PARAMETER = conf.TEXT_HELP_TOKENS_PARAMETER



# CASE_3 :: 3. Summarize for a 2nd grader
TEXT_HELP_CASE_3_1 = conf.TEXT_HELP_CASE_3_1

# CASE_21 :: 21. TL DR summarization
TEXT_HELP_CASE_21_1 = conf.TEXT_HELP_CASE_21_1


### 3. FUNCTIONS ###

def greatExpender(label, text):
    with st.expander(label):
         st.info(f"{text}")
        
def detectVersion():
    st.sidebar.markdown('* * *')
    st.sidebar.markdown('**VERSIONS**')
    st.sidebar.write("streamlit ::", st.__version__)

# CASE_3
# 3. Summarize for a 2nd grader
def Case_3_DrawMainAppBody():
    
    if (menu == f"{str(MENU_SIDEBAR_USECASE_TITLE_OPTIONS[3])}"):
        st.markdown(
            f'**Summarize so it translates difficult text into simpler concepts.**')

        # st.code, st.latex or st.caption
        st.markdown('**Example:**')
        st.code('Canzone napoletana (pronounced[kanˈtsoːne napoleˈtaːna.Neapolitan: canzona napulitana[kanˈdzoːnə napuliˈtɑːnə]), sometimes referred to as Neapolitan song, is a generic term for a traditional form of music sung in the Neapolitan language, ordinarily for the male voice singing solo, although well represented by female soloists as well, and expressed in familiar genres such as the love song and serenade. Many of the songs are about the nostalgic longing for Naples as it once was. The genre consists of a large body of composed popular music—such songs as "’O sole mio" and others. The Neapolitan song became a formal institution in the 1830s due to an annual song-writing competition for the Festival of Piedigrotta, dedicated to the Madonna of Piedigrotta, a well-known church in the Mergellina area of Naples.')

        text_input = st.text_area(
            "Enter the text to summarize", help=TEXT_HELP_CASE_3_1)

        # Define function to summarize text using OpenAI Codex
        def summarize_text(text_input):
            # Change to the desired OpenAI model
            model_engine = "text-davinci-003"
            prompt = f"Summarize this for a second-grade student:\n\n{text_input}"
            response = openai.Completion.create(
                engine=model_engine,
                prompt=prompt,
                temperature=temperature,
                max_tokens=tokens,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            return response.choices[0].text

        st.markdown('**Set parameters:**')
        # Temperature and token slider
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help=TEXT_HELP_TEMPERATURE_PARAMETER
        )
        tokens = st.slider(
            "Tokens",
            min_value=64,
            max_value=2048,
            value=256,
            step=64,
            help=TEXT_HELP_TOKENS_PARAMETER
        )

        # Define Streamlit app behavior
        if st.button("Summarize"):
            output_text = summarize_text(text_input)
            st.text_area("Text summarized", output_text)

# CASE_6
# 6. English to other languages
def Case_6_DrawMainAppBody():
    
    if (menu == f"{str(MENU_SIDEBAR_USECASE_TITLE_OPTIONS[6])}"):
        
        st.markdown(
            f'**Translates English text into languages via the dropdown menu.**')

        # st.code, st.latex or st.caption
        st.markdown('**Example:**')
        st.code('How many children, do you have?')

        st.markdown('**Use case:**')
        language = st.multiselect(
            "Select Language", CASE_6_LANGUAGES_SELECTION, help=TEXT_HELP_CASE_6_1)

        text_input = st.text_area(
            "Enter the text to translate", help=TEXT_HELP_CASE_6_2)
        
        # Define function to translate text using OpenAI Codex
        def explain_code(text_input, language):
            # Change to the desired OpenAI model
            model_engine = "text-davinci-003"
            prompt = f"Translate into: {language} this text: \n\n{text_input}"
            response = openai.Completion.create(
                engine=model_engine,
                prompt=prompt,
                temperature=temperature,
                max_tokens=tokens,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            return response.choices[0].text

        st.markdown('**Set parameters:**')
        # Temperature and token slider
        temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1,
                help=TEXT_HELP_TEMPERATURE_PARAMETER
            )
        tokens = st.slider(
                "Tokens",
                min_value=64,
                max_value=2048,
                value=100,
                step=64,
                help=TEXT_HELP_TOKENS_PARAMETER
            )

        # Define Streamlit app behavior
        if st.button("Translate"):
            output_text = explain_code(text_input, language)
            st.text_area("Text translated", output_text)
# CASE_17
# 17. Keywords
def Case_17_DrawMainAppBody():
    if (menu == f"{str(MENU_SIDEBAR_USECASE_TITLE_OPTIONS[17])}"):
        
        st.markdown(
            f'**Extract keywords from a block of text. At a lower temperature it picks keywords from the text. At a higher temperature it will generate related keywords which can be helpful for creating search indexes.**')

        # st.code, st.latex or st.caption
        st.markdown('**Example:**')
        st.code("What does the future hold for Wagner in Africa after the failed rebellion?\nRussian Foreign Minister Sergei Lavrov on Monday said that Wagner group troops would continue to operate in Mali and the Central African Republic following the rebellion led by their commander Yevgeny Prigozhin last weekend. Lavrov’s words came amid questions over the private militia’s role in Africa after more than five years of deployment to the continent.\nMembers of the Close Protection Unit for Central African Republic President Touadera, composed of Russian private security company operatives from Sewa Security, are seen in Berengo on August 4, 2018.\nMembers of the Close Protection Unit for Central African Republic President Touadera, composed of Russian private security company operatives from Sewa Security, are seen in Berengo on August 4, 2018. © Florent Vergnes, AFP\nThe Wagner group’s mutiny against Moscow last weekend has raised questions over the private militia’s presence in Africa.\nFor more than five years, thousands of Wagner group troops have deployed to Africa. In 2018 they were sent to support Central African Republic President Faustin-Archange Touadera as he grappled with an armed rebellion. In December 2021 they deployed to Mali in the wake of a coup and they have operated in Sudan since the end of 2017. Wagner troops have also been spotted in Libya and Mozambique.\nBut Lavrov vowed on Monday that the “events” of the last weekend would not impact the militia’s operations on the continent.\nThe Russian \"instructors\" and “private military contractors” in both Mali and the Central African Republic will continue their work, Lavrov said in an interview with Russia Today.\nPrigozhin’s rebellion will not change anything in Russia's ties with its allies, Lavrov added. \"There have been many calls(from foreign partners) to President(Vladimir) Putin ... to express their support, \" he said.\nNo African government has officially commented on the events of the past weekend. But according to Cyril Payen, senior reporter for FRANCE 24, Russia \"has undoubtedly become a slightly less reliable partner\" since Prigozhin’s rebellion.\"You can bet that people in Bangui and Bamako are wondering what the future holds, \" Payen added.\"The Malian state is now engaged in a double partnership, with the Russian state – the Putin camp – and with the Wagner group – the Prigozhin camp. Until now, this did not make much difference, but that could change if the two camps don’t reconcile in the long term, \" said lawyer and political scientist Oumar Berté in an interview with FRANCE 24’s sister radio station Radio France Internationale.\nOverlapping interests\nHowever, a senior official in the Central African Republic’s presidency told AFP that Russia will continue to operate in the central African country, with or without Wagner.\"The Central African Republic signed (in 2018, editor's note) a defence agreement with the Russian Federation, not with Wagner, \" said Fidele Gouandjika, special adviser to President Touadera.\"Russia has subcontracted with Wagner. If Russia no longer agrees with Wagner, then it will send us a new contingent.\"\n\nA bridgehead for Russian ambitions on the continent, the Central African Republic is particularly dependent on the Russian militia, whose men even work as private protection officers for Touadera.\n\nSome 1, 500 Wagner troops have been deployed to Mali since 2021. The paramilitary group has developed close ties with the junta in power, helping to train soldiers as well as taking part in operations to combat terrorist groups.\n\nPrigozhin’s men have also been seen in Libya, Sudan and Mozambique. Since the Wagner's group arrival in Africa, the UN, international NGOs and French authorities have regularly accused the paramilitary group of committing abuse and crimes against civilians.\n\nWagner always uses the same strategy every time it advances: disinformation campaigns(based on rejecting former colonial powers) and an offer of security in exchange for the exploitation of natural resources to supply Prigozhin’s war chest and serve the Kremlin’s interests.\n\nIn Sudan, the partnership between Wagner and the Rapid Support Forces(RSF), led by the junta’s number two, General Mohammed Hamdan Daglo, has enabled the paramilitary group to profit from illegal gold trafficking. It has also enabled them to organise the transport of the metal straight into the coffers of the Russian state, helping to swell its gold reserves and circumvent Western sanctions.")
        
        text_input = st.text_area(
            "Enter the text to extract keywords", help=TEXT_HELP_CASE_3_1)

        # Define function to summarize text using OpenAI Codex
        def extract_kws_text(text_input):
            # Change to the desired OpenAI model
            model_engine = "text-davinci-003"
            prompt = f"Extract keywords from this text:\n\n{text_input}"
            response = openai.Completion.create(
                engine=model_engine,
                prompt=prompt,
                temperature=temperature,
                max_tokens=tokens,
                top_p=1.0,
                frequency_penalty=0.8,
                presence_penalty=0.0
            )
            return response.choices[0].text

        st.markdown('**Set parameters:**')
        # Temperature and token slider
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help=TEXT_HELP_TEMPERATURE_PARAMETER
        )
        tokens = st.slider(
            "Tokens",
            min_value=60,
            max_value=2048,
            value=256,
            step=64,
            help=TEXT_HELP_TOKENS_PARAMETER
        )

        # Define Streamlit app behavior
        if st.button("Extract"):
            output_text = extract_kws_text(text_input)
            st.text_area("Text summarized", output_text)
# CASE_21
# 21. TL;DR summarization
def Case_21_DrawMainAppBody():
    
    st.markdown(
        f'**Summarize text by adding a \'tl dr: \' to the end of a text passage.**')
    
    st.info(
        f'TL;DR or TLDR stands for "Too Long; Didn\'t Read." TLDR can be used to express that a text is too long, identify a short summary of a long text, or ask for a summary of a long text.')
    
    st.markdown('**Example:**')
    st.code(
        "A neutron star is the collapsed core of a massive supergiant star, which had a total mass of between 10 and 25 solar masses, possibly more if the star was especially metal-rich.[1] Neutron stars are the smallest and densest stellar objects, excluding black holes and hypothetical white holes, quark stars, and strange stars.[2] Neutron stars have a radius on the order of 10 kilometres (6.2 mi) and a mass of about 1.4 solar masses.[3] They result from the supernova explosion of a massive star, combined with gravitational collapse, that compresses the core past white dwarf star density to that of atomic nuclei.")

    text_input = st.text_area(
        "Enter the text to TL;DR summarize", help=TEXT_HELP_CASE_21_1)

    # Define function to summarize text using OpenAI Codex
    def extract_kws_text(text_input):
            # Change to the desired OpenAI model
            model_engine = "text-davinci-003"
            prompt = f"{text_input}\n\nTl;dr"
            response = openai.Completion.create(
                engine=model_engine,
                prompt=prompt,
                temperature=temperature,
                max_tokens=tokens,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=1
            )
            return response.choices[0].text

    st.markdown('**Set parameters:**')
        # Temperature and token slider
    temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help=TEXT_HELP_TEMPERATURE_PARAMETER
        )
    tokens = st.slider(
            "Tokens",
            min_value=60,
            max_value=2048,
            value=60,
            step=64,
            help=TEXT_HELP_TOKENS_PARAMETER
        )

    # Define Streamlit app behavior
    if st.button("TL;DR Summarize"):
        output_text = extract_kws_text(text_input)
        st.text_area("TL;DR Text summarized", output_text)

# CASE_9
# 9. Parse unstructured data

# CASE_10
# 10. Classification

# CASE_11
# 11. Python to natural language

# CASE_12
# 12. Movie to Emoji(movie_to_emoji)

# CASE_22
# 22. Python bug fixer

# CASE_23
# 23. Spreadsheet creator

# CASE_39
# 39. Notes to summary


###############################################################################

# The code below is for the layout of the page
if "widen" not in st.session_state:
    layout = "centered"
else:
    layout = "wide" if st.session_state.widen else "centered"

st.set_page_config(
    layout=layout, page_title=TEXT_TITLE_APP, page_icon="🔌"
)

def main():
    """ A simple attempt with Streamlit. Examples extracted from  Python Documentation ChatGPT API """


# MAIN LAYOUT
st.header(f'{TEXT_TITLE_APP}')

### INTRO ###
st.markdown(f'{TEXT_SUBHEADER_APP}')
st.warning(f'{TEXT_WARNING}')

### MENU ###
options = MENU_SIDEBAR_USECASE_TITLE_OPTIONS

menu = st.sidebar.selectbox("Menu options", options, help=TEXT_HELP_1)

for i, option in enumerate(MENU_SIDEBAR_USECASE_TITLE_OPTIONS):
    if menu == str(option):
        if i == 0:
            st.markdown(f"### {option}")
            greatExpender(LABEL_EXPANDER,
                          f"{MENU_SIDEBAR_USECASE_DESCRIPTION_OPTIONS[i]}")
        
        if i == 3:
            greatExpender(LABEL_EXPANDER,
                          f"{MENU_SIDEBAR_USECASE_DESCRIPTION_OPTIONS[i]}")
            Case_3_DrawMainAppBody()

        if i == 6:
            
            greatExpender(LABEL_EXPANDER,
                          f"{MENU_SIDEBAR_USECASE_DESCRIPTION_OPTIONS[i]}")
            # MAIN BODY
            Case_6_DrawMainAppBody()
        if i == 17:

            greatExpender(LABEL_EXPANDER,
                          f"{MENU_SIDEBAR_USECASE_DESCRIPTION_OPTIONS[i]}")
            # MAIN BODY
            Case_17_DrawMainAppBody()
        
        if i == 21:

            greatExpender(LABEL_EXPANDER,
                          f"{MENU_SIDEBAR_USECASE_DESCRIPTION_OPTIONS[i]}")
            # MAIN BODY
            Case_21_DrawMainAppBody()
            
        else:
            # st.markdown(f"### USECASE_{i} :: {option}")
            # st.markdown(f"### {option}")
            # just here to make it happen
            st.write('')
            

if __name__ == '__main__':
    main()
    detectVersion()
