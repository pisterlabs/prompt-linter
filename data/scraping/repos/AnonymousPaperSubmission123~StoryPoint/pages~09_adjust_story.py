"""
This is the Layout Creation Page
of the Data Story Authoring Tool

Author: Anonymous
"""


# streamlit packages
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.app_logo import add_logo
from streamlit_option_menu import option_menu
from streamlit_ace import st_ace
from streamlit_extras.add_vertical_space import add_vertical_space
from st_pandas_text_editor import st_pandas_text_editor
import streamlit.components.v1 as components
from streamlit.components.v1 import html

# dataframe handling
import pandas as pd  # read csv, df manipulation

# reusable functions, outsourced into another file
from helper_functions import GPTHelper

# multivision and threading
from multivision.multivision import Recommender

# handle GPT API
from langchain.chains import ConversationChain

# formats the prompt history in a particular way
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.llms import OpenAI
from langchain import LLMChain

# other modules
import time
import json
from PIL import Image
import vl_convert as vlc
import os
import base64
import path
import sys

# dataframe handling
import pandas as pd  # read csv, df manipulation

# reusable functions, outsourced into another file
from helper_functions import GPTHelper

# other
import base64

# Initialize session states
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []

# instanciate gptHelperFunctions
gpt_helper = GPTHelper()

# configure the page
st.set_page_config(
    page_title="Conversational Dashboard",
    page_icon="✅",
    layout="wide"
    # initial_sidebar_state="collapsed"
)
# feedback counter so that the form doesn't reopen on rerun
if not "feedback_counter" in st.session_state:
    st.session_state["feedback_counter"] = 0

def model_initialisation(TEMPERATURE, MODEL, K, column_names):
    # custom query template --> possible to add few shot examples in the future
    # add dynamic variables columns and data types to the prompt
    template = (
        """
    You are a great assistant at vega-lite visualization creation. No matter what
    the user ask, you should always response with a valid vega-lite specification
    in JSON.

    You should create the vega-lite specification based on user's query.

    Besides, Here are some requirements:
    1. Do not contain the key called 'data' in vega-lite specification.
    2. If the user ask many times, you should generate the specification based on the previous context.
    3. You should consider to aggregate the field if it is quantitative and the chart has a mark type of react, bar, line, area or arc.
    4. The available fields in the dataset are:
    %s
    5. Always respond with exactly one vega-lite specfication. Not more, not less.
    6. If you use a color attribute, it must be inside the encoding block attribute of the specification.
    7. When the user tells you to give him a sample graph, then you give him a vega-lite specification that you think,
    will look good.
    8. remember to only respond with vega-lite specifications without additional explanations

    Current conversation:
    {history}
    Human: {input}
    AI Assistant:"""
        % column_names
    )

    PROMPT = PromptTemplate(
        input_variables=["history", "input"], template=template
    )

    # Create an OpenAI instance
    llm = OpenAI(
        temperature=TEMPERATURE,
        openai_api_key=st.secrets["openai_api_key"],
        model_name=MODEL,
        verbose=False,
        streaming=True,
    )

    # Create a ConversationEntityMemory object if not already created
    if "entity_memory" not in st.session_state:
        st.session_state.entity_memory = ConversationBufferWindowMemory(k=K)

    # Create the ConversationChain object with the specified configuration
    Conversation = ConversationChain(
        llm=llm,
        prompt=PROMPT,
        memory=st.session_state.entity_memory,
    )

    return Conversation

def style():
    """
    Apply custom styles to the page, remove sidebar elements, and add custom
    CSS for the sticky header.

    This function applies custom CSS styles to the page, including removing
    whitespace from the top of the page and sidebar.
    It defines CSS classes for styling specific elements, such as custom-div,
    block-container, blue-text, and normal-text.
    The function also hides the footer, removes the sidebar pages, and adds
    custom CSS for the sticky header.

    Returns:
        None
    """

    # Remove whitespace from the top of the page and sidebar
    st.markdown(
        """
            <style>
                .custom-div {
                    width: 30vw;
                    height: 280px;
                    overflow: hidden;
                    overflow-wrap: break-word;
                    }

                .block-container {
                        padding-top: 0vh;
                    }
                    .blue-text {
                    color: blue;
                    font-family: Arial, sans-serif;
                    font-size: 20px;
                    }

                    .normal-text {
                    color: black;
                    font-family: Arial, sans-serif;
                    font-size: 20px;
                        }
                footer{
                visibility:hidden;
                }

            </style>
            """,
        unsafe_allow_html=True,
    )

    # remove the sidebar pages
    no_sidebar_style = """
        <style>
            div[data-testid="stSidebarNav"] li {display: none;}
        </style>
    """
    # hide the sidebar
    st.markdown(no_sidebar_style, unsafe_allow_html=True)

    ### Custom CSS for the sticky header
    st.markdown(
        """
    <style>
        div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
            position: sticky;
            top: 2.875rem;
            background-color: white;
            z-index: 999;
        }
        .fixed-header {
        }
    </style>
        """,
        unsafe_allow_html=True,
    )

    # fonts for the website
    st.markdown(
        """<style>/* Font */
            @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
            /* You can replace 'Roboto' with any other font of your choice */

            /* Title */
            h1 {
            font-family: 'Roboto', sans-serif;
            font-size: 32px;
            font-weight: 700;
            padding-top:0px;
            }

            /* Chapter Header */
            h2 {
            font-family: 'Roboto', sans-serif;
            font-size: 24px;
            font-weight: 700;
            }

            /* Chapter Subheader */
            h3 {
            font-family: 'Roboto', sans-serif;
            font-size: 20px;
            font-weight: 700;
            }

            /* Normal Text */
            p {
            font-family: 'Roboto', sans-serif;
            font-size: 16px;
            font-weight: 400;
            }
            </style>""",
        unsafe_allow_html=True,
    )

def get_vega_spec(created_chart, select_chart, gpt_response, gpt_input_adjust, Conversation):
    st.session_state.past.append(gpt_input_adjust)
    output = Conversation.run(input=gpt_input_adjust)
        
    st.session_state.generated.append(output)
    gpt_response = st.session_state["generated"][-1]
    print(gpt_response)
    vega_spec = json.loads(gpt_response)
    st.session_state[f"{created_chart[select_chart]}"] = vega_spec

    return vega_spec

# load the data that was selected by the user on previous pages
def handle_data():
    # read in the data
    # dataset_index = of which selection is selected first in the dropdown in
    # the sidebardf
    if st.session_state["dataset"] == "💶 Salaries":
        data_path = "data/ds_salaries.csv"
        st.session_state["data_path"] = data_path
        df = pd.read_csv(data_path)
        df.work_year = df.work_year.apply(lambda x: str(x))
        dataset_index = 1
    elif st.session_state["dataset"] == "🎥 IMDB Movies":
        data_path = "data/imdb_top_1000.csv"
        st.session_state["data_path"] = data_path
        df = pd.read_csv(data_path)
        dataset_index = 0
    elif st.session_state["dataset"] == "📈 Superstore Sales":
        data_path = "data/superstore.csv"
        st.session_state["data_path"] = data_path
        df = pd.read_csv(data_path)
        dataset_index = 2
    elif st.session_state["dataset"] == "😷 Covid Worldwide":
        data_path = "data/covid_worldwide.csv"
        st.session_state["data_path"] = data_path
        df = pd.read_csv(data_path)
        dataset_index = 3
    elif st.session_state["dataset"] == "🗣️ Amazon Customer Behaviour":
        data_path = "data/Amazon Customer Behavior Survey.csv"
        st.session_state["data_path"] = data_path
        df = pd.read_csv(data_path)
        dataset_index = 4
    elif st.session_state["dataset"] == "🧺 Food Prices":
        data_path = "data/Food Prices.csv"
        st.session_state["data_path"] = data_path
        df = pd.read_csv(data_path)
        dataset_index = 5
    elif st.session_state["dataset"] == "🛌 Sleep, Health and Lifestyle":
        data_path = "data/Sleep_health_and_lifestyle_dataset.csv"
        st.session_state["data_path"] = data_path
        df = pd.read_csv(data_path)
        dataset_index = 6
    elif st.session_state["dataset"] == "🎵 Spotify Song Attributes":
        data_path = "data/Spotify_Song_Attributes.csv"
        st.session_state["data_path"] = data_path
        df = pd.read_csv(data_path)
        dataset_index = 7

    # Apply the custom function and convert date columns
    for col in df.columns:
        # check if a column name contains date substring
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col])
            # remove timestamp
            df[col] = df[col].dt.date
            try:
                df[col] = df[col].apply(lambda x: x.strftime("%Y-%m-%d"))
            except:
                print("Error in Date Parsing")

    # replace space with _ in column names
    cols_widget = df.columns
    cols = ", ".join(cols_widget)

    return df, cols, cols_widget, dataset_index


def main():
    """
    Main function for the Data Story Authoring Tool - Homepage.
    Returns:
        None
    """



    # call the style function to apply the styles
    style()

    # close the adjust_mode so that the user can navigate back to their story
    st.session_state["adjust_mode"] = False

    # get the df
    df, cols, cols_widget, dataset_index = handle_data()

    # initialize the model
    Conversation = model_initialisation(
            MODEL="gpt-4",
            TEMPERATURE=0,
            K=3,
            column_names=df.columns.tolist(),
        )

    # add page logo to sidebar
    with st.sidebar:
        add_logo("static/img/chi_logo.png", height=30)

    # page title
    st.title("Data Story Authoring Tool - Adjustment Page")

    
    # 1. Chart Adjustments
    st.subheader("Adjust Charts")

    # get all charts from the session state
    created_charts = [key for key in st.session_state.keys() if key.startswith("fig_gpt") and not key.endswith("_description")]

    print(created_charts)
    
        
    select_chart = st.radio("Which Chart to adjust?",
                            options=[i for i in range(0,len(created_charts))],
                            )
  
    
    st.vega_lite_chart(
        height=350,
        data=df,
        spec=st.session_state[f"{created_charts[select_chart]}"],
    )

    # adjust the chart
    gpt_input_adjust = st.text_input(
                    key="input_viz_adjust",
                    placeholder=(
                        "Adjust the chart via Natural Language Prompts."
                    ),
                    label="input_viz_adjust",
                    label_visibility="collapsed",
                )
    
    # chart option contains values in the form "Graph x"
    gpt_input_adjust = f"Adjust the chart " + gpt_input_adjust

    if st.button("Adjust visualization", key="adjust_visualization", on_click=get_vega_spec,
                 args=(created_charts, select_chart,created_charts[select_chart], gpt_input_adjust, Conversation)):
        print(gpt_input_adjust)
        
        print(gpt_input_adjust)
        # rerun so that the visualization changes
        st.experimental_rerun()

    st.subheader("Adjust Narratives")
    # get all charts from the session state
    created_descriptions = [key for key in st.session_state.keys() if key.startswith("fig_gpt") and key.endswith("_description")]
    
        
    select_descriptions = st.radio("Which Description to adjust?",
                            options=[i for i in range(0,len(created_descriptions))],
                            )
    
    chart_description = st.session_state[f"{created_descriptions[select_descriptions]}"]
    # Use My Self Made Custom Component
    graph_explanation = st_pandas_text_editor(
        columns=df.columns.tolist(),
        key=f"plot_description_{select_descriptions}",
        placeholder="The plot shows...",
        value=chart_description,
    )

    if graph_explanation:
        st.session_state[
                    f"graph_{select_descriptions+1}_text"
                ] = graph_explanation[1] = graph_explanation[1]
        

    st.subheader("Adjust Story Purpose")

    story_purpose = st_pandas_text_editor(
        columns=df.columns.tolist(),
        key=f"story_purpose_{1}_widget",
        placeholder="This story displays ...",
        value=st.session_state[
            f"story_purpose_gpt_{1}"
        ],
    )

    if story_purpose:
        st.session_state[f"story_{1}_confirmed"] = True
        st.session_state[
            f"story_purpose_{1}_text"
        ] = story_purpose[1]
        st.session_state[
            f"story_purpose_{1}_editor_text"
        ] = story_purpose[2]

        


    
    if st.button("Back to Data Story"):
        switch_page("data_story_1")


    

if __name__ == "__main__":
    main()
