"""
The code creates a web application using Streamlit, a Python library for building interactive web apps.
# Author: Anonymous
# Date: June 06, 2023
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

# instanciate gptHelperFunctions
gpt_helper = GPTHelper()

# set the path in deployment
dir = path.Path(__file__).abspath()
sys.path.append(dir.parent.parent)

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

# Initialize session states
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []


def graph_counter():
    if "graph_counter" not in st.session_state:
        st.session_state["graph_counter"] = 1
    return st.session_state["graph_counter"]


def increase_graph_counter():
    st.session_state["graph_counter"] += 1
    print(st.session_state["graph_counter"])


def page_counter():
    if "page_counter" not in st.session_state:
        st.session_state["page_counter"] = 1
    return st.session_state["page_counter"]


def increase_page_counter():
    st.session_state["page_counter"] += 1
    print(st.session_state["page_counter"])


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


def model_initialisation_chart_description(TEMPERATURE, MODEL):
    # custom query template --> possible to add few shot examples in the future
    # add dynamic variables columns and data types to the prompt
    template = """
    You are a great assistant at chart to text tasks.\
    Please describe the following vega lite chart. Your\
    description will be shown on a data story. It should be concise and contain only 4 short bullet points.
    It should also include additional\
    information that is not included in the chart.\
    Try not to explain in a descriptive style but be more user centric.

    Human: {input}
    AI Assistant:"""

    PROMPT = PromptTemplate(input_variables=["input"], template=template)

    # Create an OpenAI instance
    llm = OpenAI(
        temperature=TEMPERATURE,
        openai_api_key=st.secrets["openai_api_key"],
        model_name=MODEL,
        verbose=False,
        streaming=True,
    )

    # Create the ConversationChain object with the specified configuration
    Conversation = LLMChain(
        llm=llm,
        prompt=PROMPT,
    )

    return Conversation


def model_initialisation_story_title(TEMPERATURE, MODEL):
    # custom query template --> possible to add few shot examples in the future
    # add dynamic variables columns and data types to the prompt
    template = """
    You are a great assistant to create interesting titles.\
    Summarize the following text into a 2-3 word long title.

    Human: {input}
    AI Assistant:"""

    PROMPT = PromptTemplate(input_variables=["input"], template=template)

    # Create an OpenAI instance
    llm = OpenAI(
        temperature=TEMPERATURE,
        openai_api_key=st.secrets["openai_api_key"],
        model_name=MODEL,
        verbose=False,
        streaming=True,
    )

    # Create the ConversationChain object with the specified configuration
    Conversation = LLMChain(
        llm=llm,
        prompt=PROMPT,
    )

    return Conversation


def model_initialisation_story_purpose(TEMPERATURE, MODEL):
    # custom query template --> possible to add few shot examples in the future
    # add dynamic variables columns and data types to the prompt
    template = """
    You are a great assistant to create interesting titles and descriptions.\
    Create a  data story title plus a one or two sentence long description from the following text:

    Human: {input}
    AI Assistant:"""

    PROMPT = PromptTemplate(input_variables=["input"], template=template)

    # Create an OpenAI instance
    llm = OpenAI(
        temperature=TEMPERATURE,
        openai_api_key=st.secrets["openai_api_key"],
        model_name=MODEL,
        verbose=False,
        streaming=True,
    )

    # Create the ConversationChain object with the specified configuration
    Conversation = LLMChain(
        llm=llm,
        prompt=PROMPT,
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


# handle the session state callbacks
def change_handler_num_pages():
    st.session_state["num_pages_data_story"] = st.session_state[
        "num_pages_input"
    ]


def change_handler_dataset(data_path):
    st.session_state["dataset"] = st.session_state["dataset_input"]
    if (
        f"multivision_specs_{st.session_state['dataset_input']}"
        not in st.session_state
    ):
        # the thread stores the created vega lite specifications in a
        # session state variable called multivision_specs_{dataset}
        recommender_thread = Recommender(
            num_rec=12,
            data_path=data_path,
            dataset=st.session_state["dataset_input"],
        )
        recommender_thread.run()


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
        df = pd.read_csv(data_path, encoding="windows-1252")
        df["Postal Code"] = df["Postal Code"].apply(lambda x: str(x) + "_")
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


def get_vega_spec():
    # display the code
    gpt_response = st.session_state["generated"][-1]
    print(gpt_response)
    vega_spec = json.loads(gpt_response)

    return vega_spec


def get_low_level_values(nested_dict):
    values = []
    for value in nested_dict.values():
        if isinstance(value, dict):
            values.extend(get_low_level_values(value))
        else:
            values.append(value)
    return values


# deletes the last conversation so that we can go back to the old chart
def handle_undo_changes():
    del st.session_state["generated"][-1]
    del st.session_state["past"][-1]
    del st.session_state.entity_memory.buffer[-1]
    del st.session_state.entity_memory.buffer[-2]


def handle_confirm_viz(current_graph, spec, df):
    with st.spinner("Generating the Story Purpose"):
        # delete the flash message container
        if "created_graph" in st.session_state:
            del st.session_state["created_graph"]
        if "created_page" in st.session_state:
            del st.session_state["created_page"]
        st.session_state[f"visualization_{current_graph}_confirmed"] = True
        # save the vega-lite spec
        st.session_state[f"fig_gpt_{current_graph}"] = spec
        # get the fields that are used for the graph
        used_fields = []
        spec_fields = get_low_level_values(spec)
        try:
            for field in spec_fields:
                if field in df.columns:
                    used_fields.append(field)
            # use only the values that are relevant for this visualization
            df_spec = df[used_fields].sample(10).to_dict(orient="records")
        except:
            df_spec = df.sample(10).to_dict(orient="records")
        # create the spec for gpt to create a description
        description_spec = spec.copy()
        description_spec["data"] = {"values": df_spec}

        # generate the chart description text
        Conversation = model_initialisation_chart_description(
            MODEL="gpt-4",
            TEMPERATURE=1,
        )
        # save the chart description
        st.session_state[
            f"fig_gpt_{current_graph}_description"
        ] = Conversation.run(input=description_spec)


def main():
    """
    Main function for the Data Story Authoring Tool - Create Visualizations.

    Returns:
        None
    """

    # create a container to place in sticky header content
    header = st.container()
    with header:
        # top page navigation bar
        choose = option_menu(
            "StoryPoint",
            [
                "Homepage",
                "Data Exploration",
                "Story Composition",
                "Story Narration",
                "Data Story",
            ],
            icons=[
                "house",
                "clipboard-data",
                "list-check",
                "bar-chart",
                "award",
                "send-check",
            ],
            menu_icon="app-indicator",
            default_index=3,
            key="visualization-menu",
            orientation="horizontal",
            styles={
                "container": {
                    "padding": "0!important",
                    "background-color": "#FFFFFF",
                },
                "icon": {"color": "orange", "font-size": "16px"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "#1A84C7"},
            },
        )
        # delete the other session states so when we navigate back to the respective
        # pages, we dont get endless loops
        if "story-menu" in st.session_state:
            del st.session_state["story-menu"]
        if "exploration-menu" in st.session_state:
            del st.session_state["exploration-menu"]
        if "layout-menu" in st.session_state:
            del st.session_state["layout-menu"]
        if "homepage-menu" in st.session_state:
            del st.session_state["homepage-menu"]
        # handle the option that got chosen in the navigation bar
        if choose == "Data Exploration":
            switch_page("Exploratory Data Analysis")
        elif choose == "Story Composition":
            switch_page("Layout Creation")
        elif choose == "Homepage":
            switch_page("Homepage")
        elif choose == "Data Story":
            switch_page("Data Story 1")
        st.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)

    # call the style function to apply the styles
    style()

    # use the handleData method
    df, cols, cols_widget, dataset_index = handle_data()

    # streamlit create counter
    current_page = page_counter()

    # streamlit graph counter
    current_graph = graph_counter()

    # add page logo to sidebar
    with st.sidebar:
        add_logo("static/img/chi_logo.png", height=30)

    st.sidebar.write("### Your Dataset contains the following features")
    with st.sidebar.expander("Dataset Features", expanded=True):
        nl = "\n".join(df.columns)
        st.write(
            f"""
                 \n{nl}"""
        )

    # another sidebar header
    with st.sidebar:
        st.subheader("Configure the Chat Model")
    # Set up sidebar with various options
    with st.sidebar.expander("🛠️ Adjust Chatbot Settings", expanded=True):
        MODEL = st.selectbox(
            label="Model",
            options=[
                "gpt-3.5-turbo",
                "gpt-4",
                "text-davinci-003",
                "text-davinci-002",
                "code-davinci-002",
            ],
        )
        K = st.number_input(
            " (#)Summary of prompts to consider", min_value=3, max_value=1000
        )
        TEMPERATURE = st.slider(
            "Creativity of the Model", 0.0, 1.0, step=0.1, value=0.0
        )

    with st.sidebar:
        gpt_helper.feedback(page=choose)

    # Set up the Streamlit app layout
    st.title("Data Story Authoring Tool - Visualizations")

    # explanation text
    st.write(
        "This is the visualization creator page of the data story authoring tool.\
              Here, you will sequentially create the graphs for your data story. \
             For each page in your data story, you will be prompted to enter a Story Purpose\
              by typing it into the story purpose text editor. Afterwards, you will use an \
             Open AI Large Language Model to create Vega Lite visualizations through Natural \
             Language input. For each created visualization, you will also be prompted to add \
             explaining text to it. Make sure that the explaining text contains information that goes beyond the \
             information that the viewer of the story can get from the visualization alone. \
             Additionally, at the top of the page, you can also choose a set of filters for each page of the data story."
    )
    st.write(
        f"###### Currently creating page {current_page} - Graph"
        f' {current_graph}/{st.session_state["num_pages_data_story"]*2}'
    )
    st.write(f'###### Chosen Dataset: {st.session_state["dataset"]}')
    st.write("***")

    # show further headings on one side and the datastory on the other side
    c1, c2 = st.columns([2, 2])

    with c1:
        # when reloading the page because of saving the graph, we keep the selected
        # filters for the page
        if f"filter_choice_{current_page}" in st.session_state:
            pre_selected = st.session_state[f"filter_choice_{current_page}"]
        else:
            pre_selected = cols_widget[0]

        # let the user choose the filters to be used on the current story page
        st.write(
            "Here you can select global filters for your Data Story. Once the Data Story is created, the filters\
                 will appear in the sidebar. There will also be a clear filter button to unapply them."
        )
        st.subheader(
            "1️⃣Choose a set of filters that can be applied on the charts"
        )
        options = st.multiselect(
            "Filter Choice",
            cols_widget,
            pre_selected,
            help="Choose a set of Filters that you can use for the dashboard on the next page",
            key=f"filter_choice_{current_page}_widget",
        )

    with c2:
        # which layout was chosen by the user for the current page
        page_layout = st.session_state[f"page_layout_{current_page-1}_entered"]
        if page_layout == "Image 1":
            img_path = "static/img/DataStory State"
        elif page_layout == "Image 2":
            img_path = "static/img/DataStory2 State"

        with st.expander(
            expanded=True, label=f"Data Story Progress of Page {current_page}"
        ):
            # display the data story's state
            # second graph is finished
            if f"fig_gpt_{(current_page*2)}" in st.session_state:
                image = Image.open(f"{img_path}/Folie5.PNG")
                st.image(image)
            # first text is finished
            elif f"graph_{(current_page*2) - 1}_text" in st.session_state:
                image = Image.open(f"{img_path}/Folie4.PNG")
                st.image(image)
            # first graph is finished
            elif f"fig_gpt_{(current_page*2) - 1}" in st.session_state:
                image = Image.open(f"{img_path}/Folie3.PNG")
                st.image(image)
            # story purpose is given
            elif f"story_purpose_{current_page}_text" in st.session_state:
                image = Image.open(f"{img_path}/Folie2.PNG")
                st.image(image)
            # no story purpose is given
            else:
                image = Image.open(f"{img_path}/Folie1.PNG")
                st.image(image)

    # make space
    # add_vertical_space(2)
    # give feedback when first page was created
    if "created_page" in st.session_state:
        html(
            """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Flash Message Example</title>
            <!-- Add jQuery library -->
            <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
            <style>
                /* Style for the flash message container */
                #flash {
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    background-color: #4CAF50; /* Green background color */
                    padding: 10px;
                    text-align: center;
                    display: none; /* hide initially */
                    font-size: 24px; /* bigger font size */
                    font-weight: bold; /* bold text */
                    color: #000000; /* Black font color */
                }
            </style> 
        </head>
        <body> 
            <div id="flash">Page saved, continue with the rest.</div>

            <script>
                $(function() { 
                    // Show and hide the flash message
                    $('#flash').delay(500).fadeIn('normal', function() {
                        $(this).delay(2500).fadeOut();
                    });
                }); 
            </script>
        </body>
        </html>

        """,
            height=50,
        )

    # example usage:
    # st.markdown(eval(f'f"""{st.session_state[f"story_purpose_{current_page}_text"]}"""'), unsafe_allow_html=True)

    # which number of visualization
    if current_graph == 1:
        st.subheader("2️⃣ Create the 1st visualization")
    elif current_graph == 2:
        st.subheader(f"2️⃣ Create the 2nd visualization")
    elif current_graph == 3:
        st.subheader(f"2️⃣ Create the 3rd visualization")
    else:
        st.subheader(f"2️⃣ Create the {current_graph}th visualization")

    # give feedback when first graph was created
    if "created_graph" in st.session_state:
        html(
            """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Flash Message Example</title>
            <!-- Add jQuery library -->
            <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
            <style>
                /* Style for the flash message container */
                #flash {
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    background-color: #4CAF50; /* Green background color */
                    padding: 10px;
                    text-align: center;
                    display: none; /* hide initially */
                    font-size: 24px; /* bigger font size */
                    font-weight: bold; /* bold text */
                    color: #000000; /* Black font color */
                }
            </style> 
        </head>
        <body> 
            <div id="flash">The graph has been created and saved!</div>

            <script>
                $(function() { 
                    // Show and hide the flash message
                    $('#flash').delay(500).fadeIn('normal', function() {
                        $(this).delay(2500).fadeOut();
                    });
                }); 
            </script>
        </body>
        </html>

        """,
            height=50,
        )

    st.write(
        "Create visualizations via Natural Language Prompts or get inspired by example visualizations\
                in the kickstart tab."
    )

    tab1, tab2 = st.tabs(
        [
            "Use Large Language Model",
            "Kickstart with example visualization",
        ]
    )

    with tab1:
        # initialize the model
        Conversation = model_initialisation(
            MODEL=MODEL,
            TEMPERATURE=TEMPERATURE,
            K=K,
            column_names=df.columns.tolist(),
        )

        # use chat GPT to write Code
        gpt_input = st.text_input(
            key="input_viz",
            placeholder=(
                "Briefly explain what you want to plot from your data. For example:"
                " Plot the average salary per year"
            ),
            label=(
                "💡Use GPT to help generating the code for the visualizations. Refer to the help symbol for ideas. "
            ),
            help=f"""# The dataframe has the following columns:
            \n{[str(column) for column in df.columns]}\n
            Possible prompts:\n
            - Make a Scatterplot of <column x> and <column y>
            - Create an ordered PieChart of ...
            - Create a bar chart for the distribution of ...""",
        )

        if "json_decode_error" in st.session_state:
            del st.session_state["json_decode_error"]
            html(
                """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Flash Message Example</title>
                <!-- Add jQuery library -->
                <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
                <style>
                    /* Style for the flash message container */
                    #flash {
                        position: fixed;
                        top: 0;
                        left: 0;
                        width: 100%;
                        background-color: #ff3333; /* Green background color */
                        padding: 10px;
                        text-align: left;
                        display: none; /* hide initially */
                        font-size: 12px; /* bigger font size */
                        font-weight: bold; /* bold text */
                        color: #FFFFFF; /* Black font color */
                    }
                </style> 
            </head>
            <body> 
                <div id="flash">Error, please give the model more specification.</div>

                <script>
                    $(function() { 
                        // Show and hide the flash message
                        $('#flash').delay(500).fadeIn('normal', function() {
                            $(this).delay(2500).fadeOut();
                        });
                    }); 
                </script>
            </body>
            </html>

            """,
                height=30,
            )

        if st.button("Commit Prompt"):
            # for development environment: measure time it takes for API request
            start_time = time.time()
            with st.spinner("Busy API Servers (5-10 seconds) ...."):
                output = Conversation.run(input=gpt_input)
                st.session_state.past.append(gpt_input)
                st.session_state.generated.append(output)
                st.session_state[f"prompt_commited_{current_graph}"] = True
            # for development environment: measure time it takes for API request
            end_time = time.time()
            execution_time = end_time - start_time
            print("It took " + str(round(execution_time, 2)) + "seconds")

    # implement a carousel to show the visualizations created by multivision
    with tab2:
        st.write(
            "The visualizations are created by the visualization recommendation framework\
                [MultiVision](https://arxiv.org/pdf/2107.07823.pdf) by Wu et al. (2021). \
                    Select a visualization from the list below and adjust it further in the next steps.\
                    If no visualization is shown, that means, that you're dataset is not suitable for the\
                    algorithm."
        )
        viz_container = st.container()
        with st.spinner("Loading Visualizations"):
            # create the dir if it doesn't exist
            directory = f"static/img/VegaSpecs/{st.session_state['dataset']}"
            os.makedirs(directory, exist_ok=True)
            if (
                f"spec_imgs_created_{st.session_state['dataset']}"
                not in st.session_state
            ):
                # prepare the data to be added to the vega specs
                multivision_specs = []
                data = df.dropna().iloc[0:30].to_dict(orient="records")
                # add the data key to the dict
                for i, multivision_spec in enumerate(
                    st.session_state[
                        f"multivision_specs_{st.session_state['dataset']}"
                    ]
                ):
                    # create a copy so that the original reference is not overwritten
                    multivision_spec_copy = multivision_spec.copy()
                    multivision_spec_copy["data"] = {"values": data}
                    multivision_specs.append(multivision_spec_copy)
                    # for debugging
                    print(multivision_spec_copy)
                    # convert every spec into a png
                    png_data = vlc.vegalite_to_png(
                        vl_spec=multivision_spec_copy, scale=2
                    )

                    # numbers should be two digits long for the lexicographical ordering to work
                    if i <= 9:
                        with open(
                            f"static/img/VegaSpecs/{st.session_state['dataset']}/spec_0{i}.png",
                            "wb",
                        ) as f:
                            f.write(png_data)
                    if i >= 10:
                        with open(
                            f"static/img/VegaSpecs/{st.session_state['dataset']}/spec_{i}.png",
                            "wb",
                        ) as f:
                            f.write(png_data)
                # add to the session state
                st.session_state[
                    f"spec_imgs_created_{st.session_state['dataset']}"
                ] = True

        # collect all image files from the folder
        imageUrls = []

        for file in os.listdir(
            f"static/img/VegaSpecs/{st.session_state['dataset']}"
        ):
            with open(
                f"static/img/VegaSpecs/{st.session_state['dataset']}/" + file,
                "rb",
            ) as image:
                encoded = base64.b64encode(image.read()).decode()
                imageUrls.append(f"data:image/png;base64,{encoded}")

        # create the component with the code from the frontend folder
        imageCarouselComponent = components.declare_component(
            "image-carousel-component", path="frontend/public"
        )

        selectedImageUrl = imageCarouselComponent(
            imageUrls=imageUrls, height=300
        )

        if selectedImageUrl is not None:
            # st.image(selectedImageUrl[0])
            # index of the vega lite spec from multivision
            index = selectedImageUrl[1]
            viz_container.success(
                f'Visualization number {index+1} selected, scroll down and click "Select as Kickstarter Template" to continue'
            )
            # get the spec json without the data attribute
            multivision_spec = st.session_state[
                f"multivision_specs_{st.session_state['dataset']}"
            ][index]

        # confirm the visuakization and send the vega lite spec without the data attribute
        if st.button("Select as Kickstarter Template"):
            # artificially add the vega lite spec to the gpt responses
            st.session_state.past.append(
                "Create a nice vega_lite visualization."
            )
            st.session_state.generated.append(
                str(multivision_spec)
                .replace("'", '"')
                .replace("True", "true")
                .replace("False", "false")
            )
            # also add the answer to the entity memory
            st.session_state.entity_memory.save_context(
                {"input": "Create a nice vega_lite visualization."},
                {
                    "output": str(
                        st.session_state[
                            f"multivision_specs_{st.session_state['dataset']}"
                        ][index]
                    )
                    .replace("'", '"')
                    .replace("True", "true")
                    .replace("False", "false")
                },
            )
            # go to the next step
            st.session_state[f"prompt_commited_{current_graph}"] = True

    if f"prompt_commited_{current_graph}" in st.session_state:
        # display the success message later on
        container = st.empty()

        st.subheader("3️⃣Choose one of the Graphs and adjust it")

        try:
            # vega specs
            vega_spec = get_vega_spec()
        except Exception as e:
            # when gpt returns an empty answer
            st.session_state[
                "json_decode_error"
            ] = "Please write something more specific."
            del st.session_state[f"prompt_commited_{current_graph}"]
            st.experimental_rerun()

        # make two views for example (Coding Expert, Business user)
        tab1, tab2 = st.tabs(["Business user", "Coding Expert"])

        with tab1:
            # charts and their explanation
            c1, _, c2 = st.columns([4, 1, 8])
            with c1:
                # give the user the possibility to adjust the plot
                st.write("###### 2. Adjust the chart if needed")

                gpt_input_adjust = st.text_input(
                    key="input_viz_adjust",
                    placeholder=(
                        "Give the plot ... color, add the plot title ..."
                    ),
                    label="input_viz_adjust",
                    label_visibility="collapsed",
                )

                with st.expander("Expand for Examples"):
                    st.write(
                        """
                            - Add the Plot Title *PlotTitle*
                            - Change the y-axis label to *yAxisLable*
                            - Use *FeatureX* on the x-Axis
                            - Use *FeatureX* as a color gradient
                            - Make it a Scatterplot instead
                            - Use timeUnit year --> only shows year on xaxis without months
                            - to group a bar chart, prompt: use 'xOffset':{'field':'<grouping field>'} within encoding
                            - make the information hoverable by including variables into the tooltip
                            - use aggregate by mean to get mean values for an axis
                            - use transform calculation to calculate the deaths divided by population 
                            """
                    )

                # give the information, which plot shall be adjusted
                # chart option contains values in the form "Graph x"
                gpt_input_adjust = f"Adjust the chart " + gpt_input_adjust

            with c2:
                # display the chart
                # render the vega lite chart that came as response
                # use only the values that are relevant for this visualization
                df_spec = df.head(100).to_dict(orient="records")
                # create the spec for gpt to create a description
                vega_spec_copy = vega_spec.copy()
                vega_spec_copy["data"] = {"values": df_spec}
                # by parsing the json string of the response
                st.vega_lite_chart(
                    height=320,
                    data=df,
                    spec=vega_spec,
                    use_container_width=True,
                )

            # new columns for prettier layout with the two buttons
            c1, _, c2 = st.columns([4, 1, 8])
            with c1:
                if st.button(
                    "Adjust visualization", key="adjust_visualization"
                ):
                    print(gpt_input_adjust)
                    output = Conversation.run(input=gpt_input_adjust)
                    st.session_state.past.append(gpt_input_adjust)
                    st.session_state.generated.append(output)
                    print(gpt_input_adjust)
                    # rerun so that the visualization changes
                    st.experimental_rerun()
            with c2:
                col1, col2 = st.columns([5, 2])
                # disable the button if the plot has not been adjusted by the user yet
                if len(st.session_state["generated"]) > 1:
                    st.session_state["button_disabled"] = False
                else:
                    st.session_state["button_disabled"] = True
                with col1:
                    st.button(
                        "Undo last Changes",
                        key="undo_last_changes",
                        on_click=handle_undo_changes,
                        disabled=st.session_state["button_disabled"],
                    )
                with col2:
                    # let user confirm the visualization
                    confirm_visualization = st.button(
                        "Confirm visualization",
                        key="confirm_visualization",
                        on_click=handle_confirm_viz,
                        args=(current_graph, vega_spec, df),
                    )

        with tab2:
            # charts and their explanation
            c1, _, c2 = st.columns([4, 1, 8])
            with c1:
                # give the user the possibility to adjust the plot
                st.write("###### 2. Adjust the chart if needed")
                content = st_ace(
                    language="json5", key="code-editor-one", value=vega_spec
                )
                if content:
                    print("content")

            with c2:
                # display the chart that was selected in the chart_option selectbox
                try:
                    # render the vega lite chart that came as response
                    # by parsing the json string of the response
                    st.vega_lite_chart(
                        height=320,
                        data=df,
                        spec=vega_spec,
                    )
                except Exception as e:
                    st.write(e)

            # new columns for prettier layout with the two buttons
            c1, _, c2 = st.columns([4, 1, 8])
            with c1:
                if st.button(
                    "Adjust visualization",
                    key="adjust_visualization_coding_expert",
                ):
                    # append the changed visualization from the Ace editor called content
                    # to the chatGPT conversation
                    vega_spec = get_vega_spec()

                    # append it to the response
                    st.session_state.generated.append(vega_spec)
                    st.experimental_rerun()
            with c2:
                # let user confirm the visualization
                confirm_viz = st.button(
                    "Confirm visualization",
                    key="confirm_visualization_coding_expert",
                    on_click=handle_confirm_viz,
                    args=(current_graph, vega_spec, df),
                )

    if f"visualization_{current_graph}_confirmed" in st.session_state:
        # DP2
        st.subheader("4️⃣ Describe the plot and give further information")
        # use the chart description from chatGPT
        if f"fig_gpt_{current_graph}_description" in st.session_state:
            chart_description = st.session_state[
                f"fig_gpt_{current_graph}_description"
            ]
            # Use My Self Made Custom Component
            graph_explanation = st_pandas_text_editor(
                columns=df.columns.tolist(),
                key=f"plot_description_{current_graph}",
                placeholder="The plot shows...",
                value=chart_description,
            )
        else:
            # Use My Self Made Custom Component
            graph_explanation = st_pandas_text_editor(
                columns=df.columns.tolist(),
                key=f"plot_description_{current_graph}",
                placeholder="The plot shows...",
                value=chart_description,
            )
        if graph_explanation:
            if f"graph_{current_graph}_confirmed" not in st.session_state:
                st.session_state[f"graph_{current_graph}_confirmed"] = True
                st.session_state[
                    f"graph_{current_graph}_text"
                ] = graph_explanation[1]

        # only go further when text for the story is entered
        if f"graph_{current_graph}_confirmed" in st.session_state:
            # save the chosen filters
            st.session_state[
                f"filter_choice_{current_page}"
            ] = st.session_state[f"filter_choice_{current_page}_widget"]
            # this means, that we have the last graph and want to create the story
            # now
            if current_graph == st.session_state["num_pages_data_story"] * 2:
                # let the user input the story purpose
                if current_page == 1:
                    st.subheader(
                        f"5️⃣Describe the story purpose of the 1st page"
                    )
                elif current_page == 2:
                    st.subheader(
                        f"5️⃣Describe the story purpose of the 2nd page"
                    )
                elif current_page == 3:
                    st.subheader(
                        f"5️⃣Describe the story purpose of the 3rd page"
                    )
                else:
                    st.subheader(
                        f"5️⃣Describe the story purpose of the {current_page}th page"
                    )
                # generate the chart description text
                Conversation = model_initialisation_story_purpose(
                    MODEL="gpt-4",
                    TEMPERATURE=1,
                )
                # build the gpt query string from the former chart descriptions
                story_purpose_prompt = f"""
                    {st.session_state[f"fig_gpt_{current_graph}_description"]}


                    {st.session_state[f"fig_gpt_{current_graph-1}_description"]}
                """

                # generate the story purpose via chatgpt
                if f"story_purpose_gpt_{current_page}" not in st.session_state:
                    with st.spinner("Generating the Story Purpose"):
                        st.session_state[
                            f"story_purpose_gpt_{current_page}"
                        ] = Conversation.run(input=story_purpose_prompt)

                    # Use My Self Made Custom Component
                    story_purpose = st_pandas_text_editor(
                        columns=df.columns.tolist(),
                        key=f"story_purpose_{current_page}_widget",
                        placeholder="This story displays ...",
                        value=st.session_state[
                            f"story_purpose_gpt_{current_page}"
                        ],
                    )
                else:
                    # Use My Self Made Custom Component
                    story_purpose = st_pandas_text_editor(
                        columns=df.columns.tolist(),
                        key=f"story_purpose_{current_page}_widget",
                        placeholder="This story displays ...",
                        value=st.session_state[
                            f"story_purpose_gpt_{current_page}"
                        ],
                    )

                if story_purpose:
                    st.session_state[f"story_{current_page}_confirmed"] = True
                    st.session_state[
                        f"story_purpose_{current_page}_text"
                    ] = story_purpose[1]
                    st.session_state[
                        f"story_purpose_{current_page}_editor_text"
                    ] = story_purpose[2]

                    # generate the page title
                    page_title_generator = model_initialisation_story_title(
                        MODEL="gpt-4",
                        TEMPERATURE=1,
                    )
                    st.session_state[
                        f"page_{current_page}_title"
                    ] = page_title_generator.run(
                        st.session_state[f"story_purpose_{current_page}_text"]
                    )

                    # finish the story
                    if st.button(
                        "✅ Finish the Data Story", key="finished_story"
                    ):
                        st.session_state["first_graph"] = True
                        # delete entity memory to start a new conversation with chat model
                        del st.session_state["generated"]
                        del st.session_state["past"]
                        del st.session_state["entity_memory"]
                        # create a state for a finished data story
                        st.session_state["finished_data_story"] = True
                        switch_page("data story 1")
            else:
                # this means, that the page isnt complete yet
                if current_graph % 2 == 1:
                    # finish this template
                    if st.button("✅ Finish this graph", key="finished_graph"):
                        st.session_state["created_graph"] = True
                        st.session_state["first_graph"] = False
                        increase_graph_counter()
                        # delete entity memory to start a new conversation with chat model
                        del st.session_state["generated"]
                        del st.session_state["past"]
                        del st.session_state["entity_memory"]

                        switch_page("create visualizations")

                # this means, that one page of the story is complete
                elif current_graph % 2 == 0:
                    # let the user input the story purpose
                    if current_page == 1:
                        st.subheader(
                            f"5️⃣Describe the story purpose of the 1st page"
                        )
                    elif current_page == 2:
                        st.subheader(
                            f"5️⃣Describe the story purpose of the 2nd page"
                        )
                    elif current_page == 3:
                        st.subheader(
                            f"5️⃣Describe the story purpose of the 3rd page"
                        )
                    else:
                        st.subheader(
                            f"5️⃣Describe the story purpose of the {current_page}th page"
                        )
                    # generate the chart description text
                    Conversation = model_initialisation_story_purpose(
                        MODEL="gpt-4",
                        TEMPERATURE=1,
                    )
                    # build the gpt query string from the former chart descriptions
                    story_purpose_prompt = f"""
                        {st.session_state[f"fig_gpt_{current_graph}_description"]}


                        {st.session_state[f"fig_gpt_{current_graph-1}_description"]}
                    """

                    # generate the story purpose via chatgpt
                    if (
                        f"story_purpose_gpt_{current_page}"
                        not in st.session_state
                    ):
                        with st.spinner("Generating the Story Purpose"):
                            st.session_state[
                                f"story_purpose_gpt_{current_page}"
                            ] = Conversation.run(input=story_purpose_prompt)

                        # Use My Self Made Custom Component
                        story_purpose = st_pandas_text_editor(
                            columns=df.columns.tolist(),
                            key=f"story_purpose_{current_page}_widget",
                            placeholder="This story displays ...",
                            value=st.session_state[
                                f"story_purpose_gpt_{current_page}"
                            ],
                        )
                    else:
                        # Use My Self Made Custom Component
                        story_purpose = st_pandas_text_editor(
                            columns=df.columns.tolist(),
                            key=f"story_purpose_{current_page}_widget",
                            placeholder="This story displays ...",
                            value=st.session_state[
                                f"story_purpose_gpt_{current_page}"
                            ],
                        )

                    if story_purpose:
                        st.session_state[
                            f"story_{current_page}_confirmed"
                        ] = True
                        st.session_state[
                            f"story_purpose_{current_page}_text"
                        ] = story_purpose[1]
                        st.session_state[
                            f"story_purpose_{current_page}_editor_text"
                        ] = story_purpose[2]

                        # generate the page title
                        page_title_generator = (
                            model_initialisation_story_title(
                                MODEL="gpt-4",
                                TEMPERATURE=1,
                            )
                        )
                        st.session_state[
                            f"page_{current_page}_title"
                        ] = page_title_generator.run(
                            st.session_state[
                                f"story_purpose_{current_page}_text"
                            ]
                        )

                        # finish this template
                        if st.button(
                            "✅ Finish this page", key="finished_page"
                        ):
                            st.session_state["first_graph"] = True
                            st.session_state["created_page"] = True
                            # delete entity memory to start a new conversation with chat model
                            del st.session_state["generated"]
                            del st.session_state["past"]
                            del st.session_state["entity_memory"]
                            increase_page_counter()
                            increase_graph_counter()
                            switch_page("create visualizations")


def handle_new_number_pages():
    # set the new page number
    st.session_state["num_pages_data_story"] = st.session_state[
        "increase_num_pages"
    ]
    # set the page lyout fix to the first layout --> adjust after prototype
    current_page = st.session_state["page_counter"]
    st.session_state[f"page_layout_{current_page}_entered"] = "Image 1"
    # create the new story
    gpt_helper.create_story_layout_type_1(
        file_name=f"pages/0{3+current_page}_data_story_{current_page+1}.py",
        story_page=current_page + 1,
    )
    # increase the counters for the data story
    increase_page_counter()
    increase_graph_counter()
    # delete the session state variable to show the page from the main method again
    del st.session_state["finished_data_story"]


def finished_data_story():
    # create a container to place in sticky header content
    header = st.container()
    with header:
        # top page navigation bar
        choose = option_menu(
            "StoryPoint",
            [
                "Homepage",
                "Data Exploration",
                "Story Composition",
                "Story Narration",
                "Data Story",
            ],
            icons=[
                "house",
                "clipboard-data",
                "list-check",
                "bar-chart",
                "award",
                "send-check",
            ],
            menu_icon="app-indicator",
            default_index=3,
            key="visualization-menu",
            orientation="horizontal",
            styles={
                "container": {
                    "padding": "0!important",
                    "background-color": "#FFFFFF",
                },
                "icon": {"color": "orange", "font-size": "16px"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "#1A84C7"},
            },
        )
        # delete the other session states so when we navigate back to the respective
        # pages, we dont get endless loops
        if "story-menu" in st.session_state:
            del st.session_state["story-menu"]
        if "exploration-menu" in st.session_state:
            del st.session_state["exploration-menu"]
        if "layout-menu" in st.session_state:
            del st.session_state["layout-menu"]
        if "homepage-menu" in st.session_state:
            del st.session_state["homepage-menu"]
        # handle the option that got chosen in the navigation bar
        if choose == "Data Exploration":
            switch_page("Exploratory Data Analysis")
        elif choose == "Story Composition":
            switch_page("Layout Creation")
        elif choose == "Homepage":
            switch_page("Homepage")
        elif choose == "Data Story":
            switch_page("Data Story 1")
        st.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)

    # call the style function to apply the styles
    style()

    # add page logo to sidebar
    with st.sidebar:
        add_logo("static/img/chi_logo.png", height=30)

    st.subheader(
        "The data story has been created and can be found under the Data Story Tab"
    )
    st.write(
        "If you want to create further pages, increase the number of the pages variable"
    )
    num_pages = st.number_input(
        "\# of pages in data story",
        value=st.session_state["num_pages_data_story"] + 1,
        min_value=st.session_state["num_pages_data_story"] + 1,
        key="increase_num_pages",
    )
    increase_num_pages_button = st.button(
        "Confirm new Number of Pages", on_click=handle_new_number_pages
    )


if __name__ == "__main__":
    # when the data story is finished, we want a different page to be shown here
    if "finished_data_story" in st.session_state:
        finished_data_story()
    else:
        main()
