import streamlit as st  # web development
import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import time  # to simulate a real time data, time loop
import sqlite3
from sqlite3 import Connection
import openai
import re
from dateutil.parser import parse
import os
from string import digits
import glob
import textwrap
from streamlit.components.v1 import html
import time
from feedback_component import feedback_component
import json


class GPTHelper:
    def feedback(self, page):
        # display success message
        if "feedback_submitted" in st.session_state:
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
                        background-color: #1A84C7; /* Green background color */
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
                <div id="flash">Saved, thank you for the feedback!</div>

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
            del st.session_state["feedback_submitted"]

        # styles for the icon
        st.markdown(
            """
            <style> 
                .icon_feedback {
                position: absolute; /* Absolute positioning */
                width: 32px; /* Icon width */
                height: 32px; /* Icon height */
                }
            </style>
        """,
            unsafe_allow_html=True,
        )
        feedback = feedback_component(
            my_input_value=1,
            key=f"feedback_return_value{st.session_state['feedback_counter']}",
        )
        # feedback form expander:
        if (
            f"feedback_return_value{st.session_state['feedback_counter']}"
            in st.session_state
        ):
            if st.session_state[
                f"feedback_return_value{st.session_state['feedback_counter']}"
            ]:
                with st.form("Feedback Form"):
                    category = st.selectbox(
                        "Choose a category:",
                        ["General", "Layout", "Colors"],
                    )
                    comment = st.text_area("Your comment:")
                    rating = st.slider(
                        "Rate between 1 and 5:", min_value=1, max_value=5
                    )

                    submitted = st.form_submit_button("Submit")

                    if submitted:
                        feedback_data = {
                            "category": category,
                            "comment": comment,
                            "rating": rating,
                            "position": feedback[1],
                        }
                        # save the feedback
                        # def save_feedback(feedback_data):
                        # Load existing data if it exists
                        try:
                            # Load existing data if it exists
                            with open(f"feedback/{page}.json", "r") as f:
                                existing_data = json.load(f)
                        except (FileNotFoundError, json.JSONDecodeError):
                            # If the file does not exist or is empty, initialize with an empty list
                            existing_data = []

                        # Append the new feedback_data to existing data
                        existing_data.append(feedback_data)

                        # Save the updated data back to the file
                        with open(f"feedback/{page}.json", "w") as f:
                            json.dump(existing_data, f, indent=4)

                        # upgrade the counter
                        st.session_state["feedback_counter"] += 1
                        html(
                            """
                        <script>
                        function deleteDivContainer() {
                            const containerElement = parent.window.document.getElementById("feedback_component_div");
                            if (containerElement) {
                            containerElement.remove();
                            } else {
                            console.error('Div container with ID "container" not found.');
                            }
                        }

                        // Call the function to delete the div container
                        deleteDivContainer();
                        </script>""",
                            height=0,
                        )
                        # sleep so the html above has enough time
                        time.sleep(0.1)
                        st.session_state["feedback_submitted"] = True
                        st.experimental_rerun()

    # def extract_code(self, gpt_response):
    #     print(gpt_response)
    #     """function to extract code and sql query from gpt response"""
    #     if "```" in gpt_response:
    #         # extract text between ``` and ```
    #         pattern = r"```(.*?)```"
    #         extracted_code = re.findall(pattern, gpt_response, re.DOTALL)
    #         extracted_code = [code.replace("python", "") for code in extracted_code]
    #         extracted_code_string = ""
    #         for code in extracted_code:
    #             extracted_code_string += code

    #         return extracted_code_string
    #     else:
    #         return gpt_response

    # how many pages do we have created already
    def get_breadcrumb_data():
        # folder path (select only the data story files)
        dir_path = r"./pages/"
        # Use glob to find all files matching the pattern '*.py'
        file_paths = glob.glob(os.path.join(dir_path, "*.py"))
        # Filter the file paths to include only those containing 'data_story'
        data_story_paths = [
            path.replace("\\", "/")
            for path in file_paths
            if "data_story" in path
        ]

        # all pages to be included in the breadcrumbs will be stored in this list
        breadcrumb_data = []
        # Iterate directory
        for path in data_story_paths:
            base_path = os.path.basename(path)
            # check if current path is a file
            # file name in the form xx_text.py
            if os.path.isfile(path):
                # check if it is a data story file
                remove_digits = str.maketrans("", "", digits)
                # Assuming the label is the file name without the extension
                url = os.path.splitext(base_path)[0].replace("_", " ")[3:]
                label = os.path.splitext(base_path)[0].replace("_", " ")[2:]
                breadcrumb_data.append(
                    {"url": url, "label": label, "link_name": url}
                )
        return breadcrumb_data

    def breadcrumbs(self, current_url):
        # design and show the breadcrumbs in the custom streamlit component
        style = """<style>
        .breadcrumb {
        display: flex;
        align-items: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 14px;
        }

        .breadcrumb a {
        color: #0079c1;
        text-decoration: none;
        transition: color 0.3s;
        position: relative;
        }

        .breadcrumb a:hover {
        color: #005ba1;
        cursor: pointer;
        }

        .breadcrumb a::before {
        content: "";
        position: absolute;
        bottom: -2px;
        left: 0;
        width: 100%;
        height: 2px;
        background-color: #0079c1;
        visibility: hidden;
        transform: scaleX(0);
        transition: all 0.3s ease-in-out;
        }

        .breadcrumb a:hover::before {
        visibility: visible;
        transform: scaleX(1);
        }

        .breadcrumb-item {
        display: inline-flex;
        justify-content: center;
        align-items: center;
        padding: 12px 24px;
        background-color: #f2f2f2;
        border-radius: 40px;
        margin-right: 10px;
        font-weight: bold;
        color: #555;
        text-transform: uppercase;
        letter-spacing: 1px;
        }

        .breadcrumb-separator {
        margin: 0 5px;
        color: #ccc;
        }

        .current-page {
        background-color: #f69c55;
        color: #fff;
        }

        </style>"""

        # Get the breadcrumb data dynamically from the backend
        breadcrumb_data = (
            GPTHelper.get_breadcrumb_data()
        )  # Replace this with your own logic to fetch the breadcrumb data

        # Create the breadcrumb HTML string
        breadcrumb_html = """
        {style}
        <header>
        <div class="breadcrumb" kind="header">
        {breadcrumb_items}
        </div>
        </header>
        """

        breadcrumb_items = ""

        # Iterate over the breadcrumb data and generate HTML for each breadcrumb item
        for counter, item in enumerate(breadcrumb_data):
            # current page
            if item["url"] == current_url:
                breadcrumb_item_html = '<div class="breadcrumb-item current-page">{label}</div>'.format(
                    label=st.session_state[f"page_{counter+1}_title"]
                )
            else:
                breadcrumb_item_html = """
                <script>
                    function goTo(page) {
                        page_links = parent.document.querySelectorAll('[data-testid="stSidebarNav"] ul li a')
                        page_links.forEach((link) => {     
                            if (link.text == page) {
                                link.click()
                            }
                        })
                    }
                </script>
                <div class="breadcrumb-item"><a onclick="goTo('%s')">%s</a><br /></div>""" % (
                    item["link_name"],
                    st.session_state[f"page_{counter+1}_title"],
                )
            breadcrumb_items += breadcrumb_item_html

            # Add breadcrumb separator unless it's the last item
            if item != breadcrumb_data[-1]:
                breadcrumb_items += (
                    '<span class="breadcrumb-separator">></span>'
                )

        # Format the breadcrumb HTML string with the generated breadcrumb items
        breadcrumb_html = breadcrumb_html.format(
            breadcrumb_items=breadcrumb_items, style=style
        )

        return breadcrumb_html

    def create_story_layout_type_1(self, file_name: str, story_page: int):
        """
        Dynamically creates a python file for each story page that is requested by the user.

                Parameters:
                        filename "String": Name of the Data Story
                        story_page (int): Which page number

                Returns:
                        Nothing
        """
        filepath = os.getcwd()
        temp_path = filepath + file_name
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(
                textwrap.dedent(
                    '''\
                # streamlit packages
                import streamlit as st
                from streamlit_extras.badges import badge  # for git
                from streamlit_extras.switch_page_button import switch_page
                from streamlit_extras.app_logo import add_logo
                from streamlit_option_menu import option_menu
                import  streamlit_toggle as tog

                # handle AskData Feature
                from langchain.llms import OpenAI
                from langchain.agents.agent_types import AgentType
                from langchain.agents import create_pandas_dataframe_agent

                # dataframe handling
                import pandas as pd
                import math

                # reusable functions, outsourced into another file
                from helper_functions import GPTHelper


                # instanciate gptHelperFunctions
                gpt_helper = GPTHelper()

                # configure the page
                st.set_page_config(
                    page_title="Conversational Dashboard",
                    page_icon="✅",
                    layout="wide"
                    # initial_sidebar_state="collapsed"
                )

                # set styling format for numbers returned by pandas dataframe
                pd.options.display.float_format = "{:.2f}".format  # round to 2 decimal places

                if "fullscreen_toggle" not in st.session_state:
                    st.session_state["fullscreen_toggle"] = False
                                    
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
                    st.markdown("""<style>/* Font */
                            @import url(\'https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap\');
                            /* You can replace \'Roboto\' with any other font of your choice */

                            /* Title */
                            h1 {
                            font-family: \'Roboto\', sans-serif;
                            font-size: 32px;
                            font-weight: 700;
                            padding-top:0px;
                            }

                            /* Chapter Header */
                            h2 {
                            font-family: \'Roboto\', sans-serif;
                            font-size: 24px;
                            font-weight: 700;
                            }

                            /* Chapter Subheader */
                            h3 {
                            font-family: \'Roboto\', sans-serif;
                            font-size: 20px;
                            font-weight: 700;
                            }

                            /* Normal Text */
                            p {
                            font-family: \'Roboto\', sans-serif;
                            font-size: 16px;
                            font-weight: 400;
                            }
                            </style>""", unsafe_allow_html=True)

                # load the data that was selected by the user on previous pages
                def handle_data():
                    # read in the data
                    # dataset_index = of which selection is selected first in the dropdown in
                    # the sidebar
                    if st.session_state["dataset"] == "💶 Salaries":
                        data_path = "data/ds_salaries.csv"
                        df = pd.read_csv(data_path)
                        df.work_year = df.work_year.apply(lambda x: str(x))
                        st.session_state["dataset_index"] = 1
                    elif st.session_state["dataset"] == "🎥 IMDB Movies":
                        data_path = "data/imdb_top_1000.csv"
                        df = pd.read_csv(data_path)
                        st.session_state["dataset_index"] = 0
                    elif st.session_state["dataset"] == "📈 Superstore Sales":
                        data_path = "data/superstore.csv"
                        df = pd.read_csv(data_path, encoding="windows-1252")
                        df["Postal Code"] = df["Postal Code"].apply(lambda x: str(x) + "_")
                        st.session_state["dataset_index"] = 2
                    elif st.session_state["dataset"] == "😷 Covid Worldwide":
                        data_path = "data/covid_worldwide.csv"
                        df = pd.read_csv(data_path)
                        st.session_state["dataset_index"] = 3
                    elif st.session_state["dataset"] == "🗣️ Amazon Customer Behaviour":
                        data_path = "data/Amazon Customer Behavior Survey.csv"
                        df = pd.read_csv(data_path)
                        st.session_state["dataset_index"] = 4
                    elif st.session_state["dataset"] == "🧺 Food Prices":
                        data_path = "data/Food Prices.csv"
                        df = pd.read_csv(data_path)
                        st.session_state["dataset_index"] = 5
                    elif st.session_state["dataset"] == "🛌 Sleep, Health and Lifestyle":
                        data_path = "data/Sleep_health_and_lifestyle_dataset.csv"
                        df = pd.read_csv(data_path)
                        st.session_state["dataset_index"] = 6
                    elif st.session_state["dataset"] == "🎵 Spotify Song Attributes":
                        data_path = "data/Spotify_Song_Attributes.csv"
                        df = pd.read_csv(data_path)
                        st.session_state["dataset_index"] = 7

                    # Apply the custom function and convert date columns
                    for col in df.columns:
                        # check if a column name contains date substring
                        if "date" in col.lower():
                            df[col] = pd.to_datetime(df[col])
                            # remove timestamp
                            # df[col] = df[col].dt.date


                    # save the dataframe into session state if not already done
                    if "raw_data_filtered" not in st.session_state:
                        st.session_state["raw_data_filtered"] = df
                    if "raw_data_unfiltered" not in st.session_state:
                        st.session_state["raw_data_unfiltered"] = df


                def handle_slider_filter_change(i, filter, num_total_filters):
                    """
                    This function collects all the filters created by the user
                    and stores them in a dynamic filter setting session state variable.
                    Therefore, all filters can be applied simultaneously.
                    """
                    # load max and min value
                    min = st.session_state[f"filter_{i}"][0]
                    max = st.session_state[f"filter_{i}"][1]

                    # save the filters to apply them later alltogether
                    st.session_state[f"slider_setting_{i}"] = [filter, min, max]

                    connect_filter_settings(num_total_filters)


                def handle_multiselect_filter_change(i, filter, num_total_filters):
                    # which parts are selected
                    selected_values = st.session_state[f"filter_{i}"]
                    # save the filters to apply them later alltogether
                    st.session_state[f"multiselect_setting_{i}"] = [filter, selected_values]

                    connect_filter_settings(num_total_filters)


                def connect_filter_settings(num_total_filters):
                    """
                    handle_slider_filter_change and handle_multiselect_filter_change create their
                    respective filter conditions and store them in the variables slider_setting_{i}
                    and multiselect_setting_{i}.
                    The connect_filter_settings function takes those filter conditions and applies them
                    alltogether on an unfiltered dataframe.
                    After filtering the unfiltered df coming from the session state variable raw_data_unfiltered,
                    a filtered df will be returned into the session state variable raw_data_filtered
                    """
                    # load the unfiltered df from session state
                    df_unfiltered = st.session_state["raw_data_unfiltered"]
                    # apply the filters
                    df_filtered = df_unfiltered.copy()
                    # collect all the saved filters:
                    slider_filter_list = []
                    for i in range(num_total_filters):
                        if f"slider_setting_{i}" in st.session_state:
                            slider_filter_list.append(st.session_state[f"slider_setting_{i}"])

                    # collect all the saved filters:
                    multiselect_filter_list = []
                    for i in range(num_total_filters):
                        if f"multiselect_setting_{i}" in st.session_state:
                            multiselect_filter_list.append(st.session_state[f"multiselect_setting_{i}"])

                    # apply the slider filters alltogether on the df
                    for f in slider_filter_list:
                        df_filtered = df_filtered[
                            (df_filtered[f[0]] >= f[1]) & (df_filtered[f[0]] <= f[2])
                        ]
                    # apply the multiselect filters alltogether on the df
                    for f in multiselect_filter_list:
                        df_filtered = df_filtered[df_filtered[f[0]].isin(f[1])]

                    # push the filtered df to session state
                    st.session_state["raw_data_filtered"] = df_filtered


                def handle_clear_filters(num_total_filters):
                    if num_total_filters == 0:
                        print("No filters active")
                    else:
                        # delete all filters
                        for i in range(num_total_filters):
                            if f"multiselect_setting_{i}" in st.session_state:
                                del st.session_state[f"multiselect_setting_{i}"]
                            elif f"slider_setting_{i}" in st.session_state:
                                del st.session_state[f"slider_setting_{i}"]
                        # reset the filters on the df
                        st.session_state["raw_data_filtered"] = st.session_state["raw_data_unfiltered"]


                def main():
                    """
                    Main function for the Data Story Authoring Tool - Data Story.

                    Returns:
                        None
                    """
                    # go to adjustment page
                    if st.session_state["adjust_mode"] == True:
                        switch_page("adjust_story")

                    # call the style function to apply the styles
                    style()

                    # use the handleData method to store the dataframe in the session state
                    # we need it in session state so that filters can work
                    handle_data()
                    # st.session_state["raw_data"] = st.session_state["raw_data_unfiltered"]
                    # load the raw_data
                    df = st.session_state["raw_data_filtered"]

                    # unfiltered dataframe so the filter boundaries arent adjusted
                    df_slider = st.session_state["raw_data_unfiltered"]

                    # add page logo to sidebar
                    with st.sidebar:
                        add_logo("static/img/chi_logo.png", height=30)
                                    
                    # fullscreen button
                    with st.sidebar:
                        if st.button("Adjust Story"):
                            st.session_state["adjust_mode"] = True

                    with st.sidebar:
                        tog.st_toggle_switch(label="Fullscreen", 
                        key="fullscreen_toggle", 
                        default_value=False, 
                        label_after = True, 
                        inactive_color = '#D3D3D3', 
                        active_color="#11567f", 
                        track_color="#29B5E8"
                        )

                    # construct the filters that were chosen on the create visualizations page
                    with st.sidebar.expander("🌪️ Filter ", expanded=True):
                        for i, filter in enumerate(st.session_state[f"filter_choice_1"]):
                            num_number_filters = []
                            num_obj_filters = []
                            num_total_filters = len(st.session_state[f"filter_choice_1"])
                            # create filters according to the data type of the variable
                            if (
                                df[filter].dtype == "int64"
                                or df[filter].dtype == "int32"
                                or df[filter].dtype == "int"
                                or df[filter].dtype == "float64"
                                or df[filter].dtype == "float32"
                                or df[filter].dtype == "float"
                            ):
                                # increase counter for number filters
                                num_number_filters.append(i)
                                st.slider(
                                    min_value=float(df_slider[filter].min()),
                                    value=(float(df_slider[filter].min()), float(df_slider[filter].max())),
                                    label=f"{filter} Range Slider",
                                    help="Select a Range of values. Only raw data that lies in\
                                    between that range will now be considered for the visualizations.",
                                    args=(i, filter, num_total_filters),
                                    on_change=handle_slider_filter_change,
                                    key=f"filter_{i}",
                                )
                            if df[filter].dtype == "O":
                                # increase number for object filters
                                num_obj_filters.append(i)
                                st.multiselect(
                                    f"Select {filter}",
                                    df[filter].unique(),
                                    help="Select all elements that should be displayed in the graph.",
                                    on_change=handle_multiselect_filter_change,
                                    args=(i, filter, num_total_filters),
                                    key=f"filter_{i}",
                                )
                        # option to clear the filters
                        clear_filters = st.button(
                            "Clear Filters",
                            on_click=handle_clear_filters,
                            args=(num_total_filters,),
                            key="clear_filters",
                        )

                    # sidebar expander as a help page
                    with st.sidebar.expander("📈🔍 AskData", expanded=True):
                        st.write("Here you can query the data by using Text Input.")
                        # create an agent that knows my pandas data frame
                        agent = create_pandas_dataframe_agent(OpenAI(temperature=0,model_name="gpt-4"), df, verbose=True)
                        # let the user ask questions
                        akd_data_query = st.text_area(
                            label="Query the data",
                            label_visibility="collapsed",
                            placeholder="What is the average value for ... in the ... category?",
                        )
                        if st.button("send query"):
                            st.write(agent.run(akd_data_query))

                    # create a container to place in sticky header content
                    header = st.container()
                                    
                    if st.session_state["fullscreen_toggle"] == False:
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
                                default_index=4,
                                key="story-menu",
                                orientation="horizontal",
                                styles={
                                    "container": {"padding": "0!important", "background-color": "#FFFFFF"},
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
                            if "homepage-menu" in st.session_state:
                                del st.session_state["homepage-menu"]
                            if "exploration-menu" in st.session_state:
                                del st.session_state["exploration-menu"]
                            if "layout-menu" in st.session_state:
                                del st.session_state["layout-menu"]
                            if "visualization-menu" in st.session_state:
                                del st.session_state["visualization-menu"]
                            # handle the option that got chosen in the navigation bar
                            if choose == "Data Exploration":
                                switch_page("Exploratory Data Analysis")
                            elif choose == "Story Composition":
                                switch_page("Layout Creation")
                            elif choose == "Story Narration":
                                switch_page("Create Visualizations")
                            elif choose == "Homepage":
                                switch_page("Homepage")
                            st.write("""<div class=\'fixed-header\'/>""", unsafe_allow_html=True)

                            # use the breadcrumbs to navigate through the created data story
                            breadcrumb_html = gpt_helper.breadcrumbs(current_url="data story %i")
                            # display the breadcrumbs
                            st.components.v1.html(breadcrumb_html, height=80)

                        with st.sidebar:
                            gpt_helper.feedback(page=choose)

                    # The data Story content
                    # Story purpose
                    st.markdown(
                        eval(f\'f"""{st.session_state[f"story_purpose_%i_text"]}"""\'),
                        unsafe_allow_html=True,
                    )

                    # first visualization and text
                    c1, c2 = st.columns([4, 3])

                    with c1:
                        st.vega_lite_chart(
                            height=350,
                            data=df,
                            spec=st.session_state["fig_gpt_%i"],
                            use_container_width=True,
                        )

                    with c2:
                        st.markdown(
                            eval(f\'f"""{st.session_state[f"graph_%i_text"]}"""\'),
                            unsafe_allow_html=True,
                        )

                    # second chart and text
                    col1, col2 = st.columns([4, 3])

                    with col1:
                        st.vega_lite_chart(
                            height=350,
                            data=df,
                            spec=st.session_state["fig_gpt_%i"],
                            use_container_width=True,
                        )

                    with col2:
                        st.markdown(
                            eval(f\'f"""{st.session_state[f"graph_%i_text"]}"""\'),
                            unsafe_allow_html=True,
                        )


                if __name__ == "__main__":
                    main()
        
                '''
                    % (
                        story_page,
                        story_page,
                        story_page * 2 - 1,
                        story_page * 2 - 1,
                        story_page * 2,
                        story_page * 2,
                    )
                )
            )

    def create_story_layout_type_2(self, file_name: str, story_page: int):
        """
        Dynamically creates a python file for each story page that is requested by the user.

                Parameters:
                        filename "String": Name of the Data Story
                        story_page (int): Which page number

                Returns:
                        Nothing
        """
        filepath = os.getcwd()
        temp_path = filepath + file_name
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(
                textwrap.dedent(
                    '''\
                    # streamlit packages
                    import streamlit as st
                    from streamlit_extras.badges import badge  # for git
                    from streamlit_extras.switch_page_button import switch_page
                    from streamlit_extras.app_logo import add_logo
                    from streamlit_option_menu import option_menu
                    import  streamlit_toggle as tog

                    # handle AskData Feature
                    from langchain.llms import OpenAI
                    from langchain.agents.agent_types import AgentType
                    from langchain.agents import create_pandas_dataframe_agent

                    # dataframe handling
                    import pandas as pd
                    import math

                    # reusable functions, outsourced into another file
                    from helper_functions import GPTHelper


                    # instanciate gptHelperFunctions
                    gpt_helper = GPTHelper()

                    # configure the page
                    st.set_page_config(
                        page_title="Conversational Dashboard",
                        page_icon="✅",
                        layout="wide"
                        # initial_sidebar_state="collapsed"
                    )

                    # set styling format for numbers returned by pandas dataframe
                    pd.options.display.float_format = "{:.2f}".format  # round to 2 decimal places

                    if "fullscreen_toggle" not in st.session_state:
                        st.session_state["fullscreen_toggle"] = False
                                        
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
                        st.markdown("""<style>/* Font */
                                @import url(\'https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap\');
                                /* You can replace \'Roboto\' with any other font of your choice */

                                /* Title */
                                h1 {
                                font-family: \'Roboto\', sans-serif;
                                font-size: 32px;
                                font-weight: 700;
                                padding-top:0px;
                                }

                                /* Chapter Header */
                                h2 {
                                font-family: \'Roboto\', sans-serif;
                                font-size: 24px;
                                font-weight: 700;
                                }

                                /* Chapter Subheader */
                                h3 {
                                font-family: \'Roboto\', sans-serif;
                                font-size: 20px;
                                font-weight: 700;
                                }

                                /* Normal Text */
                                p {
                                font-family: \'Roboto\', sans-serif;
                                font-size: 16px;
                                font-weight: 400;
                                }
                                </style>""", unsafe_allow_html=True)

                    # load the data that was selected by the user on previous pages
                    def handle_data():
                        # read in the data
                        # dataset_index = of which selection is selected first in the dropdown in
                        # the sidebar
                        if st.session_state["dataset"] == "💶 Salaries":
                            data_path = "data/ds_salaries.csv"
                            df = pd.read_csv(data_path)
                            df.work_year = df.work_year.apply(lambda x: str(x))
                            st.session_state["dataset_index"] = 1
                        elif st.session_state["dataset"] == "🎥 IMDB Movies":
                            data_path = "data/imdb_top_1000.csv"
                            df = pd.read_csv(data_path)
                            st.session_state["dataset_index"] = 0
                        elif st.session_state["dataset"] == "📈 Superstore Sales":
                            data_path = "data/superstore.csv"
                            df = pd.read_csv(data_path, encoding="windows-1252")
                            df["Postal Code"] = df["Postal Code"].apply(lambda x: str(x) + "_")
                            st.session_state["dataset_index"] = 2
                        elif st.session_state["dataset"] == "😷 Covid Worldwide":
                            data_path = "data/covid_worldwide.csv"
                            df = pd.read_csv(data_path)
                            st.session_state["dataset_index"] = 3
                        elif st.session_state["dataset"] == "🗣️ Amazon Customer Behaviour":
                            data_path = "data/Amazon Customer Behavior Survey.csv"
                            df = pd.read_csv(data_path)
                            st.session_state["dataset_index"] = 4
                        elif st.session_state["dataset"] == "🧺 Food Prices":
                            data_path = "data/Food Prices.csv"
                            df = pd.read_csv(data_path)
                            st.session_state["dataset_index"] = 5
                        elif st.session_state["dataset"] == "🛌 Sleep, Health and Lifestyle":
                            data_path = "data/Sleep_health_and_lifestyle_dataset.csv"
                            df = pd.read_csv(data_path)
                            st.session_state["dataset_index"] = 6
                        elif st.session_state["dataset"] == "🎵 Spotify Song Attributes":
                            data_path = "data/Spotify_Song_Attributes.csv"
                            df = pd.read_csv(data_path)
                            st.session_state["dataset_index"] = 7

                        # Apply the custom function and convert date columns
                        for col in df.columns:
                            # check if a column name contains date substring
                            if "date" in col.lower():
                                df[col] = pd.to_datetime(df[col])
                                # remove timestamp
                                # df[col] = df[col].dt.date


                        # save the dataframe into session state if not already done
                        if "raw_data_filtered" not in st.session_state:
                            st.session_state["raw_data_filtered"] = df
                        if "raw_data_unfiltered" not in st.session_state:
                            st.session_state["raw_data_unfiltered"] = df


                    def handle_slider_filter_change(i, filter, num_total_filters):
                        """
                        This function collects all the filters created by the user
                        and stores them in a dynamic filter setting session state variable.
                        Therefore, all filters can be applied simultaneously.
                        """
                        # load max and min value
                        min = st.session_state[f"filter_{i}"][0]
                        max = st.session_state[f"filter_{i}"][1]

                        # save the filters to apply them later alltogether
                        st.session_state[f"slider_setting_{i}"] = [filter, min, max]

                        connect_filter_settings(num_total_filters)


                    def handle_multiselect_filter_change(i, filter, num_total_filters):
                        # which parts are selected
                        selected_values = st.session_state[f"filter_{i}"]
                        # save the filters to apply them later alltogether
                        st.session_state[f"multiselect_setting_{i}"] = [filter, selected_values]

                        connect_filter_settings(num_total_filters)


                    def connect_filter_settings(num_total_filters):
                        """
                        handle_slider_filter_change and handle_multiselect_filter_change create their
                        respective filter conditions and store them in the variables slider_setting_{i}
                        and multiselect_setting_{i}.
                        The connect_filter_settings function takes those filter conditions and applies them
                        alltogether on an unfiltered dataframe.
                        After filtering the unfiltered df coming from the session state variable raw_data_unfiltered,
                        a filtered df will be returned into the session state variable raw_data_filtered
                        """
                        # load the unfiltered df from session state
                        df_unfiltered = st.session_state["raw_data_unfiltered"]
                        # apply the filters
                        df_filtered = df_unfiltered.copy()
                        # collect all the saved filters:
                        slider_filter_list = []
                        for i in range(num_total_filters):
                            if f"slider_setting_{i}" in st.session_state:
                                slider_filter_list.append(st.session_state[f"slider_setting_{i}"])

                        # collect all the saved filters:
                        multiselect_filter_list = []
                        for i in range(num_total_filters):
                            if f"multiselect_setting_{i}" in st.session_state:
                                multiselect_filter_list.append(st.session_state[f"multiselect_setting_{i}"])

                        # apply the slider filters alltogether on the df
                        for f in slider_filter_list:
                            df_filtered = df_filtered[
                                (df_filtered[f[0]] >= f[1]) & (df_filtered[f[0]] <= f[2])
                            ]
                        # apply the multiselect filters alltogether on the df
                        for f in multiselect_filter_list:
                            df_filtered = df_filtered[df_filtered[f[0]].isin(f[1])]

                        # push the filtered df to session state
                        st.session_state["raw_data_filtered"] = df_filtered


                    def handle_clear_filters(num_total_filters):
                        if num_total_filters == 0:
                            print("No filters active")
                        else:
                            # delete all filters
                            for i in range(num_total_filters):
                                if f"multiselect_setting_{i}" in st.session_state:
                                    del st.session_state[f"multiselect_setting_{i}"]
                                elif f"slider_setting_{i}" in st.session_state:
                                    del st.session_state[f"slider_setting_{i}"]
                            # reset the filters on the df
                            st.session_state["raw_data_filtered"] = st.session_state["raw_data_unfiltered"]


                    def main():
                        """
                        Main function for the Data Story Authoring Tool - Data Story.

                        Returns:
                            None
                        """
                        # go to adjustment page
                        if st.session_state["adjust_mode"] == True:
                            switch_page("adjust_story")


                        # call the style function to apply the styles
                        style()

                        # use the handleData method to store the dataframe in the session state
                        # we need it in session state so that filters can work
                        handle_data()
                        # st.session_state["raw_data"] = st.session_state["raw_data_unfiltered"]
                        # load the raw_data
                        df = st.session_state["raw_data_filtered"]

                        # unfiltered dataframe so the filter boundaries arent adjusted
                        df_slider = st.session_state["raw_data_unfiltered"]

                        # add page logo to sidebar
                        with st.sidebar:
                            add_logo("static/img/chi_logo.png", height=30)
                                        
                        # fullscreen button
                        with st.sidebar:
                            if st.button("Adjust Story"):
                                st.session_state["adjust_mode"] = True
                        with st.sidebar:
                            tog.st_toggle_switch(label="Fullscreen", 
                            key="fullscreen_toggle", 
                            default_value=False, 
                            label_after = True, 
                            inactive_color = '#D3D3D3', 
                            active_color="#11567f", 
                            track_color="#29B5E8"
                            )

                        # construct the filters that were chosen on the create visualizations page
                        with st.sidebar.expander("🌪️ Filter ", expanded=True):
                            for i, filter in enumerate(st.session_state[f"filter_choice_1"]):
                                num_number_filters = []
                                num_obj_filters = []
                                num_total_filters = len(st.session_state[f"filter_choice_1"])
                                # create filters according to the data type of the variable
                                if (
                                    df[filter].dtype == "int64"
                                    or df[filter].dtype == "int32"
                                    or df[filter].dtype == "int"
                                    or df[filter].dtype == "float64"
                                    or df[filter].dtype == "float32"
                                    or df[filter].dtype == "float"
                                ):
                                    # increase counter for number filters
                                    num_number_filters.append(i)
                                    st.slider(
                                min_value=float(df_slider[filter].min()),
                                value=(float(df_slider[filter].min()), float(df_slider[filter].max())),
                                        label=f"{filter} Range Slider",
                                        help="Select a Range of values. Only raw data that lies in\
                                        between that range will now be considered for the visualizations.",
                                        args=(i, filter, num_total_filters),
                                        on_change=handle_slider_filter_change,
                                        key=f"filter_{i}",
                                    )
                                if df[filter].dtype == "O":
                                    # increase number for object filters
                                    num_obj_filters.append(i)
                                    st.multiselect(
                                        f"Select {filter}",
                                        df[filter].unique(),
                                        help="Select all elements that should be displayed in the graph.",
                                        on_change=handle_multiselect_filter_change,
                                        args=(i, filter, num_total_filters),
                                        key=f"filter_{i}",
                                    )
                            # option to clear the filters
                            clear_filters = st.button(
                                "Clear Filters",
                                on_click=handle_clear_filters,
                                args=(num_total_filters,),
                                key="clear_filters",
                            )

                        # sidebar expander as a help page
                        with st.sidebar.expander("📈🔍 AskData", expanded=True):
                            st.write("Here you can query the data by using Text Input.")
                            # create an agent that knows my pandas data frame
                            agent = create_pandas_dataframe_agent(OpenAI(temperature=0,model_name="gpt-4"), df, verbose=True)
                            # let the user ask questions
                            akd_data_query = st.text_area(
                                label="Query the data",
                                label_visibility="collapsed",
                                placeholder="What is the average value for ... in the ... category?",
                            )
                            if st.button("send query"):
                                st.write(agent.run(akd_data_query))

                        # create a container to place in sticky header content
                        header = st.container()
                                        
                        if st.session_state["fullscreen_toggle"] == False:
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
                                    default_index=4,
                                    key="story-menu",
                                    orientation="horizontal",
                                    styles={
                                        "container": {"padding": "0!important", "background-color": "#FFFFFF"},
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
                                if "homepage-menu" in st.session_state:
                                    del st.session_state["homepage-menu"]
                                if "exploration-menu" in st.session_state:
                                    del st.session_state["exploration-menu"]
                                if "layout-menu" in st.session_state:
                                    del st.session_state["layout-menu"]
                                if "visualization-menu" in st.session_state:
                                    del st.session_state["visualization-menu"]
                                # handle the option that got chosen in the navigation bar
                                if choose == "Data Exploration":
                                    switch_page("Exploratory Data Analysis")
                                elif choose == "Story Composition":
                                    switch_page("Layout Creation")
                                elif choose == "Story Narration":
                                    switch_page("Create Visualizations")
                                elif choose == "Homepage":
                                    switch_page("Homepage")
                                st.write("""<div class=\'fixed-header\'/>""", unsafe_allow_html=True)

                                # use the breadcrumbs to navigate through the created data story
                                breadcrumb_html = gpt_helper.breadcrumbs(current_url="data story %i")
                                # display the breadcrumbs
                                st.components.v1.html(breadcrumb_html, height=80)

                            with st.sidebar:
                                gpt_helper.feedback(page=choose)

                        # The data Story content
                        # Story purpose
                        st.markdown(
                            eval(f\'f"""{st.session_state[f"story_purpose_%i_text"]}"""\'),
                            unsafe_allow_html=True,
                        )

                        # first visualization and text
                        c1, c2 = st.columns([3, 3])

                        with c1:
                            st.vega_lite_chart(
                                height=350,
                                data=df,
                                spec=st.session_state["fig_gpt_%i"],
                                use_container_width=True,
                            )

                        with c2:
                            st.vega_lite_chart(
                                height=350,
                                data=df,
                                spec=st.session_state["fig_gpt_%i"],
                                use_container_width=True,
                            )

                        # second chart and text
                        col1, col2 = st.columns([3, 3])

                        with col1:
                            st.markdown(
                                eval(f\'f"""{st.session_state[f"graph_%i_text"]}"""\'),
                                unsafe_allow_html=True,
                            )


                        with col2:
                            st.markdown(
                                eval(f\'f"""{st.session_state[f"graph_%i_text"]}"""\'),
                                unsafe_allow_html=True,
                            )


                    if __name__ == "__main__":
                        main()
            
                    '''
                    % (
                        story_page,
                        story_page,
                        story_page * 2 - 1,
                        story_page * 2,
                        story_page * 2 - 1,
                        story_page * 2,
                    )
                )
            )
