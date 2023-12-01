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
                    help="Select a Range of values. Only raw data that lies in                                    between that range will now be considered for the visualizations.",
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
            st.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)

            # use the breadcrumbs to navigate through the created data story
            breadcrumb_html = gpt_helper.breadcrumbs(current_url="data story 1")
            # display the breadcrumbs
            st.components.v1.html(breadcrumb_html, height=80)

        with st.sidebar:
            gpt_helper.feedback(page=choose)

    # The data Story content
    # Story purpose
    st.markdown(
        eval(f'f"""{st.session_state[f"story_purpose_1_text"]}"""'),
        unsafe_allow_html=True,
    )

    # first visualization and text
    c1, c2 = st.columns([4, 3])

    with c1:
        st.vega_lite_chart(
            height=350,
            data=df,
            spec=st.session_state["fig_gpt_1"],
            use_container_width=True,
        )

    with c2:
        st.markdown(
            eval(f'f"""{st.session_state[f"graph_1_text"]}"""'),
            unsafe_allow_html=True,
        )

    # second chart and text
    col1, col2 = st.columns([4, 3])

    with col1:
        st.vega_lite_chart(
            height=350,
            data=df,
            spec=st.session_state["fig_gpt_2"],
            use_container_width=True,
        )

    with col2:
        st.markdown(
            eval(f'f"""{st.session_state[f"graph_2_text"]}"""'),
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()

