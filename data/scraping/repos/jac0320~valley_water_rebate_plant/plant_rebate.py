import streamlit as st
import pandas as pd
from math import ceil
import os
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.agents import AgentType
from langchain.agents import create_pandas_dataframe_agent
from langchain.callbacks import StreamlitCallbackHandler
import webbrowser

st.set_page_config(layout='wide')


@st.cache_data
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')


def clear_wishlist():
    st.session_state["wishlist"] = {}


def add_to_wishlist(plant_index):
    st.session_state["wishlist"][plant_index] = 1


def remove_from_wishlist(plant_index):
    if plant_index in st.session_state.wishlist:
        st.session_state["wishlist"].pop(plant_index)


def search_google(keyword):
    webbrowser.open_new_tab(f"https://www.google.com/search?q={keyword}")


def clear_history():
    st.session_state['chat_dialogue'] = [{"role": "system"}]


def render_grid(df, key=""):

    if not df.empty:

        controls = st.columns(3)
        with controls[0]:
            batch_size = st.select_slider("Batch size:",range(0,51,5), value=5, key=f"{key}_batch_size")
        with controls[1]:
            row_size = st.select_slider("Row size:", range(1,16), value=5, key=f"{key}_row_size")
        num_batches = ceil(len(df)/batch_size)
        with controls[2]:
            page = st.selectbox("Page", range(1,num_batches+1), key=f"{key}_page")

        batch = df[(page-1)*batch_size : page*batch_size]
        # st.dataframe(batch, use_container_width=True)
        
        grid = st.columns(row_size)
        col = 0

        for idx, plant in batch.to_dict(orient='index').items():

            with grid[col]:

                if key == "selected":
                    button_cols = st.columns([0.7, 0.2, 0.1])
                else:
                    button_cols = st.columns([0.8, 0.2])

                button_cols[0].button(
                    f"{plant['Scientific_Name']} - {plant['Plant_Name']} - {plant['Coverage']} sqft", 
                    key=f"{key}_google_{idx}",
                    on_click=search_google, 
                    args=[f"{plant['Scientific_Name']}"], 
                    type="primary", 
                    use_container_width=True
                )

                if key == "selected":
                    st.session_state.wishlist[idx] = button_cols[1].number_input("Quantity", key=f"{key}_wishlist_qty_{idx}", min_value=1, value=1, label_visibility="collapsed")
                    button_cols[2].button(
                        "🗑️",
                        key=f"{key}_cart_{idx}",
                        on_click=remove_from_wishlist,
                        args=[idx],
                        type="secondary",
                        use_container_width=True
                    )
                else:
                    button_cols[1].button(
                        "🛒",
                        key=f"{key}_cart_{idx}",
                        on_click=add_to_wishlist,
                        args=[idx],
                        type="secondary",
                        use_container_width=True
                    )

                if len(plant['source']) == 0:
                    st.write(f"😢 No free picture was found. Click the button to Google for pics.")
                else:
                    for i, _ in plant['source'].items():
                        try:
                            st.image(plant['source'].get(i), use_column_width=True, caption=plant['source'].get(i))
                        except Exception as err:
                            st.write(f"☹️ error loading image due to {err}")

                st.divider()

            col = (col + 1) % row_size


def help_doc():

    intro_text = """
            This app is to help non-botanist navigate your garden's plant selection for [Valley Water Landscape Rebate Program](https://valleywater.dropletportal.com/).
            There is a total of 2700+ plants names available. Each plants maps to a specific ground coverage and your goal is to select the favorite 
                combinations so that it can cover 50% of the area for you land conversion rebate.
            
            If you are a garden newbie like me who have no idea how most plants look/grow/maintain 😵‍💫, the 2700 plants names will throw you off. The app is 
                    built to simplify the process of plant selection for non-. You can use this app to:
            - 👀 View/Search the plants (of your interest) with pictures 🖼️
            - 🛒 Collect the candidate plants for your garden wish list
            - 🤖 Chat with an Landscape Professional bot                 
        """
    
    bot_text = """
            The 🤖 knows what's happening including your selection and available plants database. You can ask:
                    
            * What about the plants selected for my garden?
            * How tall X(plant) will grow?
            * Are my selection pet safe?
            * Will X(plant) attract pest? If so, what type of pest will it attract?
            * What are the typical price with the plants I selected?
            * Which season will my selected plants bloom/mature?
            * How should I water X(plant)?
            * ...
        """
    if st.session_state.start_app:
        with st.expander("🤔 What is this APP About? 👉"):
            st.markdown(intro_text)
            st.divider()
            st.markdown(bot_text)
    else:
        st.markdown(intro_text)
        st.divider()
        st.markdown(bot_text)


def render_chatbot():

    for message in st.session_state.chat_dialogue:
        if message["role"] == "system":
            continue
        with st.chat_message(message["role"]):
            try:
                st.markdown(message["content"].content)
            except:
                st.markdown(message["content"])

    chat_model = ChatOpenAI(
        model_name='gpt-3.5-turbo',
        temperature=0.7,
        streaming=True,
        openai_api_key=st.session_state.openai_api_key
    )

    selected = st.session_state.orig_df.iloc[list(st.session_state["wishlist"].keys())]

    template = f""" You are a helpful landscape professional who help select plants for my drought-proof garden project in California.
    Currently, the user have selected '{','.join(selected['Scientific_Name'])}' as a combination. You should help provide 
    helpful guidance upon asked. While answering questions, you want to comprehensively consider the garden design, plant combination, 
    planting tips, overall budget for a successful garden project at northern california. Try to be percise with your answer. Try to 
    answer in short sentences. The dataframe I will be passing are available plants in this program for selection. I can continue to add 
    them into my selection.
    """

    if template != st.session_state.chat_dialogue[0].get('content', ''):  # Reset dialogue when user updates the cart
        st.session_state.chat_dialogue = [{"role": "system", "content": template}]

    pandas_df_agent = create_pandas_dataframe_agent(
        chat_model,
        st.session_state.orig_df,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        handle_parsing_errors=True,
    )

    if prompt := st.chat_input(placeholder="Ask your question to your landscope AI"):
        st.session_state.chat_dialogue.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            try:
                response = pandas_df_agent.run(st.session_state.chat_dialogue, callbacks=[st_cb])
            except Exception as err:
                fallback_dialogue = []
                for i in st.session_state.chat_dialogue:
                    if i['role'] == 'system':
                        fallback_dialogue.append({"role": "system", "content": SystemMessage(content=i['content'])})
                    elif i['role'] == 'user':
                        fallback_dialogue.append({"role": "system", "content": HumanMessage(content=i['content'])})
                    elif i['role'] == 'assistant':
                        fallback_dialogue.append({"role": "system", "content": AIMessage(content=i['content'])})
                response = chat_model([i["content"] for i in fallback_dialogue]).content

            st.session_state.chat_dialogue.append({"role": "assistant", "content": response})
            st.write(response)

def render_app():

    st.title("🌿🌵 Plant Selector | Valley Water Rebate Program 💐🌾")

    if 'wishlist' not in st.session_state:
        st.session_state['wishlist'] = {}

    if 'chat_dialogue' not in st.session_state:
        st.session_state['chat_dialogue'] = [{"role": "system"}]

    if 'chat_dialogue_display' not in st.session_state:
        st.session_state['chat_dialogue_display'] = []

    if ('start_app' not in st.session_state) or (not st.session_state.start_app):
        st.session_state['start_app'] = st.button("☘️ Start App ☘️", use_container_width=True, type="primary")

    help_doc()

    if not st.session_state.start_app:
        return

    df = pd.read_json("Valley_Water_Qualified_Plants_sourced.json")
    df = df.reset_index(drop=True)

    st.session_state['orig_df'] = df.copy()

    binary_cols = {
        'Bamboo': False, 
        'Bulb': True, 
        'Grass': True, 
        'Groundcover': True, 
        'Perennial': True, 
        'Palm': True, 
        'Shrub': False, 
        'Succulent': True, 
        'Tree': False, 
        'Vine': False, 
        'Native': True, 
        'Genetic_Concerns': False, 
        'Potentially_Invasive': False,
    }

    user_filter = {}

    st.session_state['openai_api_key'] = st.sidebar.text_input("OPENAI_API_KEY", value=os.environ.get("OPENAI_API_KEY", ""), type="password")
    if len(st.session_state['openai_api_key']) == 0:
        st.sidebar.warning("Add your OPENAI_API_KEY to use chat functionality.")

    # Sidebar Filters
    st.sidebar.header("Filters")
    search_str = st.sidebar.selectbox("Search by Name", options=[""] + df['Scientific_Name'].unique().tolist() + df['Plant_Name'].unique().tolist(), index=0, placeholder="")
    (coverage_min, coverage_max) = st.sidebar.slider(
        "Coverage (sqft)", 
        min_value=0, 
        max_value=int(df.Coverage.max()), 
        value=(0, int(df.Coverage.max()))
    )
    sidebar_cols = st.sidebar.columns(2)
    for ind, (k, default) in enumerate(binary_cols.items()):
        user_filter[k] = sidebar_cols[ind % 2].checkbox(k, value=default)

    if len(search_str) == 0:
        for k, val in user_filter.items():
            if not val:
                df = df[df[k] == 'No']

        df = df[df.Coverage > 0]
        df = df[(df.Coverage >= coverage_min) & (df.Coverage <= coverage_max)]
    else:
        df = df[df['Scientific_Name'].str.contains(search_str) | df['Plant_Name'].str.contains(search_str)]

    df = df.sort_values('Scientific_Name')

    # Main Selection Screen
    tab_available, tab_selected = st.tabs(["🌿 Available Plant", "🛒 Selected Plant"])

    with tab_available:
        st.subheader(f"{len(df)} Total Plants Available. Change the view settings to see them.")
        render_grid(df[~df.index.isin(list(st.session_state["wishlist"].keys()))], key="available")
        
    with tab_selected:
        if len(st.session_state["wishlist"].keys()) == 0:
            st.subheader("Add plants to your wish list from Available Plant 🌿 Tab")
        else:
            render_grid(st.session_state.orig_df.iloc[list(st.session_state["wishlist"].keys())], key="selected")


    # Wishlist Controls
    st.sidebar.header("Your Wish List")

    if len(st.session_state["wishlist"]) > 0:
        selected = st.session_state.orig_df.iloc[list(st.session_state["wishlist"].keys())]
        total_coverage = 0
        for idx, plant in selected.to_dict(orient='index').items():
            total_coverage += st.session_state.wishlist[idx] * int(plant["Coverage"])
    else:
        selected = {}
        total_coverage = 0
    
    st.sidebar.info(f"Selected {len(selected)} Plants | Total Coverage is {total_coverage}")

    st.sidebar.button("🤖 Clear Chat History", use_container_width=True, on_click=clear_history)
    st.sidebar.button("🫙 Clear Wish List", use_container_width=True, on_click=clear_wishlist)
    st.sidebar.download_button(
        "🗳️ Download Wish List",
        convert_df(st.session_state.orig_df.iloc[list(st.session_state["wishlist"].keys())]),
        "wishlist.csv",
        "text/csv",
        key='download-csv',
        use_container_width=True
    )

    if len(st.session_state.openai_api_key) > 0:
        render_chatbot()


render_app()