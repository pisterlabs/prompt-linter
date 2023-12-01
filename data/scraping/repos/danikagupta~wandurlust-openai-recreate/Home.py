# Based on original from https://github.com/andfanilo/social-media-tutorials/blob/master/20231116-st_assistants/streamlit_app.py
import streamlit as st
import plotly.graph_objects as go
from openai import OpenAI
import random
import json
import time

st.set_page_config(page_title="Wanderlust Recreate",page_icon=" 🗺️",layout="wide")

st.title("Wanderlust Demo (Streamlit's version)")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

assistant_id = st.secrets["OPENAI_ASSISTANT_ID"]

assistant_state = "assistant"
thread_state = "thread"
conversation_state = "conversation"
last_openai_run_state = "last_openai_run"
map_state = "map"
markers_state = "markers"

user_msg_input_key = "input_user_msg"


if "map" not in st.session_state:
    st.session_state["map"] = {
        "latitude": 37.38875850950195,
        "longitude": -121.97601068975905,
        "zoom": 12,
    }

if markers_state not in st.session_state:
    st.session_state["markers"] = {
        "lat": [37.38875850950195],
        "lon": [-121.97601068975905],
        "text": ["WYKYK"],
    }

if "conversation" not in st.session_state:
    st.session_state["conversation"] = []

if (assistant_state not in st.session_state) or (thread_state not in st.session_state):
    st.session_state["assistant"] = client.beta.assistants.retrieve(assistant_id)
    st.session_state["thread"] = client.beta.threads.create()
    st.session_state["run"] = None

#with st.sidebar.expander("Session_state"):
#    st.write(st.session_state.to_dict())



#left_col,right_col=st.columns(2)

right_col=st.sidebar
left_col,blank_col=st.columns([10,1])

with left_col:
    for role,message in reversed(st.session_state["conversation"]):
        with st.chat_message(role):
            st.markdown(message)


with right_col:
    fig=go.Figure(go.Scattermapbox(mode="markers"))
    if st.session_state[markers_state] is not None:
        fig.add_trace(
            go.Scattermapbox(
                mode="markers",
                lat=st.session_state[markers_state]["lat"],
                lon=st.session_state[markers_state]["lon"],
                marker=go.scattermapbox.Marker(
                    size=18,
                    color="rgb(0, 128, 255)",
                    opacity=0.7
                ),
                text=st.session_state[markers_state]["text"],
                hoverinfo="text",
            )
        )
    fig.update_layout(
        mapbox=dict(
            accesstoken=st.secrets["MAPBOX_TOKEN"],
            center=go.layout.mapbox.Center(
            lat=st.session_state["map"]["latitude"],
            lon=st.session_state["map"]["longitude"],
            ),
            pitch=0,
            zoom=st.session_state["map"]["zoom"],
        ),
        margin=dict(l=0,r=0,t=0,b=0)
    )
    st.plotly_chart(fig,config={"displayModeBar":False},use_container_width=True,key="plotly")

def random_coordinates():
    random_lat = random.uniform(-90,90)
    random_lon = random.uniform(-180,180)
    return random_lat,random_lon

def update_map_state(latitude,longitude,zoom):
    #st.session_state["map"]["latitude"] = latitude
    #st.session_state["map"]["longitude"] = longitude
    #st.session_state["map"]["zoom"] = zoom
    st.session_state["map"] = {
        "latitude": latitude,
        "longitude": longitude,
        "zoom": zoom,
    }
    return "Map updated"

def add_markers_state(latitudes, longitudes, labels):
    """OpenAI tool to update markers in-app
    """
    st.session_state["markers"] = {
        "lat": latitudes,
        "lon": longitudes,
        "text": labels,
    }
    return "Markers added"

tool_to_function = {
    "update_map": update_map_state,
    "add_markers": add_markers_state,
}

def on_text_input():
    # Send message to OpenAI
    client.beta.threads.messages.create(
        thread_id=st.session_state["thread"].id,
        role="user",
        content=st.session_state["input_user_msg"],
    )
    st.session_state["run"] = client.beta.threads.runs.create(
        thread_id=st.session_state["thread"].id,
        assistant_id=st.session_state["assistant"].id,
    )
    completed=False
    while not completed:
        run=client.beta.threads.runs.retrieve(
            thread_id=st.session_state["thread"].id,
            run_id=st.session_state["run"].id,
        )
        if(run.status=="completed"):
            completed=True
        elif run.status=="requires_action":
            tools_output=[]
            for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                f=tool_call.function
                f_name=f.name
                f_args=json.loads(f.arguments)
                tool_result=tool_to_function[f_name](**f_args)
                print(f"FUNCTION CALL: Got {tool_result} for function {f_name} with args {f_args}")
                tools_output.append(
                    {
                        "tool_call_id": tool_call.id,
                        "output": tool_result,
                    }
                )
            #with st.sidebar.expander("Tools Output"):
            #    st.write(tools_output)
            print(f"Submitting tools output = {tools_output} for input={st.session_state['input_user_msg']}")
            client.beta.threads.runs.submit_tool_outputs(
                thread_id=st.session_state["thread"].id,
                run_id=st.session_state["run"].id,
                tool_outputs=tools_output,
            )
        else:
            time.sleep(0.3)
    
    st.session_state["conversation"] = [
        (m.role,m.content[0].text.value) for m in client.beta.threads.messages.list(st.session_state["thread"].id).data
    ]

    return


st.chat_input(placeholder="Where do you want to go today?",key="input_user_msg", on_submit=on_text_input)
