# Import necessary libraries 
from dash import html
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash import callback, clientside_callback
from dash.dependencies import Input, Output, State, ClientsideFunction
import os 
import pandas as pd
import datetime
from dash import get_asset_url
from datetime import date
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import openai
from io import StringIO
# from boto.s3.connection import S3Connection
import os

# openai.api_key = os.getenv("KEY")



# MAX_HEIGHT = 20 
# MAX_LEN = 480 


# def get_height (time):
#     out = int(MAX_HEIGHT * (time / MAX_LEN)) 
#     return out if out > 5 else 5

dir_path = os.path.dirname(os.path.realpath(__file__))
data = pd.read_csv(dir_path + r'/data/' + r'processed_data_v2.csv')
activities = ["Other"] + list(set(data["name"]))
openai.api_key = os.getenv("KEY")
gpt_model = "gpt-3.5-turbo"
sample_answer1 = """00:00,06:00,Sleep
                    06:00,07:00,Morning Routine
                    07:00,08:00,Exercise
                    08:00,08:30,Breakfast
                    08:30,09:30,Work
                    09:30,10:00,Break
                    10:00,12:00,Work
                    12:00,13:00,Lunch
                    13:00,15:00,Work
                    15:00,15:30,Break
                    15:30,17:30,Work
                    17:30,18:00,Break
                    18:00,19:00,Personal Development
                    19:00,20:00,Dinner
                    20:00,21:00,Relaxation
                    21:00,22:00,Hobby/Leisure Activity
                    22:00,23:00,Preparation for Bedtime
                    23:00,23:59,Relaxation/Self-Care"""

master_messages = [{"role": "system", "content": "Your only job is to generate schedule suggestions for someone who wishes to stay productive, as well as to improve his well-being by having a well-organized schedule. All of your answers must be in a format of a csv string with three columns, corresponding to start and end times of activities as well as activity names. Your answer can contain nothing but this csv string with 3 columns. You are not allowed to give comments or anything that is different from a csv string. If there's not enough information, then provide a generic schedule that you think works best. Activities must begin at 00:00 and end at 23:59."}]
assistant_messages = [{"role": "assistant", "content": sample_answer1}]
# user_messages = [{"role": "user", "content": "My current schedule is [00:00 - 09:00 : Sleep; 09:00 - 11:00 - Breakfast; 11:00 - 12:30 - Lunch; 12:30 - 19:30 - Studies; 9:30 - 24:00 - Partying with friends]. Provide a csv string with a new schedule which would help improve my productivity and overall well-being. Only this csv string should be the output. While you should make some improvements, you are not allowed to completely erase this schedule."}]
init_prompt = master_messages + assistant_messages
temp_param = 0.75 #governs how non-determenistic chatGPT is.

schedule = pd.DataFrame(columns = ["start_time", "end_time", "activity_name"])

if "" in activities:
    activities.remove("")
# print(activities)


layout = dbc.Container([html.Div(id = "planning", 
                  children = [
                      html.H4("Tomorrow schedule creation:"),
                      dbc.Row(
                        [
                            dbc.Col(
                                html.Div(
                                    [
                                        dbc.Label("Activity", style={'font-weight': 'bold'}),
                                        html.Br(),
                                        dbc.Select(
                                            options=[{'label': i, 'value': i} for i in activities],
                                            id="activity_sel",
                                            placeholder=activities[0],
                                            size='sm',
                                            value=activities[0],
                                        ),
                                    ]
                                ),
                                width=2,
                            ),
                            dbc.Col(
                                [
                                    dbc.Stack(
                                        [
                                            html.Div(
                                                [
                                                    dbc.Label("Start time: ", style={'font-weight': 'bold'}, id = "start_label"),
                                                    html.Br(),
                                                    dcc.Input(type="time", step="600", id = "start_time")
                                                ],
                                            ),
                                            html.Div(
                                                [
                                                    dbc.Label("End time: ", style={'font-weight': 'bold'}, id = "end_label"),
                                                    html.Br(),
                                                    dcc.Input(type="time", step="600", id = "end_time"),
                                                ],
                                            ),
                                            html.Div(
                                                [
                                                    dbc.Label("   ", style={'font-weight': 'bold'}, id = "empty", hidden = False),
                                                    html.Br(),
                                                    html.Button('Add', id='add-but', n_clicks=0, 
                                                                style={'font-size': '14px', 
                                                                    'width': '80px', 
                                                                    'font-weight': 'bold',
                                                                    'display': 'inline-block', 
                                                                    'margin-bottom': '10px', 
                                                                    'margin-top': '19px', 
                                                                    'height':'28px', 
                                                                    'verticalAlign': 'bottom'}),
                                                ],
                                            ),
                                        ],
                                        direction="horizontal",
                                        gap=3,
                                    ),
                                ],
                                width=6,
                            ),
                            # dbc.Col(
                            #     html.Div(
                            #         [
                            #             dbc.Label("   ", style={'font-weight': 'bold'}, id = "empty", hidden = False),
                            #             html.Br(),
                            #             html.Button('Add', id='add-but', n_clicks=0, 
                            #                         style={'font-size': '14px', 
                            #                                'width': '140px', 
                            #                                'font-weight': 'bold',
                            #                                'display': 'inline-block', 
                            #                                'margin-bottom': '10px', 
                            #                                'margin-right': '5px', 
                            #                                'height':'32px', 
                            #                                'verticalAlign': 'bottom'}),
                            #         ]
                            #     ),
                            #     width=3,
                            # ),


                        ],
                        style={"height": "10vh", "background-color": "white"},
                        align='center',
                    ),

                    dbc.Row(
                        [
                            dbc.Col(
                                html.Div(
                                    [
                                        dbc.Label("Add new activity", style={'font-weight': 'cursive'}, id = "label11"),
                                        dbc.Input(id="manual-activ", placeholder="Other activity...", type="text")
                                    ]
                                ),
                                width=2,
                            ),

                        ],
                        style={"height": "10vh", "background-color": "white"},
                        align='center',
                    ),
                    dbc.Row(
                        dbc.Col(
                            dbc.Card([
                            html.Div(
                                [
                                    dcc.Graph(id='cur_sched'),
                                    # html.P(id='routine_err', className='m-3 text-danger'),
                                ],
                            )]),
                            width=12,
                        ),
                    ),
                    html.Br(),
                    html.H4("Schedule suggested by ChatGPT:"),
                    dbc.Label("Important: it is a limitation of OpenAI API that each request takes 15-20 seconds. Please wait after clicking button Add above."),
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Div(
                                    [
                                        # dbc.Label("Preferences", style={'font-weight': 'cursive'}, id = "desires_label"),
                                        dbc.Input(id="desires", placeholder="Preferences to consider", type="text")
                                    ]
                                ),
                                width=12,
                            ),

                        ],
                        style={"height": "10vh", "background-color": "white"},
                        align='center',
                    ),
                    dbc.Row(
                        dbc.Col(
                            dbc.Card([
                            html.Div(
                                [
                                    dcc.Graph(id='suggested'),
                                    # html.P(id='routine_err', className='m-3 text-danger'),
                                ],
                            )]),
                            width=12,
                        ),
                    ),
                  ]
                )
            ]
        )


def df2txt(df):
    if len(df) == 0:
        return "[No activities scheduled yet]"
    else:
        return "".join([f"{df.iloc[i, 0]} - {df.iloc[i, 1]}: {df.iloc[i, 2]};" for i in range(len(df))])


@callback(
   Output(component_id='manual-activ', component_property='style'),
   [Input(component_id='activity_sel', component_property='value')])

def show_hide_element(value):
    if value == 'Other':
        return {'display': 'block'}
    if value != 'Other':
        return {'display': 'none'}
    

@callback(
   Output(component_id='label11', component_property='style'),
   [Input(component_id='activity_sel', component_property='value')])

def show_hide_element(value):
    if value == 'Other':
        return {'display': 'block'}
    if value != 'Other':
        return {'display': 'none'}
    

@callback(
    Output('cur_sched', 'figure'),
    [Input('add-but', 'n_clicks')],
    [State('activity_sel', 'value'),
    State('start_time', 'value'),
    State('end_time', 'value'),
    State('manual-activ', 'value')]
)
def update_output(n_clicks, activ1, start_time, end_time, other_activ):
    global schedule
    # print(activ1, start_time, end_time, other_activ)
    # if n_clicks == 0:
        # return go.Figure(go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers"))
    if n_clicks != 0:
        start_time = pd.to_datetime(start_time) + datetime.timedelta(days=1)
        end_time = pd.to_datetime(end_time) + datetime.timedelta(days=1)
        activity_name = activ1 if activ1 != "Other" else other_activ

        if end_time > start_time:
            new_row = {"start_time": start_time, "end_time": end_time, "activity_name": activity_name}
            schedule = schedule.append(new_row, ignore_index = True)

    # print(schedule)


    fig = px.timeline(schedule, 
                      x_start="start_time", 
                      x_end="end_time", 
                      y=np.ones(len(schedule)),
                      hover_name = "activity_name",
                      color = "activity_name")
    fig.update_yaxes(autorange="reversed") # otherwise tasks are listed from the bottom up

    fig.update_yaxes(visible=False)
    fig.update_layout(
        plot_bgcolor = "white",
        showlegend=True,
        height=200,
        legend=dict(
            orientation="h",
            # entrywidth=70,
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title='Activity name',
            font=dict(
                family="Arial",
                size=12,
                color="black"
            ),
        )
    )
    fig.update_xaxes(range = [pd.to_datetime("00:00") + datetime.timedelta(days=1), 
                              pd.to_datetime("23:59") + datetime.timedelta(days=1)])


    return fig

@callback(
    Output('suggested', 'figure'),
    [Input('add-but', 'n_clicks')],
    [State('desires', 'value')]
)
def update_chatgpt(n_clicks, desires):
    global schedule
    global init_prompt
    global temp_param
    # print(desires)
    if n_clicks == 0:
        fig =  go.Figure(go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers"))
        fig.update_yaxes(autorange="reversed") # otherwise tasks are listed from the bottom up

        fig.update_yaxes(visible=False)
        fig.update_layout(
            plot_bgcolor = "white",
            showlegend=True,
            height=200,
            legend=dict(
                orientation="h",
                # entrywidth=70,
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                title='Activity name',
                font=dict(
                    family="Arial",
                    size=12,
                    color="black"
                ),
            )
        )

        fig.update_xaxes(range = [pd.to_datetime("00:00") + datetime.timedelta(days=1), 
                                  pd.to_datetime("23:59") + datetime.timedelta(days=1)])

        return fig
    
    optional = f"Also, please consider that {desires}" if desires != None else ""
    prompt = [{"role": "user", "content": f"My current schedule is [{df2txt(schedule)}]. Please provide a csv string with a new schedule consisting of three columns: start time, end time and name of activity, which would help improve my productivity and overall well-being. However, all activities from my schedule must remain in the new schedule, possible with changed start or end times. Only this csv string should be the output. IF you think there is not enough information provided, then give a generic schedule that you think works best. Activities must begin at 00:00 and end at 23:59." + optional}]

    user_prompt = init_prompt + prompt
    # gpt_schedule = pd.DataFrame(columns = ["start_time", "end_time", "activity_name"])

    completion = openai.ChatCompletion.create(model = gpt_model, 
                                            temperature = temp_param,
                                            messages = user_prompt)

    csv_str = completion.choices[0].message.content
    print(csv_str)
    csvStringIO = StringIO(csv_str)
    gpt_schedule = pd.read_csv(csvStringIO, 
                            sep=",",
                            header = None,
                            names = ["start_time", "end_time", "activity_name"])
    # gpt_schedule.to_csv(r"C:\Users\Ashnv\OneDrive\Documents\dataviz\pages\data\gpt_sched.csv")

    gpt_schedule["start_time"] = gpt_schedule["start_time"].str.replace(" ", "")
    gpt_schedule["end_time"] = gpt_schedule["end_time"].str.replace(" ", "")

    gpt_schedule["start_time"] = pd.to_datetime(list(map(str, gpt_schedule["start_time"])))
    gpt_schedule["end_time"] = pd.to_datetime(list(map(str, gpt_schedule["end_time"])))
    gpt_schedule["start_time"] = gpt_schedule["start_time"] + datetime.timedelta(days=1)
    gpt_schedule["end_time"] = gpt_schedule["end_time"] + datetime.timedelta(days=1)
    print(gpt_schedule)

    fig = px.timeline(gpt_schedule, 
                      x_start="start_time", 
                      x_end="end_time", 
                      y=np.ones(len(gpt_schedule)),
                      hover_name = "activity_name",
                      color = "activity_name")
    fig.update_yaxes(autorange="reversed") # otherwise tasks are listed from the bottom up

    fig.update_yaxes(visible=False)
    fig.update_layout(
        plot_bgcolor = "white",
        showlegend=True,
        height=200,
        legend=dict(
            orientation="h",
            # entrywidth=70,
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title='Activity name',
            font=dict(
                family="Arial",
                size=12,
                color="black"
            ),
        )
    )
    fig.update_xaxes(range = [pd.to_datetime("00:00") + datetime.timedelta(days=1), 
                              pd.to_datetime("23:59")+ datetime.timedelta(days=1)])

    return fig

# Define the page layout
# layout = html.Div(id="main", children=[
#     html.Div(id="drag_container0", className="container", children=[
    
#     dbc.Row([
#         dbc.Col(
#             html.Div(id = "left_container", className="lcontainer", children = [
#     html.H1("Possible Activities"),
#     html.Div(id="drag_container", className="container", children=[
#         dbc.Card([
#             dbc.CardHeader("Sleep",
#                 style = {"color": "white"}),
#             dbc.CardBody(
#                 f"Estimated time: {estim_time['Sleep']} min",
#                 style = {"color": "white"}
#             ),
#         ], color = "success",
#         style = {"height": f"{get_height(estim_time['Sleep'])}rem",
#                  "width": "90%"}),
#         dbc.Card([
#             dbc.CardHeader("Read News",
#                 style = {"color": "white"}),
#             dbc.CardBody(
#                 "Estimated time: 20 min",
#                 style = {"color": "white"}
#             ),
#         ], color = "success",
#         style = {"height": f"{get_height(estim_time['Read News'])}rem",
#                  "width": "90%"}),
#         dbc.Card([
#             dbc.CardHeader("Nap",
#                 style = {"color": "white"}),
#             dbc.CardBody(
#                 "Estimated time: 30 min",
#                 style = {"color": "white"}
#             ),
#         ], color = "warning",
#         style = {"height": f"{get_height(estim_time['Nap'])}rem",
#                  "width": "90%"}),
#         dbc.Card([
#             dbc.CardHeader("Homework",
#                 style = {"color": "white"}),
#             dbc.CardBody(
#                 "Estimated Time: 1.5 hr",
#                 style = {"color": "white"}
#             ),
#         ], color = "danger",
#         style = {"height": f"{get_height(estim_time['Homework'])}rem",
#                  "width": "90%"}),
#         dbc.Card([
#             dbc.CardHeader("Play LoL",
#                 style = {"color": "white"}),
#             dbc.CardBody(
#                 "Estimated Time: 1 hr",
#                 style = {"color": "white"}
#             ),
#         ], color = "danger",
#         style = {"height": f"{get_height(estim_time['Play LoL'])}rem",
#                  "width": "90%"}),
#     ], style={'padding': 20}) ])) , 
#     dbc.Col(
#     html.Div(id = "right_container", className = "rcontainer", children = [
#         html.H1("Your current plan"),
#         html.Div(id="drag_container2", className="container", children=[
#         # dbc.Card([
#         #     dbc.CardHeader("Sleep"),
#         #     dbc.CardBody(
#         #         "Estimated Time: 8 hr"
#         #     ),
#         # ],
#         # style = {"height": "15rem",
#         #          "width": "90%"}),
#         # dbc.Card([
#         #     dbc.CardHeader("Play LoL"),
#         #     dbc.CardBody(
#         #         "Estimated Time: 8 hr"
#         #     ),
#         # ],
#         # style = {"height": "15rem",
#         #          "width": "90%"}),
#         # dbc.Card([
#         #     dbc.CardHeader("Jogging"),
#         #     dbc.CardBody(
#         #         "Estimated Time: 8 hr"
#         #     ),
#         # ],
#         # style = {"height": "15rem",
#         #          "width": "90%"}),
#     ], style={'padding': 20} )])
    
#     ) ])
#  ] )
# ])

# clientside_callback(
#     ClientsideFunction(namespace="clientside", function_name="make_draggable"),
#     Output("drag_container0", "data-drag"),
#     [Input("drag_container2", "id"), Input("drag_container", "id")]
# )