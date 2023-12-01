import pandas as pd
import numpy as np
import re
import networkx as nx
import plotly.express as px
import itertools
from math import log, exp, sqrt

# LP
from packages.solve_LP import solve_LP, compute_sim, compute_sim_with_t
from packages.graph_utils import *
from packages.data_utils import *
from packages.globals import *
from packages.xai import *

# Dash
import dash
import dash_cytoscape as cyto
import dash_bootstrap_components as dbc
import dash_daq as daq
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import dash_table as dt
from dash_extensions.enrich import DashProxy, Output, Input, State, ServersideOutput, html, dcc, ServersideOutputTransform, MultiplexerTransform
#from dash_extensions import Download
import json
import flask
from networkx.drawing.nx_agraph import write_dot, graphviz_layout

# Explanations
import textwrap
import shap
import io
import base64
import matplotlib.pyplot as plt
import string
from itertools import chain
from collections import Counter
import truecase

# Chat
import os
import openai

# Build App
server = flask.Flask(__name__)

app = DashProxy(#dash.Dash(
    __name__,
    server=server,
    external_stylesheets=[
    'https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css',
    'https://briankeithn.pythonanywhere.com/static/plotly.css',
    '/static/custom.css'
    ],
    prevent_initial_callbacks=True,
    suppress_callback_exceptions=False,
    transforms=[ServersideOutputTransform(), MultiplexerTransform()]
    )
app.title = 'Narrative Maps Visualization Tool'

toggle_row_ent = html.Tr([html.Td("Emphasize common entities in connections."), html.Td(daq.BooleanSwitch(id='use-entities', on=False, color="lightblue"))])
toggle_row_temp = html.Tr([html.Td("Penalize temporal distance in connections."), html.Td(daq.BooleanSwitch(id='use-temporal', on=True, color="lightblue"))])
toggle_row_si = html.Tr([html.Td("Enable semantic interactions."), html.Td(daq.BooleanSwitch(id='use-si', on=True, color="lightblue"))])
toggle_row_xai = html.Tr([html.Td("Enable explainable AI and connection explanation labels."), html.Td(daq.BooleanSwitch(id='use-xai', on=True, color="lightblue"))])
toggle_row_names = html.Tr([html.Td("Enable storyline name extraction."), html.Td(daq.BooleanSwitch(id='use-names', on=True, color="lightblue"))])
toggle_regularization = html.Tr([html.Td("Enable regularization (requires start event)."), html.Td(daq.BooleanSwitch(id='use-regularization', on=True, color="lightblue"))])
toggle_strict_start = html.Tr([html.Td("Enable strict start mode (requires start event)."), html.Td(daq.BooleanSwitch(id='strict-start', on=False, color="lightblue"))])
toggle_table = dbc.Table([html.Tbody([toggle_row_ent, toggle_row_temp, toggle_row_si, toggle_row_xai, toggle_row_names, toggle_regularization, toggle_strict_start])], bordered=False, borderless=True, style={'vertical-align': 'middle'})

# Chatbot setup - Source: https://github.com/plotly/dash-sample-apps/blob/main/apps/dash-gpt3-chatbot/app.py
# Authentication
openai.api_key = os.getenv("OPENAI_API_KEY")
print(openai.api_key)
# Define controls
controls = dbc.InputGroup(
    children=[
        dbc.Input(id="user-input", placeholder="Write to the automatic assistant...", type="text"),
        dbc.Button("Submit", id="submit"),
        dbc.Button("Explain Edge", id="explain-edge-chat"),
        dbc.Button("Explain Importance", id="explain-important-chat"),
        dbc.Button("Explain Story", id="explain-story-chat")
    ]
)
# Description
description = """
You are an AI model designed to assist intelligence analysts, journalists, and fact checkers in narrative sensemaking tasks.
You will be handling a model called narrative maps.
You will be helping users analyze the events and stories contained in the data.
The users will provide you with specific queries about storylines, events, or connections between events.
"""
# Define Layout
conversation = html.Div(
    html.Div(id="display-conversation"),
    style={
        "overflow-y": "auto",
        "display": "flex",
        "height": "calc(80vh - 132px)",
        "flex-direction": "column-reverse",
    },
)
def textbox(text, box="AI", name="User"):
    text = text.replace(f"{name}:", "").replace("You:", "")
    style = {
        "max-width": "60%",
        "width": "max-content",
        "padding": "5px 10px",
        "border-radius": 25,
        "margin-bottom": 20,
    }

    if box == "user":
        style["margin-left"] = "auto"
        style["margin-right"] = 0

        return dbc.Card(text, style=style, body=True, color="primary", inverse=True)

    elif box == "AI":
        style["margin-left"] = 0
        style["margin-right"] = "auto"

        textbox = dbc.Card(text, style=style, body=True, color="light", inverse=False)

        return html.Div([textbox])

    else:
        raise ValueError("Incorrect option for `box`.")

# Files
query = pd.DataFrame(columns=['title', 'url', 'date', 'publication', 'full_text', 'id'])
element_list = generate_elements(graph_df=pd.DataFrame(), antichain=[], hub_nodes=[], sp=[], query=None, low_dim_projection=None)

app.layout = html.Div([
    dbc.Row(
        style={'display': 'flex', 'align-items': 'center', 'vertical-align': 'middle', 'border-bottom': '2px solid black'},
        children=[
        dcc.Dropdown(
            id="dataset-choice",
            options=[{'label':'Coronavirus Data', 'value': 'cv'},
                    {'label':'Coronavirus Data (Small)', 'value': 'cv_smaller'},
                    {'label':'Papers', 'value': 'papers'},
                    {'label':'Cuban Protests (Full)', 'value': 'cuba'},
                    #{'label':'Cuban Protests (Small)', 'value': 'cuba_small'},
                    {'label':'Cuban Protests (160)', 'value': 'cuba_160'},
                    {'label':'Crescent', 'value': 'crescent'},
                    {'label':'Mix Test', 'value': 'mix'},
                    {'label':'Custom', 'value': 'custom'}
                    ],
            optionHeight=50,
            style={'width': '150px', 'margin-right': '5px'},
            value='cv',
            clearable=False),
        html.Button(className="map_btn",
                    style={'background-image' : 'url("/static/load_icon.svg")'},
                    title="Load Data Set", id='load-data-button'),
        html.Span("Find in Map", style={'fontSize': 12, 'fontWeight': 'bold', 'margin-right': '5px'}),
        dcc.Input(id='search-input', inputMode='verbatim', style = {'width': "150px", 'margin-right': '5px'}),
        html.Button(className="map_btn", style={'background-image' : 'url("/static/search_highlight.svg")'},
            title="Search and Highlight", id='search-button'),
        html.Button(className="map_btn", style={'background-image' : 'url("/static/add_node.svg")'},
            title="Add Event to Map", id='add-node-button'),
        html.Button(className="map_btn", style={'background-image' : 'url("/static/remove_node.svg")'},
            title="Remove Event from Map", id='remove-node-button'),
        #html.Button(className="map_btn",  style={'background-image' : 'url("/static/toggle_representative.svg")'},
        #    title="Toggle Important Event", id='antichain-button'),
        html.Button(className="map_btn", style={'background-image' : 'url("/static/add_edge.svg")'},
            title="Add Connection", id='add-edge-button'),
        html.Button(className="map_btn", style={'background-image' : 'url("/static/remove_edge.svg")'},
            title="Remove Connection", id='remove-edge-button'),
        #html.Button(className="map_btn", style={'background-image' : 'url("/static/toggle_main_route.svg")'},
        #    title="Toggle Main Storyline", id='main-route-button'),
        html.Button(className="map_btn", style={'background-image' : 'url("/static/add_storyline.svg")'},
            title="Add Event to Cluster", id='add-cluster-list'),
        dcc.Dropdown(id="cluster-value",
            options=[{'label':str(k) + " (" + color_cluster[k - 1].capitalize() + ")", 'value': k} for k in range(1,len(color_cluster) + 1)],
            value=1, clearable=False, searchable=False, style={'width': '120px', 'margin-right': '5px'}),
        html.Button(className="map_btn", style={'background-image' : 'url("/static/compare_events.svg")'},
            title="Compare Events", id='compare-nodes'),
        html.Button(className="map_btn", style={'background-image' : 'url("/static/explain_edge.svg")'},
            title="Explain Edge", id='explain-edge'),
        #html.Button(className="map_btn", style={'background-image' : 'url("/static/save_svg_icon.svg")'},
        #    title="Download as SVG", id='get-svg-button'),
        html.Button(className="map_btn", style={'background-image' : 'url("/static/save_png_icon.svg")'},
            title="Download as PNG", id='get-png-button'),
        html.Button(className="map_btn", style={'background-image' : 'url("/static/save_json_icon.svg")'},
            title="Download as JSON", id='save-button'),
        html.Button(className="map_btn", style={'background-image' : 'url("/static/zoom_in.svg")'},
            title="Zoom In", id='zoom-in-button'),
        html.Button(className="map_btn", style={'background-image' : 'url("/static/zoom_out.svg")'},
            title="Zoom Out", id='zoom-out-button'),
        html.Button(className="map_btn", style={'background-image' : 'url("/static/zoom_reset.svg")'},
            title="Reset Zoom", id='zoom-reset-button'),
        #html.Button(className="map_btn", style={'background-image' : 'url("/static/pan_reset.svg")'},
        #    title="Reset Panning", id='pan-reset-button'),
        html.Span("Map Size", style={'fontSize': 12, 'fontWeight': 'bold', 'margin-right': '5px'}),
        dcc.Input(
            id="k-input", type="number", value=6,
            min=2, max=25, step=1, style={'width': '70px', 'margin-right': '5px'}
        ),
        html.Span("Coverage %", style={'fontSize': 12, 'fontWeight': 'bold', 'margin-right': '5px'}),
        dcc.Input(
            id="min-cover-input", type="number", value=20,
            min=0, max=100, step=1, style={'width': '60px','margin-right': '5px'}
        ),
        html.Span("Temporal Sensitivity", style={'fontSize': 12, 'fontWeight': 'bold', 'margin-right': '5px'}),
        dcc.Input(
            id="sigma-t-input", type="number", value=30,
            min=0, max=1825, step=1, style={'width': '70px', 'margin-right': '5px'}
        ),
        html.Button(className="map_btn", style={'background-image' : 'url("/static/generate_map.svg")'},
            title="Generate Map", id='recompute-button'),
        dcc.Loading(
            id="loading",
            type="default",
            fullscreen=True,
            style={ 'backgroundColor': '#FFFFFF50'}, children=html.Div('', id="loading-output")
        ),
        dcc.Loading(
            id="loading-chat",
            type="default",
            fullscreen=True,
            style={ 'backgroundColor': '#FFFFFF50'}, children=html.Div('', id="loading-chat-output")
        ),
        dcc.Loading(
            id="loading-report",
            type="default",
            fullscreen=True,
            style={ 'backgroundColor': '#FFFFFF50'}, children=html.Div('', id="loading-report-output")
        ),
        dcc.Loading(
            id="loading-edge",
            type="default",
            fullscreen=True,
            style={ 'backgroundColor': '#FFFFFF50'}, children=html.Div('', id="loading-edge-output")
        ),
        dcc.Loading(
            id="loading-node",
            type="default",
            fullscreen=True,
            style={ 'backgroundColor': '#FFFFFF50'}, children=html.Div('', id="loading-node-output")
        ),
        dcc.Loading(
            id="comparing-node",
            type="default",
            fullscreen=True,
            style={ 'backgroundColor': '#FFFFFF50'}, children=html.Div('', id="comparing-node-output")
        ),
        dcc.Loading(
            id="resetting-table",
            type="default",
            fullscreen=True,
            style={ 'backgroundColor': '#FFFFFF50'}, children=html.Div('', id="resetting-table-output")
        ),
        dcc.Loading(dcc.Store(id="store"), fullscreen=True, type="default"),
        dcc.Loading(dcc.Store(id="previous-actions"), fullscreen=True, type="default"),
        dcc.Input(
            id = 'execution-id',
            value = 0,
            type = 'hidden'
        ),
        dcc.Input(
            id = 'load-data-intermediate',
            value = 0,
            type = 'hidden'
        )
    ]),
    dbc.Row([
    html.Div(className='eight columns', style={"position": "relative"}, children=[
        cyto.Cytoscape(
            id='cytoscape',
            elements=element_list,
            zoom=1,
            layout=base_layout_fit,
            style={ 'height': '95vh', 'width': '68.7vw', 'border-style': 'solid' },
            stylesheet=cyto_stylesheet,
            boxSelectionEnabled=False,
            responsive=True,
            panningEnabled=True
    )
    ]),
    html.Div(className='four columns', children=[
        dcc.Tabs(id='tabs', value="0", style={'font-size': 12, 'height':'5.5vh'}, children=[
            dcc.Tab(label='Overview', value="0", style={'padding': '0', 'font-size': '0.7vw', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'word-spacing': '100vw'}, selected_style={'padding': '0', 'font-size': '0.7vw', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'word-spacing': '100vw'}, children=[
                html.Div(id='overview-tab', style=styles['tab'], children=[
                    html.Div('This tab provides an overview of the data set through its topic and most relevant entities. It also contains a 2D visualization of the document space and the main storyline of the map as it navigates through this space.'),
                    html.Label('Document Space', style={'fontSize': 16, 'font-weight': 'bold', 'text-decoration': 'underline'}),
                    html.Div('Visualize the documents in a 2D projection. Documents are clustered according to their topics/similarity. The main storyline of the narrative map is shown with arrows.'),
                    dcc.Graph(id='scatter-fig'),
                    html.Div(id='overview-tab-internal', children=[
                        html.Div('Topic list', style={'fontSize': 16, 'fontWeight': 'bold', 'text-decoration': 'underline'}),
                        html.Div('Generate the map to extract the topics.'),
                        html.Div('Entity list', style={'fontSize': 16, 'fontWeight': 'bold', 'text-decoration': 'underline'}),
                        html.Div('Generate the map to extract the entities.')
                    ])
                ]),
            ]),
            dcc.Tab(label='Event Details', value="1", style={'padding': '0', 'font-size': '0.7vw', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'word-spacing': '100vw'}, selected_style={'padding': '0', 'font-size': '0.7vw', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'word-spacing': '100vw'}, children=[
                html.Div(style=styles['tab'], children=[
                    html.Div(
                        id='tap-node-id',
                        children='-1',
                        style={'display': 'none'}
                    ),
                    html.Div(
                        id='tap-node-story',
                        style= {'fontSize': 14}
                    ),
                    html.Div(
                        id='tap-node-title',
                        style= {'fontSize': 16}
                    ),
                    html.Div(
                        id='tap-node-text',
                        style= {'fontSize': 14}
                    ),
                ])
            ]),
            dcc.Tab(label='Edge Details', value="2", style={'padding': '0', 'font-size': '0.7vw', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'word-spacing': '100vw'}, selected_style={'padding': '0', 'font-size': '0.7vw', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'word-spacing': '100vw'}, children=[
                html.Div(style=styles['tab'], children=[
                    html.Div(
                        id='tap-edge',
                        style= {'fontSize': 14}
                    ),
                    html.Div(
                        id='explain-edge-text',
                        style= {'fontSize': 14}
                    ),
                ])
            ]),
            dcc.Tab(label='Event Comparison', value="3", style={'padding': '0', 'font-size': '0.7vw', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'word-spacing': '100vw'}, selected_style={'padding': '0', 'font-size': '0.7vw', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'word-spacing': '100vw'}, children=[
                html.Div(style=styles['tab'], children=[
                    html.Div(
                        id='compare-node-text',
                        style= {'fontSize': 14}
                    ),
                ])
            ]),
            dcc.Tab(label='Data Set', value="4", style={'padding': '0', 'font-size': '0.7vw', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'word-spacing': '100vw'}, selected_style={'padding': '0', 'font-size': '0.7vw', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'word-spacing': '100vw'}, children=[
                html.Div(id='data-table-tab', style=styles['tab'], children=[
                    html.Div('Data table with all the events from the current data set. You can search for specific events here to add them to the map (select rows and add them to the map). You may also set a single starting event (use the radio button of the row you want to set as the starting event).'),
                    html.Button("Clear Selection", id="clear-tbl"),
                    dt.DataTable(
                        id='data-tbl', data=query.to_dict('records'), row_selectable='multi',
                        style_data={
                            'whiteSpace': 'normal',
                            'height': 'auto',
                            'textAlign': 'left'
                        },
                        style_header={
                            'backgroundColor': 'rgb(230, 230, 230)',
                            'fontWeight': 'bold',
                            'textAlign': 'center'
                        },
                        filter_action='native',
                        page_current=0,
                        page_size=20,
                        selected_row_ids=[0],
                        page_action='native',
                        columns=[{"name": i, "id": i} for i in query.loc[:,['id', 'date','title']]]
                    )
                ]),
                html.Div(id='data-table-xai-tab', style={'display': 'none'}, children=[dt.DataTable(id='hidden-xai-tbl', data=pd.DataFrame().to_dict('records'))])
            ]),
            dcc.Tab(label='Options', value="5", style={'padding': '0', 'font-size': '0.7vw', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'word-spacing': '100vw'}, selected_style={'padding': '0', 'font-size': '0.7vw', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'word-spacing': '100vw'}, children=[
                html.Div(style=styles['tab'], children=[
                    html.Label('Display Similar Documents', style={'font-weight': 'bold'}),
                    html.Label('For each document in the narrative map, the top N most similar documents will be displayed as gray circles close to it.'),
                    dcc.Input(id='similar-input',  type='number', min=0, max=15, step=1, value=2, style = {'width': "150px", 'margin-right': '5px'}),
                    html.Label('Toggles', style={'font-weight': 'bold'}),
                    toggle_table,
                    html.Label('Interaction Log', style={'font-weight': 'bold'}),
                    html.Label('This part displays the interaction log of the user. This is for debugging purposes only.'),
                    html.Div(
                        id='interact-log-div',
                        style= {'fontSize': 12}
                    )
                ])
            ]),
            dcc.Tab(label='Assistant', value="6", style={'padding': '0', 'font-size': '0.7vw', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'word-spacing': '100vw'}, selected_style={'padding': '0', 'font-size': '0.7vw', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'word-spacing': '100vw'}, children=[
                html.Div(style=styles['tab'], children=[
                            html.Label("GPT-3.5 Explanation Assistant", style={'fontSize': 16, 'font-weight': 'bold', 'text-decoration': 'underline'}),
                            dcc.Store(id="store-conversation", data=""),
                            conversation,
                            controls
                ])
            ]),
            dcc.Tab(label='Reports', value="7", style={'padding': '0', 'font-size': '0.7vw', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'word-spacing': '100vw'}, selected_style={'padding': '0', 'font-size': '0.7vw', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'word-spacing': '100vw'}, children=[
                html.Div(style=styles['tab'], children=[
                            html.Label("GPT-3.5 Report Generation", style={'fontSize': 16, 'font-weight': 'bold', 'text-decoration': 'underline'}),
                            html.Button("Main Story Report", id="msr-btn"),
                            html.Button("Highlights Report", id="hgl-btn"),
                            html.Button("All Storylines Report", id="asl-btn"),
                            html.Div(id='report-output', style={'whiteSpace': 'pre-line'}),
                ])
            ]),
        ]),
    ])
    ]),
    dcc.Download(id="download")
])

@app.callback(Output('load-data-intermediate', 'value'),
              Input('load-data-button', 'n_clicks'),
              State('load-data-intermediate', 'value'))
def intermediate_btn_query(n_clicks, cur_val):
    if n_clicks:
        if cur_val == 0:
            return 1
        if cur_val == 1:
            return 0
    else:
        raise PreventUpdate

@app.callback(ServersideOutput('store', 'data'),
              Input('load-data-intermediate', 'value'),
              [State('dataset-choice', 'value'),
               ], memoize=True)
def query_data(n_clicks, dataset):
    if n_clicks:
        query = read_query(dataset)
        print("Read new data set: " + str(dataset))
        return query
    else:
        raise PreventUpdate

@app.callback(Output('interact-log-div', 'children'),
              Input('previous-actions', 'data'))
def update_interactions(previous_actions):
    if previous_actions:
        return [html.P(str(item)) for item in previous_actions]
    return html.P("No interactions have been logged yet.")

@app.callback([Output('data-table-tab', 'children'),
               Output('previous-actions','data'),
               Output('data-tbl', 'selected_rows')],
               Input('store', 'data'))
def update_data_table(query):
    if query is not None:
        print("Updating data table with new query.")
        new_table = [html.Div('Data table with all the events from the current data set. You can search for specific events here to add them to the map (select rows and add them to the map). You may also set a single starting event (use the radio button of the row you want to set as the starting event).'),
                    html.Button("Clear Selection", id="clear-tbl"),
                    dt.DataTable(
                        id='data-tbl', data=query.to_dict('records'), row_selectable='multi',
                        style_data={
                            'whiteSpace': 'normal',
                            'height': 'auto',
                            'textAlign': 'left'
                        },
                        style_header={
                            'backgroundColor': 'rgb(230, 230, 230)',
                            'fontWeight': 'bold',
                            'textAlign': 'center'
                        },
                        filter_action='native',
                        page_current=0,
                        page_size=20,
                        selected_row_ids=[0],
                        page_action='native',
                        columns=[{"name": i, "id": i} for i in query.loc[:,['id','date','title']]]
                    )]
        return new_table, [], [0]
    else:
        raise PreventUpdate

@app.callback([Output('cytoscape', 'elements'),
               Output('cytoscape', 'layout'),
               Output('loading-output', 'children'),
               Output('previous-actions','data'),
               Output('scatter-fig', 'figure'),
               Output('execution-id', 'value'),
               Output('data-table-xai-tab', 'children'),
               Output('overview-tab-internal', 'children')
              ],
              [Input('remove-node-button', 'n_clicks'),
               Input('remove-edge-button', 'n_clicks'),
               Input('add-edge-button', 'n_clicks'),
               Input('add-node-button', 'n_clicks'),
               #Input('main-route-button', 'n_clicks'),
               #Input('antichain-button', 'n_clicks'),
               Input('recompute-button', 'n_clicks'),
               Input('add-cluster-list', 'n_clicks'),
               Input('search-button', 'n_clicks'),
               Input('search-input', 'n_submit')
              ],
              [State('store', 'data'),
               State('cytoscape', 'elements'),
               State('cytoscape', 'selectedNodeData'),
               State('cytoscape', 'selectedEdgeData'),
               State('k-input', 'value'),
               State('min-cover-input', 'value'),
               State('sigma-t-input', 'value'),
               State('dataset-choice', 'value'),
               State('cluster-value', 'value'),
               State('search-input', 'value'),
               State('similar-input', 'value'),
               State('data-tbl', 'selected_rows'),
               State('data-tbl', 'selected_cells'),
               State('scatter-fig', 'figure'),
               State('execution-id', 'value'),
               State('data-table-xai-tab', 'children'),
               State('overview-tab-internal', 'children'),
               State('previous-actions','data'),
               State('use-entities', 'on'),
               State('use-temporal', 'on'),
               State('use-si', 'on'),
               State('use-xai', 'on'),
               State('use-names', 'on'),
               State('use-regularization', 'on'),
               State('strict-start', 'on')
              ], prevent_initial_call=True)
def interact_with_graph(rmv_node, rmv_edge, add_edge, add_node, 
                        #main_route, anti_chain, 
                        recompute_lp, add_cluster_list, search_btn, search_enter_key, query,
                        elements, nodes, edges, k_input, mincover_input, sigma_t,# trans_reduce,
                        dataset, cluster_value, search_value, similar_input,
                        selected_rows_table, selected_cells, scatter_fig, execution_id, xai_tab, overview_tab, previous_actions,
                        use_entities, use_temporal, use_si, use_xai, use_names, use_regularization, strict_start):#, rep_landmark_mode):
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
        return [elements, base_layout, '', previous_actions, scatter_fig, execution_id, xai_tab, overview_tab]

    # If we are here this means we clicked a "classic" button!
    elements = remove_execution_id(elements, execution_id)
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'remove-node-button':
        if elements and nodes:
            ids_to_remove = [ele_data['id'].replace("_" + str(execution_id) + "I", '') for ele_data in nodes]
            ids_to_remove = [node_id for node_id in ids_to_remove if not node_id.startswith("A")] # Do not count annotations.
            ids_to_remove = [node_id for node_id in ids_to_remove if not node_id.endswith("_out")] # Do not count outside map.
            actual_removed_nodes = []
            for node_id in ids_to_remove:
                for ele in elements:
                    if ele["data"]["id"] == node_id:
                        ele["data"]["id"] = node_id + "_out"
                        ele["data"]["type"] = "out_map"
                        ele["classes"] = "n" # not antichain anymore.
                        actual_removed_nodes.append(ele) # Save current node for prev actions list.
                        break

            # This removes the node and ALL incoming and outgoing edges to the node.
            if len(actual_removed_nodes) > 0:
                new_elements = [ele for ele in elements if ele['data']['id'] not in ids_to_remove and ele['data'].get('target') not in ids_to_remove and ele['data'].get('source') not in ids_to_remove] # Created new list.
                previous_actions.append({"action": "remove-node", "content": ids_to_remove})
                return [add_execution_id(new_elements, execution_id), base_layout, '', previous_actions, scatter_fig, execution_id, xai_tab, overview_tab]
            else:
                return [add_execution_id(elements, execution_id), base_layout, '', previous_actions, scatter_fig, execution_id, xai_tab, overview_tab]
    elif button_id == 'add-node-button':
        if elements and nodes: # If nodes are selected in the map, use those.
            warnings = []
            new_nodes = [node["id"].replace("_" + str(execution_id) + "I", '') for node in nodes] # Remove execution IDs.
            new_nodes = [node.replace("_out", "") for node in new_nodes if node.endswith("_out")] # Check which ones are outside.
            actual_new_nodes = []
            for node_id in new_nodes:
                for ele in elements:
                    if ele["data"]["id"] == node_id + "_out":
                        ele["data"]["id"] = node_id
                        ele["data"]["type"] = "in_map"
                        actual_new_nodes.append(ele) # Save current node for prev actions list.
                        break
            if len(actual_new_nodes) == 0:
                warnings += [html.P('All events were already present. No events were added.')]
            if len(warnings) == 0:
                warnings = ''
            previous_actions.append({"action": "add-node", "content": new_nodes})
            return [add_execution_id(elements, execution_id), base_layout, warnings, previous_actions, scatter_fig, execution_id, xai_tab, overview_tab]

        if elements and selected_cells:# If nodes are selected in the table.
            warnings = []
            new_nodes = []
            for cell in selected_cells:
                new_nodes.append(str(int(cell['row_id'])))
            new_nodes = list(set(new_nodes)) # Remove duplicates.

            actual_new_nodes = []
            for node_id in new_nodes:
                for ele in elements:
                    if ele["data"]["id"] == str(node_id) + "_out":
                        ele["data"]["id"] = str(node_id)
                        ele["data"]["type"] = "in_map"
                        actual_new_nodes.append(ele) # Save current node for prev actions list.
                        break
            if len(actual_new_nodes) == 0:
                warnings += [html.P('All events were already present. No events were added.')]
            if len(warnings) == 0:
                warnings = ''
            previous_actions.append({"action": "add-node", "content": new_nodes})
            return [add_execution_id(elements, execution_id), base_layout, warnings, previous_actions, scatter_fig, execution_id, xai_tab, overview_tab]
    elif button_id == 'search-button' or button_id == 'search-input':
        if elements:
            if len(search_value.strip()) == 0:
                for ele in elements:
                    ele["data"]["search"] = "F"
                return [add_execution_id(elements, execution_id), base_layout, '', previous_actions, scatter_fig, execution_id, xai_tab, overview_tab] # Nothing happens.
            for ele in elements:
                if "story" in ele['data']['id'] or "-" in ele['data']['id']:
                    continue
                if search_value[0] == "*" and search_value[-1] == "*": # Both asterisks.
                    if search_value[1:-1].strip().lower() in ele["data"]["full_text"].lower(): # Does not care about positioning or whole words.
                        ele["data"]["search"] = "T"
                    else:
                        ele["data"]["search"] = "F"
                elif search_value[0] == "*": # Only cares about the end of the words.
                    if re.search(r"(\b|\s)(\w*?){}(\b|\s)".format(re.escape(search_value[1:].strip())), ele["data"]["full_text"], flags=re.IGNORECASE):
                        ele["data"]["search"] = "T"
                    else:
                        ele["data"]["search"] = "F"
                elif search_value[-1] == "*": # Only cares about the start of the word.
                    if re.search(r"(\b|\s){}(\w*?)(\b|\s)".format(re.escape(search_value[:-1].strip())), ele["data"]["full_text"], flags=re.IGNORECASE):
                        ele["data"]["search"] = "T"
                    else:
                        ele["data"]["search"] = "F"
                else: # Exact match.
                    if re.search(r"(\b|\s){}(\b|\s)".format(re.escape(search_value.strip())), ele["data"]["full_text"], flags=re.IGNORECASE):
                        ele["data"]["search"] = "T"
                    else:
                        ele["data"]["search"] = "F"
            return [add_execution_id(elements, execution_id), base_layout, '', previous_actions, scatter_fig, execution_id, xai_tab, overview_tab]
    elif button_id == 'remove-edge-button':
        if elements and edges:
            # Full edge for logs.
            removed_edges = [ele_data for ele_data in edges]
            for ele_data in removed_edges:
                ele_data['id'] = ele_data['id'].replace("_" + str(execution_id) + "I", '') # Remove execution ID.
                ele_data['source'] = ele_data['source'].replace("_" + str(execution_id) + "I", '')
                ele_data['target'] = ele_data['target'].replace("_" + str(execution_id) + "I", '')
            # Now only get ids.
            ids_to_remove = [ele_data['id'].replace("_" + str(execution_id) + "I", '') for ele_data in removed_edges]
            new_elements = [ele for ele in elements if ele['data']['id'] not in ids_to_remove] # Created new list.
            previous_actions.append({"action": "remove-edge", "content": ids_to_remove})
            return [add_execution_id(new_elements, execution_id), base_layout, '', previous_actions, scatter_fig, execution_id, xai_tab, overview_tab]
    elif button_id == 'main-route-button':
        if elements and edges:
            new_elements = elements[:]
            edge_ids = [edge['id'].replace("_" + str(execution_id) + "I", '') for edge in edges]
            for edge in new_elements:
                if edge['data']['id'] in edge_ids:
                    if 'sp' in edge['classes']:
                        edge['classes'] = edge['classes'].replace('sp','').strip()
                        edge['data']['width'] -= 1
                    else:
                        edge['classes'] = (edge['classes'] + ' sp').strip()
                        edge['data']['width'] += 1
            previous_actions.append({"action": "main-route", "content": edge_ids})
            return [add_execution_id(new_elements, execution_id), base_layout, '', previous_actions, scatter_fig, execution_id, xai_tab, overview_tab]
    elif button_id == 'add-edge-button':
        if elements and nodes:
            if len(nodes) == 2: # Can't connect more than two nodes for now.
                # Temporal restriction.
                start_node = nodes[0]['id'].replace("_out", "").replace("_" + str(execution_id) + "I", '')
                end_node = nodes[1]['id'].replace("_out", "").replace("_" + str(execution_id) + "I", '')
                if nodes[0]['date'] > nodes[1]['date'] or int(start_node) > int(end_node): # Can't be equal.
                    start_node, end_node = end_node, start_node # Swap!
                new_nodes = [start_node, end_node]
                for node_id in new_nodes:
                    for ele in elements:
                        if ele["data"]["id"] == node_id + "_out":
                            ele["data"]["id"] = node_id
                            ele["data"]["type"] = "in_map"
                            break
                edge_id = start_node + '-' + end_node
                if any(edge['data']['id'] == edge_id for edge in elements):
                    return [add_execution_id(elements, execution_id), base_layout, '', previous_actions, scatter_fig, execution_id, xai_tab, overview_tab]
                weight = 0.5
                edge = {"data": { "id": edge_id,
                                 "source": start_node,
                                 "target": end_node,
                                 "label": "Manual",
                                 "weight": 0.5,
                                 "width": 0.2 + 1.8 * float(weight)},
                        "classes": ""
                       }
                new_elements = elements + [edge] # Created new list
                previous_actions.append({"action": "add-edge", "content": edge_id.replace("_" + str(execution_id) + "I", '')})
                return [add_execution_id(new_elements, execution_id), base_layout, '', previous_actions, scatter_fig, execution_id, xai_tab, overview_tab]
    elif button_id == 'antichain-button':
        if elements and nodes:
            new_elements = elements[:]
            node_ids = [node['id'].replace('_out', '').replace("_" + str(execution_id) + "I", '') for node in nodes]
            for node in new_elements:
                if node['data']['id'] in node_ids:
                    if 'ac' in node['classes']:
                        node['classes'] = node['classes'].replace('ac', '').strip()
                    else:
                        node['classes'] += ' ac'
            previous_actions.append({"action": "antichain", "content": node_ids})
            return [add_execution_id(new_elements, execution_id), base_layout, '', previous_actions, scatter_fig, execution_id, xai_tab, overview_tab]
    elif button_id == 'recompute-button':
        # Nothing to do but save the whole graph here.
        if query is None:
            return [elements, base_layout, html.P("Error: Please load the data set first."), previous_actions, scatter_fig, execution_id, xai_tab, overview_tab]

        operation_list = []
        if use_si: # If SI is disabled the list remains empty.
            for item in previous_actions:
                if item.get("action") == "add-node":
                    for node in item['content']:
                        operation_list.append("AN:" + str(node))
                if item.get("action") == "remove-node":
                    for node in item['content']:
                        operation_list.append("RN:" + str(node))
                elif item.get("action") == "add-edge":
                     operation_list.append("AE:" + str(item["content"]))
                elif item.get("action") == "remove-edge":
                    for edge in item['content']:
                        operation_list.append("RE:" + str(edge))
                elif item.get("action") == "add-cluster-list":
                     operation_list.append("ACL:" + str(item['content'][0]) + "-" + str(item['content'][1]))
        # Solve LP and create new graph_df.
        cluster_size_est = np.sqrt(len(query.index))/2
        cluster_size_est = max(5 * round(cluster_size_est / 5), 2) # Round to nearest multiple of 5, cannot be below 2.

        n_neighbors = 2
        init = 'random'
        print(init)
        if len(query.index) > 40:
            n_neighbors = 10
            init = 'spectral'
        elif len(query.index) > 120:
            n_neighbors = cluster_size_est
            init = 'spectral'
        start_nodes=[]
        end_nodes=[]
        if len(selected_rows_table) > 2:
            return [elements, base_layout, html.P("Error: Select at most 2 events in the table."), previous_actions, scatter_fig, execution_id, xai_tab, overview_tab]            
        elif len(selected_rows_table) == 2:
            start_nodes = [selected_rows_table[0]]
            end_nodes = [selected_rows_table[1]]
            if start_nodes[0] > end_nodes[0]: # If not in chronological order
                swap = start_nodes[0]
                start_nodes[0] = end_nodes[0]
                end_nodes[0] = swap
        elif len(selected_rows_table) == 1:
            start_nodes = [selected_rows_table[0]]
            # If this is not the case then there are zero rows selected.
        min_dist = 0.0
        graph_df_new, status, scatter_df, sim_table, clust_sim_table, ent_table, ent_doc_list, cluster_assignment = solve_LP(query,
                dataset=str(dataset), operations=operation_list,
                K=k_input, mincover=mincover_input/100, sigma_t=sigma_t,
                min_samples=1, min_cluster_size=cluster_size_est, n_neighbors=n_neighbors, min_dist=min_dist,
                start_nodes=start_nodes, end_nodes=end_nodes, umap_init=init,
                use_entities=use_entities, use_temporal=use_temporal, strict_start=strict_start, use_regularization=use_regularization)
        status_msg = "LP Status: " + status[1] + ", Clusters: " + str(status[0])# + ", Storylines: " + str(numstories)

        if 'Optimal' in status[1]:
            # Generate graph from LP solution.
            G = build_graph(graph_df_new)
            storylines = graph_stories(G, start_nodes=start_nodes, end_nodes=end_nodes)
            status_msg += ", Storylines: " + str(len(storylines))

            # We now apply transitive reduction.
            remove_list_transitive = []
            for story in storylines:
                max_idx = len(story) - 1
                for idx, node in enumerate(story):
                    out_edges = G.out_edges(node)
                    # If it's not the last element in the story (no need to clean the last one, because there are no in-story outgoing edges)
                    if idx != max_idx:
                        for edge in out_edges:
                            if edge[1] in story and edge[1] != story[idx + 1]: # If it's an in-story edge and not the immediately adjacent one.
                                remove_list_transitive.append(edge)
            G.remove_edges_from(remove_list_transitive)
            # We now filter redundant interstory connections.
            remove_list_interstory = []
            for (story_i, story_j) in list(itertools.combinations(storylines, 2)):
                keep_set = set()
                edge_boundary_list_i = list(nx.edge_boundary(G, story_i, story_j))
                edge_boundary_list_j = list(nx.edge_boundary(G, story_j, story_i))
                if len(edge_boundary_list_i) > 0:
                    earliest_i = min(edge_boundary_list_i, key = lambda t: (t[0], t[1]))
                    latest_i = max(edge_boundary_list_i, key = lambda t: (t[0], t[1]))
                else:
                    earliest_i = None
                if len(edge_boundary_list_j) > 0:
                    earliest_j = min(edge_boundary_list_j, key = lambda t: (t[1], t[0]))
                    latest_j = max(edge_boundary_list_j, key = lambda t: (t[1], t[0]))
                else:
                    earliest_j = None
                if not earliest_i and not earliest_j:
                    # No connections
                    continue
                elif earliest_i and not earliest_j:
                    # One-sided.
                    if earliest_i == latest_i: # There's only one connection, keep it.
                        continue
                    elif earliest_i[0] == latest_i[0]:
                        # This means that the second values are different, but the first one is the same.
                        # For this special case we keep the earliest one.
                        keep_set.add(earliest_i)
                    else:
                        # Keep both
                        keep_set.add(earliest_i)
                        keep_set.add(latest_i)
                elif not earliest_i and earliest_j:
                    # One-sided and reversed.
                    if earliest_j == latest_j: # There's only one connection, keep it.
                        continue
                    elif earliest_j[1] == latest_j[1]:
                        # This means that the first values are different, but the second one is the same.
                        # For this special case we keep the earliest one.
                        keep_set.add(earliest_j)
                    else:
                        # Keep both
                        keep_set.add(earliest_j)
                        keep_set.add(latest_j)
                else:
                    # Two-sided.
                    if earliest_i[0] < earliest_j[1]:
                        # E_i happened first. Keep it
                        keep_set.add(earliest_i)
                        if latest_i[0] > latest_j[1]:
                            # L_i happened last. Keep it.
                            keep_set.add(latest_i)
                        else:
                            keep_set.add(latest_j)
                    else:
                        # E_j happened first.
                        keep_set.add(earliest_j)
                        if latest_i[0] > latest_j[1]:
                            # E_i happened last. Keep it.
                            keep_set.add(latest_i)
                        else:
                            keep_set.add(latest_j)
                redundant_edge_set = set(edge_boundary_list_i) | set(edge_boundary_list_j)
                remove_list_interstory += list(redundant_edge_set - keep_set)

            G.remove_edges_from(remove_list_interstory)

            for node in G.nodes:
                sum_node = sum([exp(-data['weight']) for u, v, data in G.out_edges(node, data=True)])
                if sum_node == 0:
                    continue
                for u, v, data in G.out_edges(node, data=True):
                    G[u][v]['weight'] = -log(exp(-data['weight']) / sum_node)
                sum_node = sum([exp(-G[u][v]['weight']) for u, v, data in G.out_edges(node, data=True)])
            # Update graph_df
            for i, row in graph_df_new.iterrows():
                new_adj_list = [v for u, v in G.out_edges(str(row['id']))]
                new_adj_weights = [exp(-G[u][v]['weight']) for u, v in G.out_edges(str(row['id']))]
                graph_df_new.at[i,'adj_list'] = new_adj_list
                graph_df_new.at[i,'adj_weights'] = new_adj_weights

            A = nx.nx_agraph.to_agraph(G)
            A.graph_attr.update(newrank="true", splines="false", rankdir="TB", ranksep="0.5", nodesep="1.0")
            A.node_attr.update(width='1.2')
            # Column alignment
            for idx, story in enumerate(storylines):
                B = A.add_subgraph(story, name="cluster_" + str(idx))
            # Row alignment
            A.layout(prog="dot", args="-y")
            A.write("test.dot")
            layout = [(n, A.get_node(n).attr['pos']) for n in A.nodes()]
            positions_dot= {n: tuple(float(ss) for ss in s.split(",")) for (n,s) in layout}
            if use_temporal:
                temporal_sim_table = compute_sim_with_t(query, str(dataset), sigma_t, min_dist=min_dist)
            else:
                temporal_sim_table = compute_sim(query, min_dist=min_dist)
            low_dim_projection, unraveled_set = force_layout(query, positions_dot, temporal_sim_table, top_k=similar_input)
            hub_nodes = get_representative_landmarks(G, storylines, query, mode="centrality")
            antichain = get_representative_landmarks(G, storylines, query, mode="centroid")

            if len(storylines) > 0:
                sp = storylines[0]#get_shortest_path(G) # CHECK DIFFERENCE BETWEEN SHORTEST PATH AND STORYLINES!
            else:
                sp = [] # There is no shortest path?
            if use_names:
                story_names = generate_names(graph_df_new, storylines, topn=1)
            else:
                story_names = [[""]] * len(storylines)

            if use_xai:
                graph_df_new, cluster_description = add_connection_types(query, graph_df_new, sim_table, clust_sim_table, ent_table, ent_doc_list, cluster_assignment)

            label_length = 450
            if dataset == 'crescent':
                label_length = 150
            new_elements = generate_elements(graph_df_new, antichain, hub_nodes, sp,
            low_dim_projection, query, positions_dot,
            storylines, story_names, unraveled_set, elements, label_length)
            # Additional plot.
            scatter_df.title = scatter_df.title.apply(lambda txt: '<br>'.join(textwrap.wrap(txt, width=40)))
            scatter_fig = px.scatter(scatter_df, x="X", y="Y", color="cluster_id", custom_data=['cluster_id','title'])
            scatter_fig.update_traces(
                hovertemplate="<br>".join([
                    "Cluster: %{customdata[0]}",
                    "Title: %{customdata[1]}"
                ])
            )
            scatter_fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
            scatter_fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
            scatter_fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
            #for edge in G.edges:
            #offset = min(query.id.astype('int'))
            for edge in zip(sp, sp[1:]):
                x_1 = scatter_df.loc[scatter_df['id'] == edge[0]]['X'].values[0]
                x_2 = scatter_df.loc[scatter_df['id'] == edge[1]]['X'].values[0]
                y_1 = scatter_df.loc[scatter_df['id'] == edge[0]]['Y'].values[0]
                y_2 = scatter_df.loc[scatter_df['id'] == edge[1]]['Y'].values[0]
                annotation = {
                  'x':x_2,  # arrows' head
                  'y':y_2,  # arrows' head
                  'ax':x_1,  # arrows' tail
                  'ay':y_1,  # arrows' tail
                  'xref':'x',
                  'yref':'y',
                  'axref':'x',
                  'ayref':'y',
                  'text':'',  # if you want only the arrow
                  'showarrow':True,
                  'arrowhead':3,
                  'arrowsize':1,
                  'arrowwidth':1,
                  'arrowcolor':'black'
                }
                scatter_fig.add_annotation(annotation)
            execution_id += 1
            if use_xai: # No topic or entity details if XAI is disabled.
                topic_count = len(cluster_description)
                topic_count_desc = "There are " + str(topic_count) + " topic clusters in the data set."
                if topic_count == 1:
                    topic_count_desc = "There is only a single topic cluster in the data set."
                # Process topic lists
                topic_list = [html.P([html.B("Topic " + str(idx_topic) + ": "), html.Span(desc)], style={'fontSize': 14}) for idx_topic, desc in enumerate(cluster_description)]
                # Process entities
                extended_ent_list = chain.from_iterable(ent_doc_list)
                entity_list_count = Counter([truecase.get_true_case(ent.text.lower().strip()) for ent in extended_ent_list])
                entity_list_string = []
                for ent, count in entity_list_count.most_common():
                    entity_list_string.append(html.Div(ent + ": " + str(count), style={'fontSize': 14}))
                new_xai_table = [dt.DataTable(id='hidden-xai-tbl', data=graph_df_new.to_dict('records'))]
                new_overview_tab = [html.Div('Topic list', style={'fontSize': 16, 'fontWeight': 'bold', 'text-decoration': 'underline'})]
                new_overview_tab += topic_list
                new_overview_tab.append(html.Div('Entity list', style={'fontSize': 16, 'fontWeight': 'bold', 'text-decoration': 'underline'}))
                new_overview_tab += entity_list_string
                return [add_execution_id(new_elements, execution_id), base_layout_fit, html.P([status_msg]), previous_actions, scatter_fig, execution_id, new_xai_table, new_overview_tab]
            else:
                new_overview_tab = [html.Div('Topic list', style={'fontSize': 16, 'fontWeight': 'bold', 'text-decoration': 'underline'}),
                                    html.Div('Explainable AI component is disabled. Enable it to generate the topic list.'),
                                    html.Div('Entity list', style={'fontSize': 16, 'fontWeight': 'bold', 'text-decoration': 'underline'}),
                                    html.Div('Explainable AI component is disabled. Enable it to generate the entity list.')]
                return [add_execution_id(new_elements, execution_id), base_layout_fit, html.P([status_msg]), previous_actions, scatter_fig, execution_id, xai_tab, new_overview_tab]
        return [add_execution_id(elements, execution_id), base_layout, html.P([status_msg]), previous_actions, scatter_fig, execution_id, xai_tab, overview_tab]
    elif button_id == 'add-cluster-list':
        if elements and nodes:
            new_elements = elements[:]
            node_ids = [node['id'].replace("_" + str(execution_id) + "I", '') for node in nodes]
            for node in new_elements:
                if node['data']['id'] in node_ids:
                    node['data']['cluster'] = int(cluster_value)
                    node['classes'] += (' ' + color_cluster[int(cluster_value) - 1])
            previous_actions.append({"action": "add-cluster-list", "content": (node_ids, int(cluster_value))})
            return [add_execution_id(new_elements, execution_id), base_layout, '',  previous_actions, scatter_fig, execution_id, xai_tab, overview_tab]
    return [add_execution_id(elements, execution_id), base_layout, '', previous_actions, scatter_fig, execution_id, xai_tab, overview_tab]

@app.callback([Output('compare-node-text', 'children'),
               Output('comparing-node-output', 'children')],
              [Input('compare-nodes', 'n_clicks')],
              [State('execution-id', 'value'),
              State('cytoscape', 'selectedNodeData'),
              State('store', 'data')])
def explainTwoEvents(cmp_btn, execution_id, node_data, query):
    if node_data is not None:
        if len(node_data) == 2: # Only two nodes.
            if "story" in node_data[0]['id'] or "story" in node_data[0]['id']:
                return ["", ""]
            output_list = []
            #query = read_query(dataset, start_date, end_date)
            #offset = min(query.id.astype('int'))
            print(execution_id)
            print(node_data[0]['id'])
            print(node_data[1]['id'])
            id_start = int(node_data[0]['id'].replace("_" + str(execution_id) + "I", '').replace('_out', ''))# - offset)
            id_end = int(node_data[1]['id'].replace("_" + str(execution_id) + "I", '').replace('_out', ''))# - offset)

            s1_row = query.loc[query['id'] == str(id_start)].iloc[0] # Access single row.
            s2_row = query.loc[query['id'] == str(id_end)].iloc[0]
            s1 = s1_row['title'] + "\n" + s1_row['full_text'] 
            s2 = s2_row['title'] + "\n" + s2_row['full_text']

            shap_values, features = sim_explanation(s1, s2)
            # Generate plot here.
            shap_values.base_values[0] = 0.5
            shap_plot = similarity_plot(shap_values, features, topn=10)#shap.plots.waterfall(shap_values[0], max_display=max_display, show=False)            
            shap_plot = plt.gcf()
            # Output plot as image.
            s = io.BytesIO()
            shap_plot.savefig(s, format='png', bbox_inches="tight")
            plt.close(shap_plot)
            s = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
            img = 'data:image/png;base64,%s' % s
            output_list += [html.Strong("Influential keywords in event comparison", style={'fontSize': 16, 'text-decoration': 'underline'}),
                            html.Div("Event 1: " + s1_row['title']),
                            html.Div("Event 2: " + s2_row['title']),
                            html.Img(src=img, style={"max-width": "100%"})] # Final element is shap plot.
            return [output_list, ""]
    return ["", ""]


@app.callback([Output('tap-edge', 'children'),
               Output('loading-edge-output', 'children')],
              [Input('explain-edge', 'n_clicks'),
               Input('cytoscape', 'tapEdgeData')],
              [State('execution-id', 'value'),
              State('hidden-xai-tbl', 'data'),
              State('cytoscape', 'selectedEdgeData'),
              State('store', 'data'),
              State('use-xai', 'on')])
def explainEdge(cmp_btn, data, execution_id, xai_tab, edge_data, query, use_xai):
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    node_ids = data['id'].replace("_" + str(execution_id) + "I", '').split(sep='-')
    id_start = node_ids[0]
    id_end = node_ids[1]
    if id_start.isdigit() and id_end.isdigit():
        start_row = query.iloc[int(id_start)]# - offset]
        end_row = query.iloc[int(id_end)]# - offset]
        output_list = [html.P('Edge Data', style={'fontSize': 16, 'fontWeight': 'bold', 'text-decoration': 'underline'}),
                       html.P(children=[html.Strong('Source: '), html.Span(start_row['title'])], style={'fontSize': 14}),
                       html.P(children=[html.Strong('Target: '), html.Span(end_row['title'])], style={'fontSize': 14}),
                       html.P(children=[html.Strong('Weight: '), html.Span(str(round(data['weight'], 2)))], style={'fontSize': 14})]

    if button_id == 'explain-edge':
        if not use_xai:
            output_list += [html.P(children=[html.Strong('The XAI component is disabled, please enable and re-generate the map to get connection explanations.')], style={'fontSize': 16})]
            return [output_list, ""]
        if len(edge_data) == 1: # Only one edge must be selected.
            data = edge_data[0]
            #offset = min(query.id.astype('int'))
            node_ids = data['id'].replace("_" + str(execution_id) + "I", '').split(sep='-')
            id_start = node_ids[0]# - offset
            id_end = node_ids[1]# - offset

            s1_row = query.loc[query['id'] == str(id_start)].iloc[0] # Access single row.
            s2_row = query.loc[query['id'] == str(id_end)].iloc[0]
            s1 = s1_row['title'] + "\n" + s1_row['full_text']
            s2 = s2_row['title'] + "\n" + s2_row['full_text']
            #print(s1)
            #print(s2)

            shap_values, features = sim_explanation(s1, s2)
            # Generate plot here.
            shap_values.base_values[0] = 0.5
            shap_plot = similarity_plot(shap_values, features, topn=10)#shap.plots.waterfall(shap_values[0], max_display=max_display, show=False)
            shap_plot = plt.gcf()


            # Output plot as image.
            s = io.BytesIO()
            shap_plot.savefig(s, format='png', bbox_inches="tight")
            plt.close(shap_plot)
            s = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
            img = 'data:image/png;base64,%s' % s

            xai_df = pd.DataFrame.from_dict(xai_tab)
            start_row = xai_df.loc[xai_df['id'] == int(id_start)]
            adjacency_list_start = start_row['adj_list'].values[0]
            if str(id_end) not in adjacency_list_start:
                # This must be a manual connection.
                explanation = [html.P("This connection was manually added by the user.")]
                explanation += [html.Strong("Keyword contributions to the connection", style={'fontSize': 16, 'text-decoration': 'underline'}), html.Img(src=img, style={"max-width": "100%"})]
                output_list = [html.P(children=[html.Strong('Connection Explanation')], style={'fontSize': 16, 'text-decoration': 'underline'})] + explanation
                return [output_list, ""]
            index_end = adjacency_list_start.index(str(id_end))

            explanation = []
            explanation_details = []
            top_xai = [html.P(children=[html.Strong("Event Topics: ", style={'fontSize': 16}),
                html.Span(str(start_row['topical_description'].values[0][index_end]), style={'fontSize': 14})])]
            #sim_xai = [html.P(children=[html.Strong("Common Keywords: "), html.Span(str(start_row['similarity_description'].values[0][index_end]))])]
            ent_xai = [html.P(children=[html.Strong("Common Entities: ", style={'fontSize': 16}),
                html.Span(str(start_row['entity_description'].values[0][index_end]), style={'fontSize': 14})])]


            if "Manual" in data['label']: # No more temporal connections.
                explanation = [html.P("This connection was manually added by the user.")]
            else:
                if "Topical" in data['label']:
                    explanation += [html.P("This connection is based on common topical information. See extracted topics of each event below.", style={'fontSize': 14})]
                    explanation_details += top_xai
                if "Similarity" in data['label']:
                    explanation += [html.P("This connection is based on similar content and keywords. See extracted keywords and their contributions to similarity below.", style={'fontSize': 14})]
                    #explanation_details += sim_xai
                if "Entity" in data['label']:
                    explanation += [html.P("This connection is based on common entities. See extracted entities of each event below.", style={'fontSize': 14})]
                    explanation_details += ent_xai
            explanation += explanation_details
            explanation += [html.Strong("Keyword contributions to the connection", style={'fontSize': 16, 'text-decoration': 'underline'}), html.Img(src=img, style={"max-width": "100%"})] # Final element is shap plot.

            if id_start.isdigit() and id_end.isdigit():
                output_list = [html.P(children=[html.Strong('Connection Explanation')], style={'fontSize': 16, 'text-decoration': 'underline'})] + explanation
                return [output_list, ""]
    return [output_list, ""]


@app.callback([Output('tap-node-id', 'children'),
               Output('tap-node-story', 'children'),
               Output('tap-node-title', 'children'),
               Output('tap-node-text', 'children'),
               Output('loading-node-output', 'children')],
              [Input('cytoscape', 'tapNodeData')],
              [State('execution-id', 'value'),
              State('hidden-xai-tbl', 'data'),
              State('cytoscape', 'selectedNodeData'),
              State('store', 'data')])
def displayTapNodeData(data, execution_id, xai_tab, node_data, query):
    if data:
        if "story" in data['id']:
            return ["", "", "", "", ""]
        #query = read_query(dataset, start_date, end_date)
        node_id = data['id'].replace('_out', '').replace("_" + str(execution_id) + "I", '')
        #offset = min(query.id.astype('int'))
        node_row = query.iloc[int(node_id)]# - offset]
        id_current = int(node_id)# - offset

        xai_df = pd.DataFrame.from_dict(xai_tab)
        start_row = xai_df.loc[xai_df['id'] == int(id_current)]

        string_list = node_row['full_text'].split(sep='\n')
        output_list = [html.Div(node_row['date'], style={'fontSize': 10, 'font-style': 'italic'}),
                      html.A(node_row['url'], href=node_row['url'], target='_blank', style={'fontSize': 12, 'font-style': 'italic'})]
        title = html.Div(node_row['title'], style={'fontSize': 14, 'fontWeight': 'bold'}, contentEditable=False)
        if data["storyname"]:
            node_story = [html.Div("Part of the Storyline: " + data["storyname"], style={'fontSize': 14, 'fontWeight': 'bold'})]
        else:
            # Do not show storyline assignment if there is no info about it.
            node_story = [] #[html.Div("Part of the Storyline: " + "Unassigned", style={'fontSize': 14, 'fontWeight': 'bold'})]
        explanation = data.get('explanation', None)
        if explanation:
            node_story.append(html.Div(explanation, style={'fontSize': 12}))
        node_story.append(html.P(start_row['event_topic'], style={'fontSize': 12, 'fontWeight': 'bold'}))
        node_story.append(html.Hr(style={'height': 1, 'margin-top': 0, 'margin-bottom': 0}))


        if len(string_list) == 1: # No natural split.
            splitter = re.compile(r'''((?:[^\."']|"[^"]*"|'[^']*'|\([^\)]*\))+)''')
            p_n = 4 # Split into paragraphs of 4 if necessary. Lots of bug fixes to handle U.S. case.
            split_list = splitter.split(string_list[0].strip())
            split_list_copy = []
            previous_us = ""
            for s in split_list:
                if s == "S":
                    split_list_copy[-1] += "." + s # Special U.S. case
                    previous_us = "S"
                elif s == ".":
                    if previous_us == "S":
                        split_list_copy[-1] += s
                        previous_us = "S."
                    continue # Do nothing
                else:
                    if previous_us == "S.":
                        split_list_copy[-1] += " " + s
                        previous_us = ""
                    else:
                        split_list_copy.append(s)
            split_list_p_n = [split_list_copy[i * p_n:(i + 1) * p_n] for i in range((len(split_list_copy) + p_n - 1) // p_n )]
            for s in split_list_p_n: # List of list of strings.
                # Concatenate and output.
                output_list += [html.P(". ".join(s) + ".", contentEditable=False)]    
        else:
            for s in string_list:
                if len(s.strip()) > 0:
                    output_list += [html.P(s.strip(), contentEditable=False)]

        return [node_id, node_story, title, output_list, ""]
    return ["", "", "", "", ""]


@app.callback(Output("cytoscape", "generateImage"),
              [Input("get-png-button", "n_clicks")])
def get_image(get_png_clicks):
    ctx = dash.callback_context
    if ctx.triggered:
        input_id = ctx.triggered[0]["prop_id"].split(".")[0]
        ftype = input_id.split("-")[1]# File type to ouput of 'svg, 'png', 'jpg', or 'jpeg' (alias of 'jpg')
        # 'store': Stores the image data in 'imageData' !only jpg/png are supported
        # 'download'`: Downloads the image as a file with all data handling
        # 'both'`: Stores image data and downloads image as file.
        action = 'download'
        # ftype = input_id.split("-")[-1]
        return { 'type': ftype, 'action': action }
    return {}




@app.callback([Output("cytoscape", "zoom")],
              [Input("zoom-in-button", "n_clicks"),
               Input("zoom-out-button", "n_clicks"),
               Input("zoom-reset-button", "n_clicks")],
               [State("cytoscape", "zoom")])
def modify_zoom(zi, zo, zr, current_zoom):
    ctx = dash.callback_context
    input_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if input_id == 'zoom-in-button':
        return min(current_zoom + 0.1, 10)
    elif input_id == 'zoom-out-button':
        return max(current_zoom - 0.1, 0.1)
    elif input_id == 'zoom-reset-button':
        return 1.0


#@app.callback(Output("cytoscape", "pan"), Input("pan-reset-button", "n_clicks"))
#def modify_pan(pr):
#    return {'x': 0, 'y': 0}

@app.callback(
     Output('download', 'data'),
     [Input('save-button', 'n_clicks')],
     [State('cytoscape', 'elements')])
def update_download_link(sb, elements):
    ctx = dash.callback_context
    if ctx.triggered:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'save-button':
            json_str = json.dumps(elements)
            return dict(content=json_str, filename="graph.json")

# @app.callback(Output('cytoscape', 'stylesheet'),
#               [Input('cytoscape', 'tapNode')],
#               [State('cytoscape', 'selectedNodeData')])
# def generate_stylesheet(node, node_data):
#     if not node or not node_data:
#         return cyto_stylesheet
#     follower_color="#00D900"
#     following_color="#FF4136"
#     # Copying the stylesheet is necessary for proper updates, otherwise the changes stack
#     # (particularly when there is lag, but I can't find a pattern, probably depends on some internal priorities of cytoscape).
#     # It is not necessary to do a deep copy, just a new list.
#     stylesheet = list(cyto_stylesheet)
#     if "story" not in node['data']['id']:
#         stylesheet.append({
#                 "selector": 'node[id = "{}"]'.format(node['data']['id']),
#                 "style": {
#                     'label': 'data(label)',
#                     'font-size': '8px',
#                     'background-color': '#969696',
#                     'line-color': '#969696',
#                     'background-opacity': 0.7,
#                     'text-wrap': 'wrap',
#                     'text-max-width': 150,
#                     'text-valign': 'center',
#                     'text-halign': 'right',
#                     "text-opacity": 1,
#                     'z-index': 9999
#                 }
#             })
#         for node_story in node['data']['story']:
#             if node_story != node['data']['id']:
#                 stylesheet.append({
#                         "selector": 'node[id= "{}"]'.format(node_story),
#                         "style": {
#                             'background-color': '#454545',
#                             'background-opacity': 0.7
#                         }
#                     })
#         for edge in node['edgesData']:
#             if edge['source'] == node['data']['id']:
#                 stylesheet.append({
#                     "selector": 'edge[id= "{}"]'.format(edge['id']),
#                     "style": {
#                         "mid-target-arrow-color": following_color,
#                         "mid-target-arrow-shape": "vee",
#                         "line-color": following_color,
#                         'opacity': 0.9,
#                         'z-index': 5000
#                     }
#                 })
#             if edge['target'] == node['data']['id']:
#                 stylesheet.append({
#                     "selector": 'edge[id= "{}"]'.format(edge['id']),
#                     "style": {
#                         "mid-target-arrow-color": follower_color,
#                         "mid-target-arrow-shape": "vee",
#                         "line-color": follower_color,
#                         'opacity': 1,
#                         'z-index': 5000
#                     }
#                 })
#     return stylesheet


@app.callback(
    [Output("data-tbl", "selected_rows"),
     Output("resetting-table-output", "children")],
    Input("clear-tbl", "n_clicks"),
)
def clear(n_clicks):
    return [], ""

# Chatbot code source: https://github.com/plotly/dash-sample-apps/blob/main/apps/dash-gpt3-chatbot/app.py
@app.callback(
    Output("display-conversation", "children"), [Input("store-conversation", "data")]
)
def update_display(chat_history):
    return [
        textbox(x, box="user") if i % 2 == 0 else textbox(x, box="AI")
        for i, x in enumerate(chat_history.split("<split>")[:-1])
    ]


@app.callback(
    [Output("user-input", "value")],
    [Input("submit", "n_clicks"), Input("user-input", "n_submit")],
)
def clear_input_chat(n_clicks, n_submit):
    return ""



@app.callback(
    [Output("store-conversation", "data"), Output("loading-chat", "children")],
    [Input("submit", "n_clicks"),
    Input("user-input", "n_submit"),
    Input("explain-edge-chat", "n_clicks"),
    Input("explain-important-chat", "n_clicks"),
    Input("explain-story-chat", "n_clicks")],
    [State('execution-id', 'value'), State('store', 'data'),
    State("user-input", "value"), State("store-conversation", "data"), 
    State("cytoscape", "elements"), State('cytoscape', 'selectedEdgeData'),
    State('cytoscape', 'selectedNodeData')],
)
def run_chatbot(n_clicks, n_submit, explain_edge_chat, explain_important_chat, explain_story_chat, execution_id, query, user_input, chat_history, elements, edge_data, node_data):
    print(n_clicks, n_submit, explain_edge_chat)
    if n_clicks == 0 and n_submit is None and explain_edge_chat == 0 and explain_important_event == 0:
        return "", None

    if (user_input is None or user_input == "") and explain_edge_chat == 0 and explain_important_event == 0:
            return chat_history, None

    name = "Assistant"

    ctx = dash.callback_context
    changed_id = ctx.triggered[0]['prop_id'].split('.')[0]
    # First add the user input to the chat history
    if changed_id == "submit" or changed_id == "user-input":
        print("Free query selected.")
        chat_history += f"You: {user_input}<split>{name}: "
    elif changed_id == "explain-edge-chat":
        print("Edge explanation selected.")
        if len(edge_data) == 1: # Only one edge must be selected.
            data = edge_data[0]
            #offset = min(query.id.astype('int'))
            node_ids = data['id'].replace("_" + str(execution_id) + "I", '').split(sep='-')
            id_start = node_ids[0]# - offset
            id_end = node_ids[1]# - offset

            s1_row = query.loc[query['id'] == str(id_start)].iloc[0] # Access single row.
            s2_row = query.loc[query['id'] == str(id_end)].iloc[0]
            s1 = s1_row['title']# + "\n" + s1_row['full_text']
            s2 = s2_row['title']# + "\n" + s2_row['full_text']
            user_input = "Explain the possible connections between the following two events: (" + s1 + ") and (" + s2 + ")."
            chat_history += f"You: {user_input}<split>{name}: "
        else:
            print("Edge explanation error: Multiple edges selected.")
            return chat_history, None
    elif changed_id == "explain-important-chat":
        print("Node explanation selected.")
        if len(node_data) == 1: # Only one node must be selected.
            node_id = node_data[0]['id']
            story_id = node_data[0].get('parent', "")
            story_name = [ele_data['data']['label'] for ele_data in elements if ele_data['data']['id'] == story_id][0] # Get story name.

            edge_i_list = [ele_data['data']['source'] for ele_data in elements if ele_data['data'].get('target') == node_id]
            edge_j_list = [ele_data['data']['target'] for ele_data in elements if ele_data['data'].get('source') == node_id]

            edge_i_event_labels = [query.loc[query['id'] == str(event_id).replace("_" + str(execution_id) + "I", '')].iloc[0]['title'] for event_id in edge_i_list]
            edge_j_event_labels = [query.loc[query['id'] == str(event_id).replace("_" + str(execution_id) + "I", '')].iloc[0]['title'] for event_id in edge_j_list]

            user_input = "Explain why the event (" + node_data[0]['label'] + ") is important to the storyline. Consider its preceding events: (" + ", ".join(edge_i_event_labels) + ") and its subsequent events (" + ", ".join(edge_j_event_labels) + ")."
            chat_history += f"You: {user_input}<split>{name}: "
        else:
            print("Node explanation error: Multiple nodes selected.")
            return chat_history, None
    elif changed_id == "explain-story-chat":
        print("Story explanation selected.")
        if len(node_data) == 1: # Only one node must be selected.
            story_id = node_data[0].get('parent', "")
            story_name = [ele_data['data']['label'] for ele_data in elements if ele_data['data']['id'] == story_id][0] # Get story name.
            events = [ele_data['data']['label'] for ele_data in elements if ele_data['data'].get('parent', "") == story_id]
            user_input = "Explain the storyline (" + story_name + ") formed by the following events : (" + ", ".join(events) + ")."
            chat_history += f"You: {user_input}<split>{name}: "
        else:
            print("Story explanation error: Multiple nodes selected.")
            return chat_history, None
    else:
        print("ERROR!")
        return chat_history, None

    model_input = chat_history.replace("<split>", "\n")

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = [{"role": "system", "content": description},
                    {'role': 'user', 'content': model_input}],
        max_tokens=500,
        stop=["You:"],
        temperature=0.9,
    )
    model_output = response['choices'][0]['message']['content']

    chat_history += f"{model_output}<split>"

    return chat_history, None


@app.callback(
    [Output('report-output', 'children'), Output('loading-report', 'children')],
    [Input('msr-btn', 'n_clicks'), Input('hgl-btn', 'n_clicks'), Input('asl-btn', 'n_clicks')],
    [State('execution-id', 'value'), State('store', 'data'), State("cytoscape", "elements")]
)
def report_generation(msr, hgl, asl, execution_id, query, elements):
    ctx = dash.callback_context
    changed_id = ctx.triggered[0]['prop_id'].split('.')[0]
    print(changed_id)

    model_input = ""
    if changed_id == 'msr-btn':
        # Only get main story.
        events = [ele_data['data']['label'] for ele_data in elements if "story_0" in ele_data['data'].get('parent', "")]
        events_str = "[" + ", ".join(events) + "]"
        model_input = "Please generate a report that explains the main storyline of this narrative map. Make sure to draw conclusions, inferences, and speculate on causes and effects. Do not re-list the events mentioned here. Here is the list of events: " + events_str
    elif changed_id == 'hgl-btn':
        events = [ele_data['data']['label'] for ele_data in elements if ele_data['classes'] in ["hub", "ac"]]
        events_str = "[" + ", ".join(events) + "]"
        model_input = "Please generate a report that explains the key highlights of this narrative map. Make sure to draw conclusions, inferences, and speculate on causes and effects. Do not re-list the events mentioned here. Here is the list of events: " + events_str
    elif changed_id == 'asl-btn':
        stories = [(ele_data['data']['id'], ele_data['data']['label']) for ele_data in elements if ele_data['classes'] == "story"]
        events_str = ""
        for idx, story in enumerate(stories):
            events = [ele_data['data']['label'] for ele_data in elements if ele_data['data'].get('parent', "") == story[0]]
            if idx == 0:
                events_str += "Main Story (" + story[1] + "): [" + ", ".join(events) + "] \n"
            else:
                events_str += "Side Story " + str(idx) + " (" + story[1] + "): [" + ", ".join(events) + "] \n"
        model_input = "Please generate a report that explains the storylines of this narrative map. Make sure to draw conclusions, inferences, and speculate on causes and effects. Do not re-list the events mentioned here. Here is the list of events: " + events_str
    print(model_input)
    if model_input != "":
        response = openai.ChatCompletion.create(   
            model="gpt-3.5-turbo",
            messages = [{"role": "system", "content": description},
                        {'role': 'user', 'content': model_input}],
            max_tokens=1000,
            temperature=0.9,
        )
        report = response['choices'][0]['message']['content']
        return report, None
    else: # Empty report
        return "", None


if __name__ == "__main__":
    app.run()