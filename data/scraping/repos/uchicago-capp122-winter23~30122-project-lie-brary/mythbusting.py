'''
Interactive mythbusting page powered by GPT-3.
# Prompt and initial code from: Shradha G
# Completed and put in dash by: Reza R Pratama
'''
from dash import html, register_page, callback, Input, Output
import lie_brary.scripts.scrap.key as key
from dash.exceptions import PreventUpdate
import openai
import dash_bootstrap_components as dbc

register_page(__name__)

layout = html.Div([
    html.H1('Mythbusting the topics with GPT-3'),
    html.P('Please enter the topic you want to mythbust.'
           ' it may take a while to get the result after the Submit button is clicked.'),
    html.Br(),
    # input box
    dbc.Textarea(id = 'topic',
                 className="mb-3",
                 placeholder="Write the topic here...",
                 value=''),
    dbc.Button("Submit",
                id="submit-button1",
                color="primary",
                className="me-1"),
    html.Br(),
    html.Br(),
    # output box
    html.H4('Mythbusting result:'),
    html.Br(),
    dbc.Alert(id='output1', color="primary"),
])

@callback(
    Output('output1', 'children'),
    Output('submit-button1', 'n_clicks'),
    Input('submit-button1', 'n_clicks'),
    Input('topic', 'value')
)
def update_output(n_clicks, topic):
    if n_clicks is None:
        raise PreventUpdate
    else:
        openai.api_key = key.OPENAI_KEY
        text = topic
        prompts = "Write text to counteract misinformation on Illinois' Pre-Trial Fairness Act in this topics: "
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                  messages=[{"role": "user", "content":prompts+text}])
        response = completion.choices[0].message.content
        return response, None