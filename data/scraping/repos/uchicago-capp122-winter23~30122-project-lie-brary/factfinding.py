'''
Interactive fact-finding page powered by GPT-3.
Prompt and Code by: Reza R Pratama
'''
from dash import html, register_page, callback, Input, Output
from dash.exceptions import PreventUpdate
import lie_brary.scripts.scrap.key as key
import openai
import dash_bootstrap_components as dbc

register_page(__name__)

layout = html.Div([
    html.H1('Fact-finding about the statements with GPT-3'),
    html.P('Please enter any information you want to fact-find.'
           ' it may take a while to get the result after the Submit button is clicked.'),
    html.Br(),
    # input box
    dbc.Textarea(id = 'info',
                 className="mb-3",
                 placeholder="Write the text here...",
                 value=''),
    dbc.Button("Submit",
                id="submit-button2",
                color="primary",
                className="me-1"),
    html.Br(),
    html.Br(),
    # output box
    html.H4('Fact-finding result:'),
    html.Br(),
    dbc.Alert(id='output2', color="primary"),
])

@callback(
    Output('output2', 'children'),
    Output('submit-button2', 'n_clicks'),
    Input('submit-button2', 'n_clicks'),
    Input('info', 'value')
)
def update_output(n_clicks, info):
    '''Update the output box with the fact-finding result.
    Input:
        n_clicks: number of clicks on the submit button
        info: text inputted by the user
    Output:
        response: fact-finding result'''

    if n_clicks is None:
        raise PreventUpdate
    else:
        openai.api_key = key.OPENAI_KEY
        text = info
        prompts = ("in one word and give reason, this is misinformation or"
                "not regarding the safe-t act or  Illinois Pre-Trial Fairness Act: ")
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                  messages=[{"role": "user", "content":prompts+text}])
        response = completion.choices[0].message.content
        return response, None