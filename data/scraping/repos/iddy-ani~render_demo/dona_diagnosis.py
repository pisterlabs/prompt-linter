from app import app
from dash import dcc
from dash import html
import openai
from dash.dependencies import Input, Output, State
import os

api_key = os.environ.get(r'OPENAI_API_KEY')
# Set up OpenAI API key
openai.api_key = api_key

# Define the layout for the Dona's Diagnosis page
dona_layout = html.Div([
    html.H1("Dona's Diagnosis"),
    dcc.Input(id='question-input', type='text',
              placeholder='Ask Dona a question...'),
    html.Button('Submit', id='submit-question'),
    html.Div(id='answer-output')
])


@app.callback(Output('answer-output', 'children'),
              [Input('submit-question', 'n_clicks')],
              [State('question-input', 'value')])
def generate_answer(n_clicks, question):
    if not question:
        return html.P("What has you here today?")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": question}]
    )
    answer = response['choices'][0]['message']['content']
    return html.P(answer)
