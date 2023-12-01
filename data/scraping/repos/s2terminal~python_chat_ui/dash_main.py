from dash import Dash, html, dcc, callback, Output, Input, State
import openai

app = Dash(__name__)

app.layout = html.Div([
    html.Div(dcc.Input(id='input-text', type='text')),
    html.Button('Submit', id='submit-val', n_clicks=0),
    html.Div(id='result-text')
])


@callback(
    Output('result-text', 'children'),
    Input('submit-val', 'n_clicks'),
    State('input-text', 'value')
)
def update_output(_n_clicks, input_text):
    if input_text is not None and len(input_text) > 0:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": input_text}],
            max_tokens=100,
        )
        return completion["choices"][0]["message"]["content"]
    return ""


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
