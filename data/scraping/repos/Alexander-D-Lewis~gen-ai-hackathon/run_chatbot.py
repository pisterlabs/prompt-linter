# std
import os
import base64
from io import BytesIO

# third party
from PIL import Image
from dash import html
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from dash import Dash, html, Input, Output, State, callback, dcc
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI

from langchain import FAISS

# API keys can be found here: https://platform.openai.com/account/api-keys
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")


def image(path, enc_format="png", **kwargs):
    """
    Converts a PIL (Python Image Library) Image into base64 string for HTML displaying
    :param im: PIL Image object
    :param enc_format: The image format for displaying. If saved the image will have that extension.
    :return: base64 encoding
    """
    image = Image.open(path)
    buff = BytesIO()
    image.save(buff, format=enc_format, **kwargs)
    encoded = base64.b64encode(buff.getvalue()).decode("utf-8")
    return html.Img(
        className="image", src="data:image/png;base64, " + encoded, **kwargs
    )

prompts_list = ["What lessons can we learn from other projects working in ...",
                "Can you give me examples of ...",
                "Can you summarise the lessons from ..."]

external_js = [
    "https://code.jquery.com/jquery-3.4.1.slim.min.js",
    "https://cdn.jsdelivr.net/npm/popper.js@1.16.0/dist/umd/popper.min.js",
    "https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/js/bootstrap.min.js",
]
lato_font = "https://fonts.googleapis.com/css2?family=Lato:ital,wght@0,100;0,300;0,400;0,700;0,900;1,100;1,300;1,400;1,700;1,900"



# Prompt engineering
template = """The following is a conversation between an international development professional and an AI. The AI I will act as an assistant who provides advice to international development professionals. The international development professional will ask a question and the AI will respond. The AI should respond with an answer to the question and ask follow up questions to better understand what the international development professional is looking for and what they would find useful. The answer must include references to the documents where the information was sourced. All your answers should adhere to the civil service code. Please also define any acronyms you use. Along with the answer you provide, can you give me potential follow up questions that i could ask to learn more? If my request is vague, please ask me to clarify. Please add new lines between each paragraph in the return. Can you provide the document names and authors for the sources you used at the bottom of the reply.
Please consider the chat history:{chat_history}
And please consider the following context" {context}
Human: {question} 
AI Assistant:"""

PROMPT = PromptTemplate(input_variables=["chat_history", "question", "context"], template=template)


# Chatbot engineering
chat = ChatOpenAI(temperature=0, openai_api_key=OPENAI_KEY,
              model="gpt-4")


model = "sentence-transformers/all-mpnet-base-v2"
hf = HuggingFaceEmbeddings(model_name=model)
db = FAISS.load_local("faiss_index_100", hf)
retriever = db.as_retriever(search_kwargs={"k": 4, "fetch_k": 8})

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

conversation = ConversationalRetrievalChain.from_llm(llm=chat, memory=memory, verbose=True,
                                            retriever=retriever,
                                            combine_docs_chain_kwargs={"prompt": PROMPT})


# Dash board code
app = Dash(__name__,
           external_stylesheets=[dbc.themes.FLATLY, lato_font],
           external_scripts=external_js,)


app.layout = html.Div([
    html.Div([
              dmc.Title(children=[image("assets/logo.png", **{"width": "65px",
                                                            "height": "50px"}),

                                  "  AI.d Chat  ",
                                  image("assets/ukaid.jpeg", enc_format="jpeg", **{"width": "60px",
                                                            "height": "60px"}),])                                  
                                  ],style={'textAlign': 'center',
                                                          "padding": "10px"}),
                                        
        html.Br(),

        dbc.Container(
                    fluid=True,
                    children=[
                        dbc.Row(
                            [
                                dbc.Col(
                                    width=1,
                                ),
                                dbc.Col(
                                    width=10,
                                    children=dbc.Card(
                                        [
                                            dbc.CardHeader(children=[dcc.Markdown("""**Note:** This chatbot is a prototype and is not intended to be used for any official purposes. 
                                            The chatbot is trained on a small sample of publicly available documents and, although some effort has been made to prevent hallucinations, we cannot guarantee the accuracy of the produced content.""")]),
                                            dbc.CardBody([
                                                dmc.Alert(id="mainresponse", color="white", style={"background-color":"white"}),
                                                html.Br(),
                                                html.Div([
                                                dbc.Button(id="prompt_1_button", children=prompts_list[0], value=prompts_list[0], n_clicks=0, className="option-button"),
                                                dbc.Button(id="prompt_2_button", children=prompts_list[1], value=prompts_list[1], n_clicks=0, className="option-button"),
                                                dbc.Button(id="prompt_3_button", children=prompts_list[2], value=prompts_list[2], n_clicks=0, className="option-button"),

                                                ], style={'textAlign': 'center',
                                                          "align-content":"space-between", "width":"100%"}),
                                                html.Br(),
                                                dbc.InputGroup([
                                                    dbc.Input(id='prompt', value="", placeholder='Your prompt ...', type='text'),
                                                    dbc.Button(id='sendPrompt', children=">", color="success", n_clicks=0, className="option-button"),
                                                    ],
                                                ),

                                                
                                            ], style={"width":"100%"}),
                                        ],style={"width":"100%"},
                                    )
                                ),
                                dbc.Col(
                                    width=1,
                                ),
                            ]
                        )
                    ]
                ),

    ]
)

@callback(Output(component_id='prompt', component_property='value'),
          Output("prompt_1_button", "n_clicks"),
          Output("prompt_2_button", "n_clicks"),
          Output("prompt_3_button", "n_clicks"),
          Input("prompt_1_button", "n_clicks"),
          State("prompt_1_button", "value"),
          Input("prompt_2_button", "n_clicks"),
          State("prompt_2_button", "value"),
        Input("prompt_3_button", "n_clicks"),
          State("prompt_3_button", "value"),
          prevent_initial_call=True)
def update_prompt(clicks1, value1, clicks2, value2, clicks3, value3):  
    if clicks1>0:
        return value1.strip(), 0, 0, 0
    if clicks2>0:
        return value2.strip(), 0, 0, 0
    if clicks3>0:
        return value3.strip(), 0, 0, 0



@callback(
    Output(component_id='mainresponse', component_property='children'),
    Input(component_id='sendPrompt', component_property='n_clicks'),
    State(component_id='prompt', component_property='value'),
    State(component_id='mainresponse', component_property='children'),
)
def call_openai_api(n, human_prompt, existingchildren):
    if n==0:
        return "", "", []
    else:
        result_ai = conversation.run({"question":human_prompt, "chat_history":memory})

        human_output = f"You asked: {human_prompt}"
        chatbot_output = f"{result_ai}"

        new_children = existingchildren+ [html.Br(),
                                          dmc.Alert(children=[dcc.Markdown(human_output)], title="You",
                                                    className="you-alert-box"),
                                            html.Br(),
                                            dmc.Alert(children=[dcc.Markdown(chatbot_output)], title="AI.d",
                                                      className="bot-alert-box"),
                                             html.Br(),]
        


        return new_children


if __name__ == '__main__':
    app.run_server(debug=True)