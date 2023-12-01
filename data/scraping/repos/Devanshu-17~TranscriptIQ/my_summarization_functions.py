from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2
import streamlit as st


def summarize_with_cohere(text):
    # Cohere credentials and model details
    PAT = st.secrets["PATS"]
    USER_ID = 'cohere'
    APP_ID = 'summarize'
    MODEL_ID = 'cohere-summarize'
    MODEL_VERSION_ID = 'bc1d5f9cc2834571b1322f572aca2305'


    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)

    metadata = (('authorization', 'Key ' + PAT),)

    # Create the input for Cohere summarization
    input_data = resources_pb2.Input(
        data=resources_pb2.Data(
            text=resources_pb2.Text(
                raw=text
            )
        )
    )

    post_model_outputs_response = stub.PostModelOutputs(
        service_pb2.PostModelOutputsRequest(
            user_app_id=resources_pb2.UserAppIDSet(user_id=USER_ID, app_id=APP_ID),
            model_id=MODEL_ID,
            version_id=MODEL_VERSION_ID,
            inputs=[input_data]
        ),
        metadata=metadata
    )

    if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
        raise Exception(f"Post model outputs failed, status: {post_model_outputs_response.status.description}")

    # Extract the summary from Cohere response
    summary = post_model_outputs_response.outputs[0].data.text.raw
    return summary

import spacy
from pyecharts import options as opts
from pyecharts.charts import Graph, Page
from pyecharts.globals import ThemeType
import emoji

def ner_spacy(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return doc.ents

def filter_special_chars(text):
    return emoji.demojize(text)

def get_graph(transcription_text, save_path, DEBUG = False):
    """Generates the graph with pyecharts and save that grapn in save_path

    Parameters
    ----------
    transcription_text : str
        Transcription text from youtube for doing NER. 
    save_path : str (Path that should end with HTML) 
        save path should refer to the file which has to be saved to (.html) format file
    """
    DEBUG = True 
    data = {}
    transcription_text = filter_special_chars(transcription_text)
    for ent in ner_spacy(transcription_text):
        if ent.label_ not in data.keys():
            data[ent.label_] = [ent.text]
        else:
            data[ent.label_].append(ent.text)



    category_list = list(data.keys())
    # prepare categories
    categories = [
        opts.GraphCategory(name=n)
        for i, n in enumerate(category_list)
    ]

    # prepare nodes
    nodes = []
    for key, values in data.items():
        values = list(set(values))  # Removing duplicates by converting the list to set and then back to a list.
        
        for value in values[:10]:
            nodes.append(
                opts.GraphNode(
                    name=f"{value}", 
                    symbol_size=10,
                    value=value, 
                    category=key,
                    label_opts=opts.LabelOpts(is_show=True)
                )
            )

    # Add key nodes
    for key in data.keys():
        nodes.append(
            opts.GraphNode(
                name=key, 
                symbol_size=20, 
                category=key, 
                label_opts=opts.LabelOpts(is_show=True)
            )
        )

    # prepare links
    links = []
    for key, values in data.items():
        values = list(set(values))  # Removing duplicates by converting the list to set and then back to a list.
        
        for value in values:
            links.append(opts.GraphLink(source=key, target=f"{value}"))
    if DEBUG : 
        print('=' * 15 + 'links' + '=' * 15) 
        print(links)
        print('='* 30)
    # Add links between key nodes
    prev_key = None
    for key in data.keys():
        if prev_key:
            links.append(opts.GraphLink(source=prev_key, target=key))
        prev_key = key

    (
        Graph(init_opts=opts.InitOpts(theme=ThemeType.DARK))
        .add(
            "",
            nodes=nodes,
            links=links,
            categories=categories,
            layout="force",
            repulsion= 800, 
            is_rotate_label=True,
            linestyle_opts=opts.LineStyleOpts(color="source", curve=0.3),
            label_opts=opts.LabelOpts(position="right"),
        )
        .set_global_opts(title_opts=opts.TitleOpts(title=""))
        .render(save_path)
    )