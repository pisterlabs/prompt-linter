import base64
import os
import pickle

import cohere
import numpy as np
import pandas as pd
import streamlit as st
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multioutput import ClassifierChain
from sklearn.preprocessing import MultiLabelBinarizer

from utils import get_embeddings, seed_everything, streamlit_header_and_footer_setup

seed_everything(3777)

st.set_page_config(layout="wide")
streamlit_header_and_footer_setup()
st.markdown("## Sentiment Analysis 🥺")

model_name: str = 'multilingual-22-12'
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
co = cohere.Client(COHERE_API_KEY)


def train_and_save():
    full_df = pd.read_json("./data/xed_with_embeddings.json", orient='index')
    df = full_df

    mlb = MultiLabelBinarizer()

    X = np.array(df.embeddings.tolist())
    y = mlb.fit_transform(df.labels_text)
    classes = mlb.classes_
    print(classes)

    classes_mapping = {index: emotion for index, emotion in enumerate(mlb.classes_)}
    print(classes_mapping)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

    base_lr = LogisticRegression(solver='lbfgs', random_state=0)
    chain = ClassifierChain(base_lr, order='random', random_state=0)

    chain.fit(X_train, y_train)

    print(chain.score(X_test, y_test))
    pickle.dump(chain, open("./data/models/emotion_chain.pkl", 'wb'))



classes_mapping = {
    0: 'Anger',
    1: 'Anticipation',
    2: 'Disgust',
    3: 'Fear',
    4: 'Joy',
    5: 'Sadness',
    6: 'Surprise',
    7: 'Trust'
}
model_path = "./data/models/emotion_chain.pkl"

@st.cache
def setup():
    emotions2image_mapping = {
        'Anger': './data/emotions/anger.gif',
        'Anticipation': './data/emotions/anticipation.gif',
        'Disgust': './data/emotions/disgust.gif',
        'Fear': './data/emotions/fear.gif',
        'Joy': './data/emotions/joy.gif',
        'Sadness': './data/emotions/sadness.gif',
        'Surprise': './data/emotions/surprise.gif',
        'Trust': './data/emotions/trust.gif',
    }
    for key, value in emotions2image_mapping.items():
        with open(value, "rb") as f:
            emotions2image_mapping[key] = f.read()

    chain_model = pickle.load(open(model_path, 'rb'))
    return emotions2image_mapping, chain_model


emotions2image_mapping, chain_model = setup()

feeling_text = st.text_input("How are you feeling?", "")
top_k = st.slider("Top Emotions", min_value=1, max_value=len(classes_mapping), value=1, step=1)


def score_sentence(text: str, top_k: int = 5):
    print(f"Text: {text}")
    embeddings = torch.as_tensor(get_embeddings(co=co, model_name=model_name, texts=[text]), dtype=torch.float32)
    outputs = torch.as_tensor(chain_model.predict_proba(embeddings), dtype=torch.float32)
    probas, indices = torch.sort(outputs)

    probas = probas.cpu().numpy()[0][::-1]
    indices = indices.cpu().numpy()[0][::-1]

    cols = st.columns(top_k, gap="large")
    for i, (index, p) in enumerate(zip(indices[:top_k], probas[:top_k])):
        if i % 3 == 0:
            cols = st.columns(3, gap="large")

        emotion = classes_mapping[index]

        i = i % 3
        image_file = emotions2image_mapping.get(emotion, None)
        if image_file:
            image_gif = base64.b64encode(image_file).decode("utf-8")
            cols[i].markdown(
                f'<img src="data:image/gif;base64,{image_gif}" style="width:250px;height:250px;border-radius: 25%;">',
                unsafe_allow_html=True,
            )
            cols[i].markdown("---")
        cols[i].markdown(f"**{emotion}**: {p * 100:.2f}%")

        print(f"Predicted emotion: {emotion}, with probability: {p}")


if feeling_text:
    score_sentence(feeling_text, top_k=top_k)
