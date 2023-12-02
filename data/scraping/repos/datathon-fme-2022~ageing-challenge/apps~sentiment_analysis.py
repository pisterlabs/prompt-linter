import streamlit as st


def app():
    st.title("Sentiment Analysis")

    st.markdown(
        """
    After transcripting the text, we analyze it in order to predict whether the text is a real Emergency or just a False Alarm. To do so
    we will use Sentiment analysis (also known as opinion mining or emotion AI) is the use of natural language processing and text analysis, 
    to systematically identify, extract, quantify, and study affective states and subjective information.

    In our case the user case is to detect if what an old person is saying is an emergency or not

    """
    )

    st.markdown("We used cohere's Client to train a model. To ilustrate how is used, we present you some basic code to start this machine")

    st.code("""
# Imports of the document
import cohere
from cohere.classify import Example

co = cohere.Client('xxxxxxxxxxxxxxxxxxxx') # -> Main instance of coherence
    """)

    st.markdown(""" As every NLP model, it needs some sample to train the model. Some examples of the data used to train is the following: 
    - **Example One:** *"I need an ambulance"*  **Classification:** "Emergency"
    - **Example Two:** *"Please hurry, I am in pain"*  **Classification:** "Emergency"
    - **Example Three:** *Sorry, I have touched something that is not..."*  **Classification:** "False Alarm"
    """)

    st.code('''
voice_samples = [
    Example("I need an ambulance", "Emergency"),
    Example("Please hurry, I am in pain", "Emergency"),
    Example("Sorry, I have touched something that is not...", "False Alarm")
    # ...
]
''')

    st.markdown("And simply using the cohere model with this few examples, we can try on new data")

    st.code("""
inputs=["It's hurting so much, I can't even breath",
        "Please hurry, I fell down",
        "Oops I pressed the wrong button"]

response = co.classify(
  model='large',
  inputs=inputs,
  examples=voice_samples,
)
""")
    st.markdown("We get the following predictions: LET'S GOOO!!!")

    st.code("""
//Classification<prediction: "Emergency", confidence: 0.9993907>
//Classification<prediction: "Emergency", confidence: 0.9980647>
//Classification<prediction: "False Alarm", confidence: 0.9678431>""")

    with st.expander("Do you want to get this code? Go to:"):
        st.markdown(
            """
            Check it out! [GitHub](https://github.com/datathon-fme-2022/ageing-challenge)
            
            Hope you like it!

            """
        )
