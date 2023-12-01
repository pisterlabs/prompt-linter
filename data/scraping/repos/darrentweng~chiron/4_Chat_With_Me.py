import streamlit as st
import cohere
from cohere.classify import Example 

NoLinesDisplayed = 8
NoLinesUsed = 2
co = cohere.Client(st.secrets["cohere_api"])

emotions = [
    Example("The order came 5 days early", "happy"), Example("The item exceeded my expectations", "happy"), 
    Example("I ordered more for my friends", "happy"), Example("I would buy this again", "happy"), 
    Example("I would recommend this to others", "happy"), Example("The package was damaged", "sad"), 
    Example("The order is 5 days late", "sad"), Example("The order was incorrect", "sad"), 
    Example("I want to return my item", "sad"), Example("The item\'s material feels low quality", "sad"),
    Example("How do you use my item", "confused"), Example("What is this item for", "confused"),
    Example("Why did I get this item", "confused"), Example("English makes no sense", "confused"),
    Example("I don't understand physics", "confused")]


st.title('Talk to me')

if 'chat_conv' not in st.session_state:
    st.session_state['chat_conv'] = ['Bot: Start typing below to chat with me.\n']

conv = st.session_state['chat_conv']

container_top = st.empty()
container_top.text(''.join(conv[-1*NoLinesDisplayed:]))

container_bottom = st.empty()
input = container_bottom.text_input("Your Message",value="",key="msg")

container_emotion = st.empty()
container_emotion.text("---")


if input != '':
    conv.append('You: '+input+'\n')
    chat_conv = conv[-1*NoLinesUsed:]
    # response = "Hello, I am a bot. I am here to help you. How can I help you?"
    response = co.generate(
        prompt=''.join(chat_conv)+"\Bot:",
        model='xlarge', max_tokens=20,   temperature=1.2,   k=0,   p=0.75,
        frequency_penalty=0,   presence_penalty=0, return_likelihoods='NONE',
        stop_sequences=["User:", "\n"]
    ).generations[0].text.strip()

    conv.append('Bot: '+response+'\n')
    container_top.text(''.join(conv[-1*NoLinesDisplayed:]))
    # input = container_bottom.text_input("Your Message",value="",key="placeholder")
    st.session_state['chat_conv'] = conv[-1*NoLinesDisplayed:]

    emotion = co.classify( model='large', inputs=[input], examples=emotions) 
    e0 = emotion.classifications[0].confidence[0]
    e1 = emotion.classifications[0].confidence[1]
    e2 = emotion.classifications[0].confidence[2]
    container_emotion.text(e0.label+":"+str(e0.confidence) + " - "+e1.label+":"+str(e1.confidence)+" - " + e2.label+":"+str(e2.confidence))
   
#st.write(response.generations[0].text)
