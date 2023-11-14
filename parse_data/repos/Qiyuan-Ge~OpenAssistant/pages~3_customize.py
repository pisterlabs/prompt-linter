import streamlit as st
from fastchat.conversation import conv_templates, get_conv_template

st.image("https://cdn.pixabay.com/photo/2023/02/11/16/32/ai-generated-7783344_640.jpg", width=256)

tab1, tab2, tab3 = st.tabs(["Prompt Template", "Custom Instructions", "Generate Params"])

if "customize_button" not in st.session_state:
    st.session_state.customize_button = True

def get_options():
    options = list(conv_templates.keys())
    options.remove(st.session_state.prompt_template)
    options.insert(0, st.session_state.prompt_template)
    
    return options

with tab1:
   st.markdown('**Select a conversation prompt template**')
   option = st.selectbox(
       label='Select a conversation prompt template',
       options=get_options(),
       label_visibility='collapsed',
    )
   st.session_state.prompt_template = option
   
   conv = get_conv_template(option)
   conv.append_message(conv.roles[0], "Hello!")
   conv.append_message(conv.roles[1], "Hi! How can I assist you today?")
   
   st.markdown(f'**{option} prompt template:**')
   st.text(conv.get_prompt())

with tab2:
    st.markdown('**What kind of assistant do you wish for?**')
    st.markdown('**Note:** the prompt you write below will overwrite the original system prompt.')
    if st.toggle('Turn on/off', value=st.session_state.customize_button, key="turn", label_visibility='hidden'):
        st.session_state.system_message = st.text_input(
            label="sys_prompt", 
            key="sys_prompt", 
            value=st.session_state.system_message,
            label_visibility='collapsed',
        )
    else:
        st.session_state.system_message = None

with tab3:
    st.markdown("""
    ### Max New Tokens
    Max New Tokens refers to the maximum number of new words or tokens that a language model can generate in a single response.

    For example, if the `max new tokens` value is set to 50, the model will generate a response with no more than 50 new words. If the response exceeds this limit, it may truncate or omit some text to fit within the specified maximum.
    """)
    st.session_state.generate_params['max_tokens'] = st.slider(
        'Max New Tokens', 
        min_value=32, 
        max_value=4096,
        step=16,
        value=st.session_state.generate_params['max_tokens'], 
        label_visibility="collapsed",
    )
    
    st.markdown("""
    ### Temperature
    Temperature is a setting that controls the randomness of a language model's output. 

    - A higher temperature makes the output more random and creative.
    - A lower temperature makes the output more focused and deterministic.

    Think of it as adjusting the "creativity" knob of the model to influence the diversity of generated text.
    """)
    st.session_state.generate_params['temperature'] = st.slider(
        'Temperature', 
        min_value=0.0, 
        max_value=1.0, 
        value=st.session_state.generate_params['temperature'], 
        label_visibility="collapsed",
    )
    
    st.markdown("""
    ### Top-p (Nucleus Sampling)
    Top-p, also known as Nucleus Sampling, is a technique used for controlling the diversity of generated text. 

    It determines the probability distribution of words to consider when generating text. When you set a `top-p` value (e.g., 0.8), the model considers only the most probable words that make up 80% of the cumulative probability.
    """)
    st.session_state.generate_params['top_p'] = st.slider(
        'Top-p', 
        min_value=0.0, 
        max_value=1.0, 
        label_visibility="collapsed",
        value=st.session_state.generate_params['top_p'], 
    )
