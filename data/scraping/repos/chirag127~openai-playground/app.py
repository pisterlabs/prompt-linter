import streamlit as st
import openai
import cohere


st.title("Text Generation")
# Create a sidebar
st.sidebar.title("Options")

# Company dropdown menu
companies = ["OpenAI","Cohere"]
company = st.sidebar.selectbox("Select the company", companies)

# API key input
api_key = st.sidebar.text_input("Enter the API key")
openai.api_key = api_key

# Prompt input
prompt = st.text_area("Enter the prompt", height=200)

if company == "Cohere":
    models = ["xlarge"]
elif company == "OpenAI":
    models = ["code-davinci-002", "code-cushman-001","text-davinci-003", "text-curie-001", "text-babbage-001", "text-ada-001"]

model = st.sidebar.selectbox("Select the model", models)

# Temperature slider
temperature = st.sidebar.slider("Temperature", min_value=0.00, max_value=1.00, step=0.01, value=0.00)

# Max length slider
max_length = st.sidebar.slider("Max Length", min_value=1, max_value=8000, step=10, value=250)


# make text box too for max length
max_length = st.sidebar.text_input("Enter the max length", max_length)


# convert max_length to int
max_length = int(max_length)

# Stop sequence input
stop_sequence = st.sidebar.text_input("Enter the stop sequence separated by commas")

stop_sequence = stop_sequence.split(",")


if stop_sequence == [""]:
    stop_sequence = []

# Top P slider
top_p = st.sidebar.slider("Top P", min_value=0.00, max_value=1.00, step=0.01, value=0.00)

# Frequency penalty slider
frequency_penalty = st.sidebar.slider("Frequency Penalty", min_value=0.00, max_value=1.00, step=0.01, value=0.00)

# Presence penalty slider
presence_penalty = st.sidebar.slider("Presence Penalty", min_value=0.00, max_value=1.00, step=0.01, value=0.00)


if button := st.button("Generate"):
    if company == "Cohere":
        client = cohere.Client(api_key)
        response = client.generate(
            model=model,
            prompt=prompt,
            max_tokens=max_length,
            temperature=temperature,
            k=0,
            p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop_sequences=stop_sequence,
        )
        result = response.generations[0].text



    elif company == "OpenAI":

        if stop_sequence != []:


            completions = openai.Completion.create(
        engine=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_length,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop_sequence,
    )
        else:
            completions = openai.Completion.create(
        engine=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_length,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
    )

        result = completions.choices[0].text

    st.code(result)


    
