import streamlit as st
import webvtt
import openai
from io import StringIO
import tiktoken

st.set_page_config(page_title="Summarize",page_icon="📝")

st.title('📝 Teams meeting summarizer')

# Set the API key for the openai package
openai.api_key = st.secrets['OPEN_AI_KEY']

chunk_size = 3000

def num_tokens(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding_name = 'cl100k_base'
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def slice_string(text: str) -> list[str]:
    # Split text into chunks based on space or newline
    chunks = text.split()

    # Initialize variables
    result = []
    current_chunk = ""

    # Concatenate chunks until the total length is less than 4096 tokens
    for chunk in chunks:
        # if len(current_chunk) + len(chunk) < 4096:
        if num_tokens(current_chunk+chunk) < chunk_size:
            current_chunk += " " + chunk if current_chunk else chunk
        else:
            result.append(current_chunk.strip())
            current_chunk = chunk
    if current_chunk:
        result.append(current_chunk.strip())

    return result

def summarize(context: str, model: str, convo: str) -> str:
    """Returns the summary of a text string."""
    context = context
    completion = openai.ChatCompletion.create(
    model = model,
      messages=[
        {'role': 'system','content': context},
        {'role': 'user', 'content': convo}
            ]
    )
    return completion.choices[0].message.content

def refine(summary: str,context: str, model: str, chunk: str) -> str:
    """Refine the summary with each new chunk of text"""
    context = "Refine the summary with the following context: " + summary
    summary = summarize(context,model,chunk)
    return summary

# context = st.text_input('Context','summarize the following conversation')
context = 'summarize the following conversation'
# model = st.radio('Model',('gpt-3.5-turbo','gpt-4'))
model = 'gpt-3.5-turbo'
file = st.file_uploader('Upload Teams VTT transcript',type='vtt')
maxtokens = {'gpt-3.5-turbo': 4096,'gpt-4': 8192 }
# st.write(maxtokens[model])

if file is not None:
    data = StringIO(file.getvalue().decode('utf-8'))
    chat = webvtt.read_buffer(data)
    # data = file.getvalue().decode('utf-8')
    # with open('vtt/'+file.name,'w') as f:
    #     f.write(data)
    # caption = webvtt.read('vtt/'+file.name)
    part = st.checkbox('include participants')
    time = st.checkbox('include time')
    str = []
    for caption in chat:
        if part & time:
            str.append(f'{caption.start} --> {caption.end}')
            str.append(caption.raw_text)
        elif time:
            str.append(f'{caption.start} --> {caption.end}')
            str.append(caption.text)
        elif part:
            str.append(caption.raw_text)
        else:
            str.append(caption.text)
    sep = '\n'
    convo = sep.join(str)
        
    convo = st.text_area('vtt file content',convo)
    toknum = num_tokens(convo)
    st.write(toknum,'tokens')

    if st.button('summarize'):

        if (toknum > maxtokens[model]-1000):
            # st.write(f'Text too long please prune to fit under {maxtokens[model]-1000} tokens')
            chunks = slice_string(convo)
            summary = summarize(context,model,chunks[0])
            for chunk in chunks[1:]:
                summary = refine(summary,context,model,chunk)
            st.write(summary)
        else:
            st.write(summarize(context,model,convo))

else:
    with open('vtt/YannMike_2023-03-08.vtt') as f:
        st.download_button(
            label="Sample VTT file",
            data=f,
            file_name="sample.vtt",
            mime="text/vtt"
          )
