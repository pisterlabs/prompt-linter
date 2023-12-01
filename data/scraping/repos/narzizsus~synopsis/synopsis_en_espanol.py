import streamlit as st
import openai
import json
import pandas as pd

# Get the API key from the sidebar called OpenAI API key
user_api_key = st.sidebar.text_input("OpenAI API key", type="password")

client = openai.OpenAI(api_key=user_api_key)
prompt = """Act as an AI movie director in Spanish. You will receive a piece of movie synopsis and you should extract every scenes in that synopsis and give who is the actor  of each scene and what is the name of scene.
            List the scenes in a JSON array, one scene per line.
            Each scene should have 5 fields:
- “número_de_escena” - the number of plots in the synopsis
- “escena” - the text of scene’s name that tells us what happened in scene
- “protagonista(s)” - the text of actor name
- “acción_clave” - the list of words in C2 level that evoke our mind to the scene’s name
- “narración” - a comment about the scene

Don't say anything at first. Wait for the user to say something.
        """    

st.title('Scene Extraction')
st.markdown('Input the synopsis that you want to extract. \n\
            The AI will give you scenes and C2 vocabularies.')

user_input = st.text_area("Enter synopsis:", "Your text here")

# submit button after text input
if st.button('Submit'):
    messages_so_far = [
        {"role": "system", "content": prompt},
        {'role': 'user', 'content': user_input},
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages_so_far
    )
    # Show the response from the AI in a box
    st.markdown('**AI response:**')
    suggestion_dictionary = response.choices[0].message.content


    sd = json.loads(suggestion_dictionary)

    print (sd)
    suggestion_df = pd.DataFrame.from_dict(sd)
    print(suggestion_df)
    st.table(suggestion_df)
