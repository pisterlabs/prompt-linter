import streamlit as st
import pandas as pd
import openai
import os

# create a stargate themed app in streamlit
st.title('Stargate Atlantis')
st.header('Vicky, you are now part of new Stargate Atlantis episodes')
st.subheader('Select a plot from the list to see what happens in the next exciting episode of Stargate Atlantis')

# add images side by side
col1, col2, col3 = st.columns(3)
with col1:
    st.image('https://ic.pics.livejournal.com/neevebrody/11550022/524545/524545_original.gif', width=200)
    #st.write('Stargate Atlantis')

with col2:
    st.image('https://ic.pics.livejournal.com/neevebrody/11550022/362682/original.gif', width=200)
    #st.write('Stargate Atlantis')

with col3:
    st.image('https://ic.pics.livejournal.com/neevebrody/11550022/591621/591621_original.gif', width=200)
    #st.write('Stargate Atlantis')

#summarise text
def summarise(text_input):
    openai.api_key = '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' 
    
    #text_input = text_input

    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=text_input,
        temperature=0.7,
        max_tokens=2000,
        top_p=1,
        best_of = 4,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response['choices'][0]['text']

plot_twist = st.multiselect('Plot Twists', ['The team discovers a Stargate on a planet that is about to be destroyed by a supernova.', 'The team discovers a planet that is inhabited by humans.', 'The team discovers a planet that is inhabited by aliens.', 'The team discovers a planet that is inhabited by both humans and aliens.', 'The team discovers a planet that is inhabited by Goauld.', 'The team discovers a planet that is inhabited by Wraith.', 'The team discovers a planet that is inhabited by Ancients.', 'The team discovers a planet that is uninhabitable.', 'The team is attacked by a Goauld ship.', 'The team is attacked by a Wraith ship.', 'The team is attacked by an Ancient ship.', 'The team is stranded on a planet with no Stargate.', ' The team is attacked by a human ship.', 'The team discovers a Goauld ship.', 'The team discovers a Zpm.', 'The team discovers an Ancient ship.', 'The team discovers a Wraith ship.', 'The team discovers a planet with a Stargate that is about to be destroyed by a supernova.', 'The team is stranded on a planet with no food or water.', 'The team is stranded on a planet with no way to contact Earth.'])
plot_twist = ', '.join(plot_twist)

# create an empty dataframe with the column names
df = pd.DataFrame(columns=['Plot Twists','Synopsis'])

text_input = f"Generate a two-paragraph episode synopsis for Stargate Atlantis including Victoria Martin and John Sheppard as the main characters\ninclude a plot twist about:{plot_twist}\nJohn falls in love with Victoria.\n\n\n"
#display the prompt
#st.write(text_input)

# create a button
button = st.button('Generate Episode Synopsis')
# if the button is clicked
if button:
    with st.spinner('Generating Episode Synopsis...'):
        summary = summarise(text_input)
        #save the summary to a variable and use it to display the summary
        st.write(summary)
        love = summarise(summary+"Describe a love scene between Victoria and John in Stargate Atlantis:")
        st.write(love)
        end = summarise(summary+love + 'How does this stargate episode end?')
        st.write(end)
        next = summarise(summary+love+end+'What will the next episode be about with Victoria and John?')
        st.write(next)
# add a footer
st.subheader('Created with love by: Elle Neal')
