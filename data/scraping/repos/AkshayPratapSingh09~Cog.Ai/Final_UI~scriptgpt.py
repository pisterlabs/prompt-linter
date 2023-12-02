# Bring in deps
#streamlit run your_app.py --server.port 8888
import csv
import matplotlib.pyplot as plt
from io import BytesIO
import os 
import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 
import os 
from apikey import apikey 
os.environ['OPENAI_API_KEY'] = apikey


# Create Streamlit app
st.set_page_config(
    page_title='Script GPT',
    page_icon='🎬',
    layout='wide'
)

# Inject custom CSS
custom_css = """
<style>
body {
    background-color: black;
    color: white;
}
.navbar {
    background-color: #1f1f1f;
    padding: 10px 0;
}
.navbar a {
    color: white;
    text-decoration: none;
    padding: 10px 20px;
}
.navbar a:hover {
    background-color: #333333;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# App framework
st.title('Script GPT')
prompt = st.text_input('Plug in your prompt here') 

# Prompt templates
title_template = PromptTemplate(
    input_variables=['topic'], 
    template='write me a youtube video title about {topic}'
)

script_template = PromptTemplate(
    input_variables=['title', 'wikipedia_research'], 
    template='write me a youtube video script based on this title TITLE: {title} while leveraging this wikipedia reserch:{wikipedia_research} '
)

# Memory 
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

# Llms
llm = OpenAI(temperature=0.9) 
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

wiki = WikipediaAPIWrapper()





# Initialize counters
vp, p, np, s, a,c = 0, 0, 0, 0, 0,0

# Create/open the CSV file
csv_file = 'feedback_data.csv'
with open(csv_file, mode='a+', newline='') as file:
    writer = csv.writer(file)
    file.seek(0)
    existing_data = list(csv.reader(file))
    for row in existing_data[1:]:
        category, count = row
        if category == 'Very Poor':
            vp += int(count)
            c-=2
        elif category == 'Poor':
            p += int(count)
            c-=1
        elif category == 'Not Poor':
            np += int(count)
        elif category == 'Satisfy':
            s += int(count)
            c+=0.5
        elif category == 'Accurate':
            a += int(count)
            c+=1

    if len(existing_data) == 1:
        writer.writerow(['Category', 'Count'])  # Write header if file is empty

# Navigation bar with tabs
tabs = ['Very Poor', 'Poor', 'Not Poor', 'Satisfy', 'Accurate']
selected_tab = st.radio('Feedback 4 Improvement', tabs)

# Update counters and save data to CSV file
if selected_tab == 'Very Poor':
    vp += 1
    c-=2
elif selected_tab == 'Poor':
    p += 1
    c-=1
elif selected_tab == 'Not Poor':
    np += 1
    c+=0
elif selected_tab == 'Satisfy':
    s += 1
    c+=0.5
elif selected_tab == 'Accurate':
    a += 1
    c+=1

# Save updated data to CSV file
with open(csv_file, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([selected_tab, 1])
print(c)
# Generate pie chart
data = [vp, p, np, s, a]
labels = ['Very Poor', 'Poor', 'Not Poor', 'Satisfy', 'Accurate']

fig, ax = plt.subplots()
ax.pie(data, labels=labels, autopct='%1.1f%%', startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.

st.title('Feedback Score')
st.write(c)
#st.radio('Feedback 4 Improvement', tabs)
#st.pyplot(fig)

# Show stuff to the screen if there's a prompt
if prompt: 
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt) 
    script = script_chain.run(title=title, wikipedia_research=wiki_research)

    st.write(title) 
    st.write(script) 

    with st.expander('Title History'): 
        st.info(title_memory.buffer)

    with st.expander('Script History'): 
        st.info(script_memory.buffer)

    with st.expander('Wikipedia Research'): 
        st.info(wiki_research)

