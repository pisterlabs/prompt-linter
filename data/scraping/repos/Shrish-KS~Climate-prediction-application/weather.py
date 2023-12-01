import streamlit as st
import datetime,requests
from plotly import graph_objects as go
import pandas as pd
import streamlit as st
from streamlit_chat import message
import env
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain import ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import create_csv_agent
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import openai

st.set_page_config(page_title='clither', page_icon=":cloud:")

st.title("THE WEEKLY PREDICTED WEATHER FORECAST 🌧️🌥️⚡️")
st.sidebar.header("USER INPUTS:")
with st.sidebar:
    with st.form(key='my_form'):
        city=st.text_input("ENTER THE NAME OF THE CITY ")
        unit=st.selectbox("SELECT TEMPERATURE UNIT ",["Celsius","Fahrenheit"])
        speed=st.selectbox("SELECT WIND SPEED UNIT ",["Metre/sec","Kilometre/hour"])
        graph=st.radio("SELECT GRAPH TYPE ",["Bar Graph","Line Graph"])
        uploaded_file = st.file_uploader("Choose an excel file", type="xlsx")
        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file)
        submit_button = st.form_submit_button(label='Submit')

st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://a-static.besthdwallpaper.com/artistic-mountain-landscape-wallpaper-1920x1200-3172_6.jpg")
    }
  
    </style>
    """,
    unsafe_allow_html=True
)

if unit=="Celsius":
    temp_unit=" °C"
else:
    temp_unit=" °F"
    
if speed=="Kilometre/hour":
    wind_unit=" km/h"
else:
    wind_unit=" m/s"

api="9b833c0ea6426b70902aa7a4b1da285c"
url=f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api}"
response=requests.get(url)
x=response.json()
    

try:
    lon=x["coord"]["lon"]
    lat=x["coord"]["lat"]
    ex="current,minutely,hourly"
    url2=f'https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&exclude={ex}&appid={api}'
    res=requests.get(url2)
    y=res.json()

    maxtemp=[]
    mintemp=[]
    pres=[]
    humd=[]
    wspeed=[]
    desc=[]
    cloud=[]
    rain=[]
    dates=[]
    sunrise=[]
    sunset=[]
    cel=273.15
    
    for item in y["daily"]:
        
        if unit=="Celsius":
            maxtemp.append(round(item["temp"]["max"]-cel,2))
            mintemp.append(round(item["temp"]["min"]-cel,2))
        else:
            maxtemp.append(round((((item["temp"]["max"]-cel)*1.8)+32),2))
            mintemp.append(round((((item["temp"]["min"]-cel)*1.8)+32),2))

        if wind_unit=="m/s":
            wspeed.append(str(round(item["wind_speed"],1))+wind_unit)
        else:
            wspeed.append(str(round(item["wind_speed"]*3.6,1))+wind_unit)

        pres.append(item["pressure"])
        humd.append(str(item["humidity"])+' %')
        
        cloud.append(str(item["clouds"])+' %')
        rain.append(str(int(item["pop"]*100))+'%')

        desc.append(item["weather"][0]["description"].title())

        d1=datetime.date.fromtimestamp(item["dt"])
        dates.append(d1.strftime('%d %b'))
        
        sunrise.append( datetime.datetime.utcfromtimestamp(item["sunrise"]).strftime('%H:%M'))
        sunset.append( datetime.datetime.utcfromtimestamp(item["sunset"]).strftime('%H:%M'))

    def bargraph():
        fig=go.Figure(data=
            [
            go.Bar(name="Maximum",x=dates,y=maxtemp,marker_color='crimson'),
            go.Bar(name="Minimum",x=dates,y=mintemp,marker_color='navy')
            ])
        fig.update_layout(xaxis_title="Dates",yaxis_title="Temperature",barmode='group',margin=dict(l=70, r=10, t=80, b=80),font=dict(color="white"))
        st.plotly_chart(fig)
    
    def linegraph():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=mintemp, name='Minimum '))
        fig.add_trace(go.Scatter(x=dates, y=maxtemp, name='Maximimum ',marker_color='crimson'))
        fig.update_layout(xaxis_title="Dates",yaxis_title="Temperature",font=dict(color="white"))
        st.plotly_chart(fig)
        
    icon=x["weather"][0]["icon"]
    current_weather=x["weather"][0]["description"].title()
    
    if unit=="Celsius":
        temp=str(round(x["main"]["temp"]-cel,2))
    else:
        temp=str(round((((x["main"]["temp"]-cel)*1.8)+32),2))
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("## Current Temperature ")
    with col2:
        st.image(f"http://openweathermap.org/img/wn/{icon}@2x.png",width=70)

    
    col1, col2= st.columns(2)
    col1.metric("TEMPERATURE",temp+temp_unit)
    col2.metric("WEATHER",current_weather)
    st.subheader(" ")
    
    if graph=="Bar Graph":
        bargraph()
        
    elif graph=="Line Graph":
        linegraph()

        
    table1=go.Figure(data=[go.Table(header=dict(
                values = [
                '<b>DATES</b>',
                '<b>MAX TEMP<br>(in'+temp_unit+')</b>',
                '<b>MIN TEMP<br>(in'+temp_unit+')</b>',
                '<b>CHANCES OF RAIN</b>',
                '<b>CLOUD COVERAGE</b>',
                '<b>HUMIDITY</b>'],
                line_color='black', fill_color='royalblue',  font=dict(color='white', size=14),height=32),
    cells=dict(values=[dates,maxtemp,mintemp,rain,cloud,humd],
    line_color='black',fill_color=['royalblue',['black', 'crimson']*7], font_size=14,height=32
        ))])

    table1.update_layout(margin=dict(l=10,r=10,b=10,t=10),height=328)
    st.write(table1)
    
    table2=go.Figure(data=[go.Table(columnwidth=[1,2,1,1,1,1],header=dict(values=['<b>DATES</b>','<b>WEATHER CONDITION</b>','<b>WIND SPEED</b>','<b>PRESSURE<br>(in hPa)</b>','<b>SUNRISE<br>(in UTC)</b>','<b>SUNSET<br>(in UTC)</b>']
                ,line_color='black', fill_color='royalblue',  font=dict(color='white', size=14),height=36),
    cells=dict(values=[dates,desc,wspeed,pres,sunrise,sunset],
    line_color='black',fill_color=['royalblue',['black', 'crimson']*7], font_size=14,height=36))])
    
    table2.update_layout(margin=dict(l=10,r=10,b=10,t=10),height=360)
    st.write(table2)
    
    try:
        st.header("Data from excel: ")
        styled_df = df.style.set_properties(**{'border-color': 'black','color': 'white','background-color':'royalblue'})
        st.write(styled_df)
    except:
        pass
    


except KeyError:
    pass
#     st.error("Enter a valid city!")
def load_chain():
    """Logic for loading the chain you want to use should go here."""
    llm = OpenAI(openai_api_key=st.secrets["api_key"],temperature=0)
    chain = ConversationChain(llm=llm)
    return chain

chain = load_chain()

# From here down is all the StreamLit UI.

st.subheader("LANGCHAIN CHAT MODEL")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []
text="WELCOME BACK WHAT WOULD YOU LIKE TO DO?"
st.markdown(f"<h3 style='color:grey'>{text}</h3>", unsafe_allow_html=True)
with st.expander("1.CHAT BOT"):
        input_text = st.text_input("You: ", "Hello, how are you?", key="input")
        user_input = input_text
        if user_input:
            output = chain.run(input=user_input)

            st.session_state.past.append(user_input)
            st.session_state.generated.append(output)

        if st.session_state["generated"]:

            for i in range(len(st.session_state["generated"]) - 1, -1, -1):
                message(st.session_state["generated"][i], key=str(i))
                message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")

    
with st.expander("2.ANALYSIS"):
    genre = st.radio(
    "How do You Want to Analyse",
    ('1.From a Dataset Which you Want', '2.From a Dataset which is Available'))

    if genre == "1.From a Dataset Which you Want":
        # data = pd.read_excel("environment.xlsx")
        # chart = alt.Chart(data).mark_line().encode(x='Element', y='Area')
        # st.altair_chart(chart, use_container_width=True)
        openai.api_key = st.secrets["api_key"]
        model_engine = "text-davinci-002"

        # Define the initial prompt for the OpenAI model
        model_prompt = "The answer to your question is:"

        # Create a file uploader widget for Excel files
        excel_file = st.file_uploader("Upload Excel file", type=["xlsx"])

        # If a file is uploaded, load it into a pandas dataframe
        if excel_file is not None:
            df = pd.read_excel(excel_file)

            # Extract the column names from the dataframe
            # Extract the data you need from the dataframe
            input_data = df

            # Define a function to generate OpenAI responses
            def generate_answer(input_str):
                prompt_str = model_prompt + " " + input_str
                response = openai.Completion.create(
                    engine=model_engine,
                    prompt=prompt_str,
                    temperature=0.5,
                    max_tokens=1024,
                    n=1,
                    stop=None,
                    timeout=60,
                )
                return response.choices[0].text.strip()

            # Allow the user to ask questions about the data
            question = st.text_input("Ask a question")

            # If a question is asked, generate an OpenAI response
            if question:
                # Construct the prompt for the OpenAI model
                prompt = model_prompt + " "
                for input_str in input_data:
                    prompt += f"Q: What is the {question} for {input_str}? A: "
                
                # Generate the OpenAI response and display it in the Streamlit app
                answer_list = []
                for input_str in input_data:
                    prompt_str = prompt + input_str
                    answer = generate_answer(prompt_str)
                    answer_list.append(answer)
                st.write("OpenAI's answer:")
                st.write(answer_list)

    else:
        st.write("Sorry for the Incovinience the data is not available")

    

        
    
    

def get_text():
    pass
st.header(' ')
st.header(' ')
st.markdown("BY 🖥️KAMALESH, 💻ARUN , SHRISH💪🏻")
