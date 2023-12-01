import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain

import openai


def generate_program():
        generate_button.empty()
        st.session_state.generate = True

# Set page title
st.set_page_config(
    page_title="Workout plan generator",
)



# title of content
st.title("LLM Personal Trainer")
st.header("Workout Plan Generator")


# Initialise the session state variables if they dont exist
if "generate" not in st.session_state:
    st.session_state.generate = False

if 'currentkey' not in st.session_state:
     st.session_state.currentkey = ''

if 'validate' not in st.session_state:
    st.session_state.validate = False

if 'validate_count' not in st.session_state:
    st.session_state.validate_count = 0


try:
    st.session_state.currentkey = st.secrets["open_ai_key"]
except:
    pass

openai.api_key = st.session_state.currentkey

def validate():
    try:
        text_input = st.session_state.input
        openai.api_key = text_input
        st.session_state.validate_count = st.session_state.validate_count + 1
        response = openai.Completion.create(
            engine="davinci",
            prompt="validating openaikey",
            max_tokens=5,
            
        )
        st.session_state.currentkey = text_input
        st.session_state.validate = False
    except:
        side_validation = st.sidebar.text('OPEN AI API key not valid')



with st.sidebar.form('Enter OPEN API key'):
    st.text_input("Enter open api key",key='input')
    st.form_submit_button('Validate key', on_click=validate)


if st.session_state.currentkey:

    side_text = st.sidebar.text(
        f'Current OPEN AI API Key is valid'
        )


if st.session_state.currentkey:
    weeks = st.number_input("How long should the program be (weeks)",min_value=1,max_value=12,step=1)
    days = st.number_input("how many days per week?",min_value=1,max_value=7,step=1)
    accessory = st.number_input("how many accessory lifts per session",min_value=0,max_value=5,step=1)
    squat = st.number_input("1 rep max squat?",min_value=0,max_value=1000,step=1)
    bench = st.number_input("1 rep max bench press?",min_value=0,max_value=1000,step=1)
    deadlift = st.number_input("1 rep max conventional deadlift?",min_value=0,max_value=1000,step=1)
    units = st.selectbox('units',['kilograms','pounds'])




    generate_button = st.empty()
    generate_button.button("generate program",type='primary',on_click=generate_program)

    if st.session_state.generate:
        with st.spinner("generating program"):
            output_concat = ""
            
            llm = ChatOpenAI(model='gpt-3.5-turbo',temperature=0.5,openai_api_key=st.session_state.currentkey)
            template = """
            Can you create a strength training program that focuses on back squat, bench press, and conventional deadlift

            number of sessions each week: {days}

            number of weeks: {weeks} 

            back squat 1 rep max: {squat} {units}

            bench press 1 rep max: {bench} {units}

            conventional deadlift 1 rep max: {deadlift} {units}

            number of accessory exercises each session: {accessory}

            Create the first week

            """

            
            promp = PromptTemplate(
                input_variables=['days','weeks','squat','units','bench','deadlift','accessory'],
                template=template
            )

            chain = LLMChain(llm=llm,prompt=promp)
            output = chain.run({'days':days,'weeks':weeks,'squat':squat,'units':units,'bench':bench,'deadlift':deadlift,'accessory':accessory})
            st.write(output)
            st.write('******************')

            output_concat = output_concat + output




            week_total = int(weeks)
            if week_total > 1:
                current_week = 2
                while current_week <= week_total:
                
                    
                    llm = ChatOpenAI(model='gpt-3.5-turbo',temperature=0.5,openai_api_key=st.session_state.currentkey)
                    template = """
                    create training program for week number {current_week} given information about last weeks training:
                    {program}
                    and the following details:
                    number of sessions each week: {days}
                    back squat 1 rep max: {squat} {units}

                    bench press 1 rep max: {bench} {units}

                    conventional deadlift 1 rep max: {deadlift} {units}

                    number of accessory exercises each session: {accessory}

                    make the program more challenging than the previous week.



                    """

                    
                    promp = PromptTemplate(
                        input_variables=['program','days','squat','units','bench','deadlift','accessory','current_week'],
                        template=template
                    )

                    chain = LLMChain(llm=llm,prompt=promp)
                    output = chain.run({'program':output,'days':days,'squat':squat,'units':units,'bench':bench,'deadlift':deadlift,'accessory':accessory,'current_week':current_week})
                    st.write(output)
                    st.write('******************')
                    current_week = current_week + 1
                    output_concat = output_concat + output
        st.session_state.generate = False

else:
     st.header('Enter your Open AI API key to use functionality')
        




    



