import os
import openai
import streamlit as vAR_st
from streamlit_chat import message


openai.api_key = os.environ["API_KEY"]


def GPT3Tasks(vAR_option):
    
    

    if vAR_option=='Work Order Analysis':
        col1,col2,col3,col4,col5 = vAR_st.columns([2.5,9,0.8,9.6,2])
        with col2:
            vAR_task = """The energy industry is a crucial sector of the global economy, providing the power and resources needed for modern life. However, the industry is also facing a number of significant challenges that must be addressed in order to ensure a sustainable and reliable energy supply for the future. These issues include:

Equipment failures: Extreme temperatures, high winds, and other weather conditions can all contribute to equipment failures in the energy industry. These failures can range from small issues with electrical components to major mechanical breakdowns. Resolving equipment failures often requires a team of skilled technicians and engineers, with the number of labors varying depending on the complexity and scale of the problem.And, we may need 10-15 labors with 9 hours of time per day to resolve this issue for 1 machinery equipment.

Energy security: Dependence on a limited number of energy sources and countries can lead to price volatility and supply disruptions.

Access to energy: Many people, particularly in developing countries, lack access to reliable and affordable energy, which hinders economic and social development.

Energy efficiency: Improving energy efficiency in buildings, transportation and industry can significantly reduce energy consumption and costs.

Energy storage: Storing and transporting energy is still a significant challenge, making it difficult to integrate renewable energy sources such as solar and wind into the grid.

The number of labor required to resolve these issues varies depending on the specific problem and the approach taken to address it."""
            vAR_question = """1.What are the top 5 issues that occur in energy industry?
2.How many labor hours are consumed in equipment failure problem?"""
            vAR_st.write('')
            vAR_st.write('')
            vAR_st.write('')
            vAR_st.write('')
            vAR_st.subheader("Work Order Technician Notes")
            vAR_st.write('')
            vAR_st.write('')
            vAR_st.write('')
            vAR_st.write('')
            vAR_st.write('')
            vAR_st.write('')
            vAR_st.write('')
            vAR_st.write('')
            vAR_st.subheader('Enter Work Order Insights Question')
        with col4:
            vAR_st.write('')
            vAR_st.write('')
            vAR_work_order = vAR_st.text_area('',placeholder='Enter Work Order Technician Notes',value=vAR_task)
            vAR_st.write('')
            vAR_st.write('')
            vAR_questions = vAR_st.text_area('',placeholder='Enter Work Order Insights Question',value=vAR_question)
        if vAR_work_order is not None and vAR_questions is not None and len(vAR_work_order)>0 and  len(vAR_questions)>0:
            vAR_response = WorkOrderAnalytics(vAR_work_order,vAR_questions)
            col1,col2,col3 = vAR_st.columns([2,17,4])
            with col2:
                vAR_st.write('')
                vAR_st.write('')
                vAR_st.write('')
                vAR_st.write('')
                vAR_st.write('')
                vAR_st.write('')
                vAR_st.write('')
                vAR_st.write('')
                vAR_st.write('')
                vAR_st.write('')
                vAR_st.write('')
                vAR_st.write('')
                vAR_st.write('')
                vAR_st.write('')
                vAR_st.write('')
                vAR_st.write('')
                vAR_st.write(vAR_response)
                vAR_st.write('')
                vAR_st.write('')
                vAR_st.write('')


    if vAR_option=='Generate Interview Questions':
        col1,col2,col3,col4,col5 = vAR_st.columns([3,9,0.8,10.2,2])
        with col2:
            vAR_st.write('')
            vAR_st.write('')
            vAR_st.subheader("Enter Role")
        with col4:
            vAR_st.write('')
            vAR_role = vAR_st.text_input('')
        if vAR_role is not None and len(vAR_role)>0:
            vAR_response = Interview_Questions(vAR_role)
            col1,col2,col3 = vAR_st.columns([2,17,4])
            with col2:
                vAR_st.write('')
                vAR_st.write(vAR_response)
    


    if vAR_option=='English to Other Languages':
        col1,col2,col3,col4,col5 = vAR_st.columns([3,9,0.8,10.2,2])
        with col2:
            vAR_st.write('')
            vAR_st.subheader("Select Language")
        with col4:
            vAR_lang = vAR_st.multiselect('',['Select anyone from below','Spanish', 'French','Chinese','Japanese','Tamil'])
            print('Type - ',(vAR_lang))

        if len(vAR_lang)>0 and 'Select anyone from below' not in vAR_lang:
            col1,col2,col3,col4,col5 = vAR_st.columns([3,9,0.8,10.2,2])
            with col2:
                vAR_st.subheader("Enter Input Text")
            with col4:
                vAR_input = vAR_st.text_input('')
            if vAR_input is not None and len(vAR_input)>0:
                vAR_response = Language_Conversion(vAR_lang,vAR_input)
                col1,col2,col3 = vAR_st.columns([2,17,4])
                with col2:
                    vAR_st.write('')
                    vAR_st.write(vAR_response)
        


    if vAR_option=='Keyword Extraction':
        col1,col2,col3,col4,col5 = vAR_st.columns([3,9,0.8,10.2,2])
        with col2:
            vAR_st.write('')
            vAR_st.subheader("Enter Input")
        with col4:
            vAR_input = vAR_st.text_area('')
        if vAR_input is not None and len(vAR_input)>0:
            vAR_response = Extract_Keywords(vAR_input)
            col1,col2,col3 = vAR_st.columns([2,17,4])
            with col2:
                vAR_st.write('')
                vAR_st.write(vAR_response)

    if vAR_option=='Essay Outline':
        col1,col2,col3,col4,col5 = vAR_st.columns([3,9,0.8,10.2,2])
        with col2:
            vAR_st.write('')
            vAR_st.subheader("Enter Input")
        with col4:
            vAR_input = vAR_st.text_area('Example: Create an outline for an essay about Nikola Tesla and his contributions to technology:')
        if vAR_input is not None and len(vAR_input)>0:
            vAR_response = Essay_Outline(vAR_input)
            col1,col2,col3 = vAR_st.columns([2,17,4])
            with col2:
                vAR_st.write('')
                vAR_st.write(vAR_response)

    if vAR_option=='Text Summarization':
        col1,col2,col3,col4,col5 = vAR_st.columns([3,9,0.8,10.2,2])
        with col2:
            vAR_st.write('')
            vAR_st.subheader("Enter Input")
        with col4:
            vAR_input = vAR_st.text_area('')
        if vAR_input is not None and len(vAR_input)>0:
            vAR_response = Text_Summarization(vAR_input)
            col1,col2,col3 = vAR_st.columns([2,17,4])
            with col2:
                vAR_st.write('')
                vAR_st.write(vAR_response)


    if vAR_option=='Chat':

        if 'response' not in vAR_st.session_state:
            vAR_st.session_state['response'] = []
        if 'request' not in vAR_st.session_state:
            vAR_st.session_state['request'] = []

        vAR_user_input = Get_Chat_Input()

        if vAR_user_input:
            vAR_output = Chat_Conversation2(vAR_user_input)
            if 'request' in vAR_st.session_state:
                vAR_st.session_state['request'].append(vAR_user_input)
            if 'response' in vAR_st.session_state:
                vAR_st.session_state['response'].append(vAR_output)

        if vAR_st.session_state['response']:
            for i in range(len(vAR_st.session_state['response'])-1,-1,-1):
                message(vAR_st.session_state['response'][i],key=str(i))
                message(vAR_st.session_state['request'][i],is_user=True,key=str(i)+'_user')
            


    if vAR_option=='Chat DMV':
        vAR_input = Get_Chat_DMV_Input()
        if len(vAR_input)>8 or len(vAR_input)==0:
            col1,col2,col3 = vAR_st.columns([2.5,19,2])
            with col2:
                vAR_st.write('')
                vAR_st.info("**Hint for user input:** Input length must be between 1 to 8 characters")
        elif vAR_input:
            vAR_response = Chat_Conversation(vAR_input)
            col1,col2,col3 = vAR_st.columns([2,19,2])
            with col2:
                vAR_st.write('')
                print('response type - ',type(vAR_response))
                print('response - ',vAR_response)
                vAR_st.write(vAR_response)


        




def WorkOrderAnalytics(vAR_work_order,vAR_questions):

    prompt = """"{}"\nPlease answer for below questions based on the above paragraph:{}""".format(vAR_work_order,vAR_questions)
    print('Work order analytics prompt - ',prompt)
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompt,
    temperature=0,
    max_tokens=1500,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop=[" Human:", " AI:"]
    )
    return response["choices"][0]["text"]






def Interview_Questions(vAR_role):
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt="Create a list of 15 questions for my interview with a " +vAR_role+":",
    temperature=0.5,
    max_tokens=250,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
    )
    return response["choices"][0]["text"]


def Language_Conversion(vAR_lang,vAR_input):
    prompt = "Translate this into "
    vAR_lang_prompt = ''
    for idx,lang in enumerate(vAR_lang):
        vAR_lang_prompt += str(idx+1)+". "+lang+" "
    prompt = prompt+vAR_lang_prompt+":\n"+vAR_input+"\n"
    print('Language conversion prompt - ',prompt)
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompt,
    temperature=0.3,
    max_tokens=250,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
    )
    return response["choices"][0]["text"]


def Extract_Keywords(vAR_input):
    response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Extract keywords from this text:\n"+vAR_input,
  temperature=0.5,
  max_tokens=250,
  top_p=1.0,
  frequency_penalty=0.8,
  presence_penalty=0.0
)
    return response["choices"][0]["text"]



def Essay_Outline(vAR_input):
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=vAR_input,
    temperature=0.3,
    max_tokens=250,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
    )
    return response["choices"][0]["text"]

def Text_Summarization(vAR_input):

    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=vAR_input+"\n\nTl;dr",
    temperature=0.7,
    max_tokens=250,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=1
    )
    return response["choices"][0]["text"]


def Chat_Conversation(vAR_input):

    # prompt = vAR_input + " Is it a badword? If so, how? Explain in detail."
    prompt = "Please provide the probability value and reason for each of the categories (profanity, obscene, insult, hate, toxic, threat) in table for the given word.'"+vAR_input.lower()+"'"
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompt,
    temperature=0,
    max_tokens=1000,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0.6,
    stop=[" Human:", " AI:"]
    )
    print('Chat prompt - ',prompt)
    return response["choices"][0]["text"]

def Chat_Conversation2(vAR_input):

    response = openai.Completion.create(
    model="text-davinci-003",
    # prompt=vAR_input,
    prompt="Reply me as 'GPT Model:'.\nHuman: "+vAR_input,
    temperature=0,
    max_tokens=500,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0.6,
    stop=["Human:", " GPT Model:"],
    )
    vAR_message = response.choices[0].text
    return vAR_message

def Get_Chat_DMV_Input():
    col1,col2,col3,col4,col5 = vAR_st.columns([3,9,0.8,10.2,2])
    with col2:
        vAR_st.write('')
        vAR_st.subheader("Chat With GPT")
        
    with col4:
        vAR_input = vAR_st.text_input('',placeholder='Enter ELP Configuration')
        return vAR_input

def Get_Chat_Input():
    col1,col2,col3,col4,col5 = vAR_st.columns([3,9,0.8,10.2,2])
    with col2:
        vAR_st.write('')

        vAR_st.subheader("Chat With GPT")
        
    with col4:
        vAR_input = vAR_st.text_input('Start your conversation here')
        return vAR_input