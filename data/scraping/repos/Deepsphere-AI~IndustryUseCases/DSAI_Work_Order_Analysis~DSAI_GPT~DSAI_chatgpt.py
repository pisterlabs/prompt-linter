from pyChatGPT import ChatGPT
import streamlit as vAR_st
import openai
import os

openai.api_key = os.environ["API_KEY"]

def WorkOrder_ChatGPT_Response():
    col1,col2,col3,col4,col5 = vAR_st.columns([2.5,9,0.8,9.8,2])
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



def WorkOrderAnalytics(vAR_work_order,vAR_questions):

    prompt = """"{}"\nPlease answer for below questions based on the above paragraph:{}""".format(vAR_work_order,vAR_questions)
    print('Work order analytics prompt - ',prompt)
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0301",
    messages=[{"role": "user", "content":prompt}],
    temperature=0,
    max_tokens=1500,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    )
    return response['choices'][0]['message']['content']