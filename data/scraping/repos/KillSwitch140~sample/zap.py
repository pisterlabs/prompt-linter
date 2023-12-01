# import streamlit as st
# from langchain.agents import AgentType, initialize_agent
# from langchain.agents.agent_toolkits import ZapierToolkit
# from langchain.llms import OpenAI
# from langchain.utilities.zapier import ZapierNLAWrapper
# import os
# from os import environ
# import time


# zapier_nla_api_key = st.secrets["ZAP_API_KEY"]

# openai_api_key = st.secrets["OPENAI_API_KEY"]

# llm = OpenAI(temperature=0)
# zapier = ZapierNLAWrapper()
# toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
# agent = agent = initialize_agent(
#     toolkit.get_tools(), llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
# )

# def schedule_interview(person_name, person_email, date, time):
#     # Create the combined string
#     meeting_title = f"Hiring Plug Interview with {person_name}"
#     date_time = f"{date} at {time}"
#     schedule_meet = f"Schedule a 30 min meeting titled {meeting_title} on {date_time}. Quick add the created meeting's details as a new event in my calendar"
#     send_email = (
#         f"Send email to {person_email}"
#     )

#     # Execute the agent.run function for scheduling the meeting
#     agent.run(schedule_meet)
#     agent.run(send_email)
#     return True  # Return True if the interview is scheduled and the email is sent successfully
