from utils import (run_agent_from_profile, build_report,
                   extract_entities_keywords, generate_goals,
                   build_llm, build_llm_tools, suggest_activities,
                   memory_to_pandas)
import streamlit as st
import json
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import time

# TODO: Test processing sample user profiles in memory with agents
# TODO: Test digital nudging for simulated scenario
# TODO: Write better prompts/descriptions (template for PromptTemplate, input_variables come from user profile or agent input or user input) for AI profiles in memory
# TODO: Code demo (a bit better)

# load memory globally
memory_path = "test_long_term_memory.json"
memory_df = memory_to_pandas(memory_path)

# load sample PERMA4 globally
with open("test_perma4.json") as f:
    perma4 = json.load(f)

# Agents
# 1. Well-being coach
coach = memory_df['AI_profiles'][0]

# 2. Journalist
journalist = memory_df['AI_profiles'][1]

# 3. Recommendationg Engine
recommendator = memory_df["AI_profiles"][2]

# 4. Digital Nudger
nudger = memory_df["AI_profiles"][3]

# 5. Report Generator
report_gen = memory_df["AI_profiles"][4]


def run_demo():
    st.title("Atlas Intelligence Demo")

    # user information we want to demo (used only for tuning goal recommendation not report)
    st.sidebar.header("User Profile")
    name = st.sidebar.text_input("Name", "Augusto")
    age = st.sidebar.text_input("Age", "22")
    tastes = st.sidebar.text_input("Tastes", "tennis, sports, physics, reading, economics")
    occupation = st.sidebar.text_input("Occupation", "graduate student")
    location = st.sidebar.text_input("Location", "claremont, CA")
    
    if name and age and tastes and occupation and location:
        # list of inputs to string
        user_data = str([name, age, tastes, occupation, location])
        if st.button("Get Report, Goals, and Activities"):
            
            st.spinner("Generating report")
            # build report with report agent and demo perma4 results [working function]
            report = build_report(report_generator_profile=report_gen,
                                    perma_results=perma4)
            st.write("---- DEMO REPORT ----")
            st.write(report) # around 12 seconds to display
            
            st.spinner("Generating goals")
            time.sleep(2)
        
            # generate goals using report and user profile
            goals = generate_goals(recommender_generator_profile=recommendator,
                            user_data=user_data,
                            report=report)
            st.write("---- DEMO GOALS ----")
            st.write(goals)
            
            st.spinner("Generating activities")
            time.sleep(2)
            
            # suggest activities using goals and user profile
            activities = suggest_activities(coach_profile=coach,
                                            user_data=user_data,
                                            goals=goals)
            st.write("---- DEMO ACTIVITIES ----")
            st.write(activities)
            
    else:
        st.warning("Please fill in the user profile information on the left sidebar before generating goals.")
        
    
if __name__ == "__main__":
    run_demo()
