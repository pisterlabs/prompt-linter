import openai
import streamlit as st

# Set up OpenAI API key
from dotenv.main import load_dotenv
import os
load_dotenv()
openai.api_key = os.environ['OPENAI_KEY']

# Define function to generate description using OpenAI GPT-3 model
def generate_description(company_name, year, company_type, specialty, location, employees, revenue, mission_statement):
    """
    Generate a description of the company based on user input fields using OpenAI GPT-3 model.
    """
    prompt = f"Company Name: {company_name}\nCompany Type: {company_type}\nEstablished In: {year}\nSpeciality: {specialty}"
    if location:
        prompt += f"\nLocation: {location}"
    if employees:
        prompt += f"\nNumber of Employees: {employees}"
    if revenue:
        prompt += f"\nAnnual Revenue: {revenue}"
    if mission_statement:
        prompt += f"\nMission Statement: {mission_statement}"
    prompt += "\n\nAutogenerated Description:"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=0.5,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].text.strip()

# Define Streamlit app
def app():
    # Set page title
    st.set_page_config(page_title="Company Description Generator")

    # Set app title
    st.title("Company Description Generator")

    # Set up user input fields
    company_name = st.text_input("Enter company name")
    year = st.text_input("Enter establishment year")
    company_type = st.text_input("Enter company type")
    specialty = st.text_input("Enter specialty of company")
    location = st.text_input("Enter location of company (optional)")
    employees = st.number_input("Enter number of employees (optional)", value=0)
    revenue = st.number_input("Enter annual revenue (optional)", value=0)
    mission_statement = st.text_input("Enter mission statement (optional)")

    # Generate description based on user input fields
    if st.button("Generate"):
        # Validate required fields
        if not company_name or not specialty:
            st.warning("Please enter company name and specialty to generate description.")
        else:
            # Generate description based on user input fields
            description = generate_description(company_name, year, company_type, specialty, location, employees, revenue, mission_statement)
            # Display generated description
            st.write(description)
            # Display summary of generated text
            summary = f"Summary:\n- {company_name} is a {company_type} company that specializes in {specialty}."
            if location:
                summary += f"\n- Based in {location}."
            if employees:
                summary += f"\n- Employs {employees} people."
            if revenue:
                summary += f"\n- Generates {revenue} in annual revenue."
            if mission_statement:
                summary += f"\n- Mission statement: {mission_statement}"
            st.write(summary)

    else:
        st.write("Please enter company name and speciality to generate description.")


app()