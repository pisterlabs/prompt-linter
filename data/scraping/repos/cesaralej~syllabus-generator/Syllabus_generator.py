# import required packages
import openai
import streamlit as st
from streamlit_option_menu import option_menu


# Start by creating a venv:
# python -m venv myenv

# Activate your venv:
# source venv_name/bin/activate   (mac)
# venv_name\Scripts\activate  (windows)

# Install the required packages:
# pip install -r requirements.txt

# Run the code in the terminal:
# streamlit run Syllabus_generator.py


# Read the original syllabus
def read_original_syllabus(file_path="original_syllabus.txt"):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            original_syllabus = file.read()
        return original_syllabus
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None


original_syllabus = read_original_syllabus()


# API Request to generate the syllabus
def syllabus_request():
    messages = [
        {
            "role": "system",
            "content": f"You are a teacher for the class BIG DATA & ARTIFICIAL INTELLIGENCE IN BUSINESS STRATEGY. Your class follow the framework of this Syllabus:\n\n{original_syllabus}",
        },
        {
            "role": "user",
            "content": f"""Customize the first 5 sessions of the syllabus based on the syllabus framework for the 'BIG DATA & ARTIFICIAL INTELLIGENCE IN BUSINESS STRATEGY' class for a student with 
         {st.session_state.student_exp_years} years of professional experience, with a 
         {st.session_state.student_background} role background that wants to move to a 
         {st.session_state.student_future} role in the 
         {st.session_state.student_industry} industry. Your reply should only have the updated 5 sessions of the syllabus written in the same structure as the original one""",
        },
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages, temperature=0.3, max_tokens=2048
    )
    return response["choices"][0]["message"]["content"]


# API Request to generate the capstone project
def capstone_request():
    messages = [
        {
            "role": "system",
            "content": f"You are a teacher for the class BIG DATA & ARTIFICIAL INTELLIGENCE IN BUSINESS STRATEGY. Your class follow the framework of this Syllabus:\n\n{original_syllabus}",
        },
        {
            "role": "user",
            "content": f"""Design a case study project for the 'BIG DATA & ARTIFICIAL INTELLIGENCE IN BUSINESS STRATEGY' class for a student with 
         {st.session_state.student_exp_years} years of professional experience, with a 
         {st.session_state.student_background} role background that wants to move to a 
         {st.session_state.student_future} role in the 
         {st.session_state.student_industry} industry. Your reply should only have the project instructions. The project should present a case where a fictional company of the industry is facing a challenge and the student needs to identify a solution based on the subjects learned on the syllabus""",
        },
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages, temperature=0.3, max_tokens=2048
    )
    return response["choices"][0]["message"]["content"]


st.set_page_config(
    page_title="Business Strategy Syllabus",
    page_icon="🌐",
    initial_sidebar_state="expanded",
)

# Sidebar
with st.sidebar:
    # Set up OpenAI API key
    st.header("OpenAI API Configuration")
    st.write("To personalize your syllabus, configure your OpenAI API settings below.")
    st.write(
        "Don't have an API key? Visit [OpenAI](https://beta.openai.com/signup/) to get one."
    )
    api_key = st.sidebar.text_input("Enter your OpenAI API key")
    # Validation check for API key
    if st.button("Submit"):
        if not api_key:
            st.error("Please enter your OpenAI API key.")
        else:
            openai.api_key = api_key
            st.success("API key set successfully!")

    # User Information
    st.sidebar.header("Input Your Information")

    st.session_state.student_exp_years = st.sidebar.text_input(
        "Years of Professional Experience",
        help="Enter the number of years you have been working professionally.",
        value="5",
    )

    professional_background_options = ["Business", "Tech", "Hybrid"]
    st.session_state.student_background = st.sidebar.selectbox(
        "Professional Background",
        professional_background_options,
        help="Specify your professional background, e.g., Business, Tech, or Hybrid.",
    )
    if st.session_state.student_background == "Tech":
        st.sidebar.text_input(
            "Tech Skills", help="List your relevant technical skills."
        )

    st.session_state.student_future = st.sidebar.selectbox(
        "Future Career Goal",
        professional_background_options,
        help="Describe the role you aim to achieve, e.g., Business, Tech, or Hybrid.",
    )
    st.session_state.student_industry = st.sidebar.text_input(
        "Target Industry",
        help="Enter the industry in which you aspire to work.",
        value="Consulting",
    )
    # Validations
    if (
        st.session_state.student_exp_years
        and not st.session_state.student_exp_years.isdigit()
    ):
        st.error("Please enter a valid number for years of experience.")

    def generate_syllabus():
        try:
            with st.spinner("Generating Syllabus..."):
                if (
                    st.session_state.student_exp_years
                    and st.session_state.student_background
                    and st.session_state.student_future
                    and st.session_state.student_industry
                ):
                    st.session_state.syllabus_content = syllabus_request()
                    st.success("Syllabus generated successfully!")
        except Exception as e:
            st.error(f"Error generating syllabus: {e}")

    # Submit button
    if (
        not st.session_state.student_exp_years
        or not st.session_state.student_background
        or not st.session_state.student_future
        or not st.session_state.student_industry
    ):
        st.warning(
            "Please complete all required fields before generating the syllabus."
        )
        st.button("Generate Syllabus", disabled=True)
    else:
        if st.button("Generate Syllabus"):
            generate_syllabus()
    st.image("IE_Business_School_logo.svg.png", width=100)

# Title
st.markdown(
    f"<h1 style='font-size: 36px; text-align: center;'>BIG DATA & AI IN BUSINESS STRATEGY</h1>",
    unsafe_allow_html=True,
)

# Introductory Message
st.markdown(
    f"<p style='font-size: 20px; text-align: center;'>Welcome to Your AI-Driven Learning Experience!</p>",
    unsafe_allow_html=True,
)

# Instructions on how to use the app
st.markdown(
    f"<h2 style='font-size: 28px;'>How to Use:</h2>",
    unsafe_allow_html=True,
)
st.write("1. **Configure your OpenAI API key in the sidebar.**")
st.write("2. **Input your professional information on the left.**")
st.write(
    "3. **Click on 'Generate Syllabus' to receive your personalized learning plan.**"
)

type = option_menu(
    None,
    ["Syllabus", "Capstone Project"],
    icons=[],
    default_index=0,
    orientation="horizontal",
)

# Syllabus section
if type == "Syllabus":
    st.subheader("Personalized Syllabus Generator")
    st.markdown("---")
    if "syllabus_content" not in st.session_state:
        st.subheader("No syllabus generated yet")
        st.write(
            "Your personalized syllabus is crafted based on the information you provide."
        )
        st.write("Unlock a unique learning journey with AI-driven customization.")
    else:
        st.markdown(
            f"**Your Personalized Syllabus:**\n\n{st.session_state.syllabus_content}"
        )

# Capstone Project section
if type == "Capstone Project":
    st.subheader("Capstone Project Generator")
    st.write(
        "Once your project is ready, submit to the corresponding learning platform"
    )

    # Call a function to generate and display the dynamic content
    def generate_capstone():
        try:
            with st.spinner("Generating project instructions..."):
                if (
                    st.session_state.student_exp_years
                    and st.session_state.student_background
                    and st.session_state.student_future
                    and st.session_state.student_industry
                ):
                    st.session_state.project_content = capstone_request()
                    st.success("Instructions generated successfully!")
        except Exception as e:
            st.error(f"Error generating project instructions: {e}")

    # Submit button
    if (
        not st.session_state.student_exp_years
        or not st.session_state.student_background
        or not st.session_state.student_future
        or not st.session_state.student_industry
    ):
        st.warning(
            "Please complete all required fields before generating the project instructions."
        )
        st.button("Generate Capstone Project", disabled=True)
    else:
        if st.button("Generate Project Instructions"):
            generate_capstone()
    st.markdown("---")
    if "project_content" not in st.session_state:
        st.subheader("No project instructions generated yet")
        st.write(
            "This is where the dynamic capstone project content will be displayed."
        )
    else:
        st.markdown(
            f"**Your Personalized Project Instructions:**\n\n{st.session_state.project_content}"
        )
