import streamlit as st
import openai
import pandas as pd

st.set_page_config(layout="centered")

# Set up your OpenAI API key
OPENAI_API_KEY = "YOUR_API_KEY"
openai.api_key = OPENAI_API_KEY

st.title("Chatting with Hacky")
st.write("Fill in the prompts below to chat with Hacky. Hacky will respond to your prompts with some ideas for your hackathon.")

# Add some space
st.empty().markdown("&nbsp;")

def generate_project_ideas():
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    col5, col6 = st.columns(2)

    with col1:
        category_header_empty = st.empty()
        category_choice_empty = st.empty()

    with col2:
        teammates_header_empty = st.empty()
        teammates_choice_empty = st.empty()

    with col3:
        duration_header_empty = st.empty()
        duration_choice_empty = st.empty()

    with col6:
        tech_stack_header_empty = st.empty()
        tech_stack_choice_empty = st.empty()

    with col5:
        theme_header_empty = st.empty()
        theme_choice_empty = st.empty()

    with col4:
        complexity_header_empty = st.empty()
        complexity_choice_empty = st.empty()
        
    category_header_empty.markdown("Category")
    category = category_choice_empty.selectbox(
        "Select a category", 
        ["Web", "Mobile", "Machine Learning", "Data Science", "Hardware", "Other"]
    )

    teammates_header_empty.markdown("Teammates")
    teammates = teammates_choice_empty.selectbox("Select the number of teammates", [1, 2, 3, 4, 5])

    duration_header_empty.markdown("Duration")
    duration = duration_choice_empty.selectbox("Select the duration", ["24 hours", "36 hours", "48 hours", "72 hours", "720 hours"])

    tech_stack_header_empty.markdown("Tech Stack")
    tech_stack = tech_stack_choice_empty.text_input("Enter the tech stack: (Optional)")

    theme_header_empty.markdown("Theme")
    theme= theme_choice_empty.selectbox(
        "Select a theme", 
        ["Healthcare","Sustainability", "Productivity","Sports", "Commercial", "Other"]
    )
    complexity_header_empty.markdown("Complexity")
    complexity= complexity_choice_empty.selectbox(
        "Choose complexity level" ,
        ["low","medium","high"]
    )
    # Prompt for GPT-3 API
    prompt = f"A team is participating in a hackathon, and they need a brilliant project idea for a {category} application to impress the judges. The hackathon will last  {duration}  hours, and have {teammates} teammates and complexity should be low. The theme of the hackathon is {theme}.Brainstorming time! Generate 5 innovative project ideas with a project name and a small description that perfectly fits the criteria. Think about the unique challenges we can address and the potential impact project could have in the {theme} domain. Remember to consider the time constraints and make sure the idea is achievable within the given {duration} hours."
    if st.button("Generate Ideas"):
        # Use GPT-3 API to get the response
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=400,
            stop=None,  # Custom stop sequence to end the response if needed
        )


        # Check if the response contains any ideas
        if "choices" not in response or len(response["choices"]) == 0:
            st.write("No project ideas generated for the specified criteria.")
        else:
            # Parse the response to create a DataFrame
            data = response["choices"][0]["text"].split("\n\n")
            print(data)
            titles = [line.split(': ')[0].split('. ')[1] for line in data[1:] if line.strip()]
            descriptions = [line.split(': ')[1] for line in data[1:] if line.strip()]

            # Create a DataFrame
            st.dataframe({'Title': titles, 'Description': descriptions})

            # Create the Streamlit DataFrame

            # Display the DataFrame
            

if __name__ == "__main__":
    generate_project_ideas()
