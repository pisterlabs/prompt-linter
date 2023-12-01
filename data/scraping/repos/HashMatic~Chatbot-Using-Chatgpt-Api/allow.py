import streamlit as st
import openai

# Set up OpenAI API client
api_key = st.text_input("Please enter your OpenAI API key:")
openai.api_key = api_key

# Set the engine to use (GPT-3.5 Turbo)
engine = "gpt-3.5-turbo"

# Main program loop
while True:
    # Prompt user for question
    user_input = st.text_input("Ask a question (or enter 'q' to quit):")

    # Check if user wants to quit
    if user_input.lower() == 'q':
        break

    # Call OpenAI API to generate a response
    response = openai.Completion.create(
        engine=engine,
        prompt=user_input,
        max_tokens=100,
        temperature=0.7,
        n=1,
        stop=None,
        top_p=1.0
    )

    # Extract the generated answer from the response
    answer = response.choices[0].text.strip()

    # Display the answer
    st.text("Answer: " + answer)
