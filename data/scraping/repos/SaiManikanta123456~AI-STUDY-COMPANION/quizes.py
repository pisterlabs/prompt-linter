import streamlit as st
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Set your OpenAI API key here
openai.api_key = "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"

def generate_questions(subject):
    # Customize the prompt based on the subject
    prompt = f"it is actually a tool for testing students knowledge in particular subject.so, you need to ask questions fron basic to advanced level to test student knowledge : {subject}."

    # Use OpenAI to generate questions
    response = openai.Completion.create(
        engine="davinci",  # You can experiment with different engines
        prompt=prompt,
        max_tokens=200  # Adjust based on the length of questions you want
    )

    questions = response.choices[0].text.strip().split('\n')

    return questions

# Function to calculate similarity score
def calculate_similarity(answer, question):
    # Convert the answer and question to lowercase
    answer = answer.lower()
    question = question.lower()

    # Calculate similarity using cosine similarity
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([answer, question])
    similarity = cosine_similarity(vectors[0].reshape(1, -1), vectors[1].reshape(1, -1))

    return similarity[0][0]

# Streamlit app starts here
st.title("Flashcard Generator")

# Get subject input from user
subject = st.text_input("Enter the subject:")

if subject:
    st.write(f"Generating flashcards for: {subject}")

    # Generate questions
    questions = generate_questions(subject)

    # Display flashcards and evaluate answers
    for i, question in enumerate(questions, start=1):
        st.write(f"**Flashcard {i}**")
        st.write("Question:", question)

        # Get user's answer
        answer = st.text_input("Your Answer:")

        # Evaluate the answer and provide a score
        if answer:
            similarity_score = calculate_similarity(answer, question)
            st.write(f"Similarity Score: {similarity_score:.2f}")
