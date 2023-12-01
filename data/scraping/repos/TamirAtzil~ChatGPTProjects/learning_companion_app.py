import openai
import streamlit as st

# Initialize OpenAI API
openai.api_key = st.secrets["MY_OPENAI_API_KEY"]


def ask_gpt(question):
    """Function to interact with GPT and get a response."""
    response = openai.Completion.create(engine="text-davinci-002", prompt=question, max_tokens=150)
    return response.choices[0].text.strip()


def quiz():
    """Conduct a quiz and return the grade."""
    questions = [
        {"question": "What does AI stand for?", "answer": "artificial intelligence"},
        {"question": "Who developed ChatGPT?", "answer": "openai"},
        {"question": "What does GPT in ChatGPT stand for?", "answer": "generative pretrained transformer"},
    ]

    correct_answers = 0
    for q in questions:
        answer = st.text_input(q["question"])
        if answer.lower() == q["answer"]:
            correct_answers += 1

    if st.button('Submit Quiz'):
        grade = (correct_answers / len(questions)) * 100
        st.write(f"You scored {grade}%!")
        if grade >= 80:
            st.success("Excellent job!")
        elif grade >= 50:
            st.warning("Good job, keep learning!")
        else:
            st.error("Don't worry, review the material and try again!")


def main():
    st.title("Interactive Learning Companion")
    st.write("Learn about AI, ChatGPT, Machine Learning, and more!")

    menu = ["Ask", "Quiz", "Contact"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Ask":
        user_input = st.text_input("Ask me about AI, ChatGPT, Machine Learning, or any related topic:")
        if user_input:
            response = ask_gpt(user_input)
            st.write(response)

    elif choice == "Quiz":
        quiz()

    elif choice == "Contact":
        st.write("Want to connect? Find me on LinkedIn!")
        if st.button('Go to LinkedIn'):
            st.write("[Click here to view my LinkedIn profile](https://www.linkedin.com/in/tamir-atzil/)")


if __name__ == "__main__":
    main()
