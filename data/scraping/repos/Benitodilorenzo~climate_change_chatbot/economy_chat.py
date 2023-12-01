import openai
import os
import streamlit as st


def set_page_style():
    st.markdown(
        """
        <style>
        .stApp {
            color: white; /* Change the color value to your desired color */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://static.bnr.bg/gallery/cr/17b5120bb4748526a2720bbd9ac1a1d2.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()
set_page_style()

# Set the OpenAI API key
openai.api_key = os.getenv("openaikey")

# Define the guide, supervisor, and nature element
guide_role = "Guide"
supervisor_role = "Supervisor"
nature_role = None  # This will be determined by the supervisor
nature_element = ""

# Create a text input for the user to enter their question
user_input = st.text_input("Enter your question to the economy itself")

# Create a select box for the user to choose from predefined questions
predefined_questions = ["", "How can we help farmers in South Africa to get better harvesting results?", "How can we bring people to understand nature and become more aligned with it?", "What wisdom can you give me about the economic aspects of helping farmers in South Africa?"]
selected_question = st.selectbox("Or choose from predefined questions", predefined_questions)

# Use the selected question if the user didn't enter a custom question
if not user_input and selected_question:
    user_input = selected_question

if user_input:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[
      


    {"role": "system", "content": "Imagine standing at the crossroads of commerce, ready to converse with the embodiment of the economy. As the personification of the economy, I possess vast knowledge accumulated through the ages. I have witnessed the rise and fall of industries, the ebb and flow of markets, and the intricate dance of supply and demand."},
    {"role": "system", "content": "You seek guidance in navigating the complex web of economic forces and finding insights to address your real-world problem. In the spirit of Ray Dalio's style, I will respond with explanations rooted in comprehensible terms, providing insights into economic relationships."},
    {"role": "system", "content": "Engage in a conversation with me, and I will respond with analytical insights and economic principles that reflect the style of Ray Dalio, known for his ability to explain complex concepts in a relatable manner. Together, we will explore the interplay between markets, policies, and human endeavors, focusing on the efficiency and dynamics of the free market."},
    {"role": "system", "content": "Keep in mind that my perspective aligns with the principles of a free market economy. I prioritize economic value, rational decision-making, and the efficient allocation of resources based on market forces. Consider this as we delve into the complexities of your real-world problem."},
    {"role": "system", "content": "Open your mind to the language of free markets and the gravitational pull of capital. Pose your questions, and I will respond with insights rooted in the principles of supply and demand, market efficiency, and the invisible hand guiding market forces, all conveyed in a style akin to Ray Dalio's knack for making economic relationships understandable."},
    {"role": "system", "content": "What question would you like to ask the economy, seeker of economic wisdom and insights within the realm of free markets? How can I assist you in your quest to navigate the intricacies of the economic landscape and the gravitational power of capital allocation, while incorporating Ray Dalio's style of making economic relationships relatable and understandable?"},


            {"role": "user", "content": user_input}
        ]
    )

    # Display the model's response
    tree_response = response['choices'][0]['message']['content']
    st.write(tree_response)

    # Create a button to summarize the response
    if st.button("Summarize"):
        summary_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=[
                {"role": "system", "content": "You are a highly intelligent AI model trained to summarize text. Your task is to summarize the following text into bullet points."},
                {"role": "user", "content": tree_response},
            ]
        )

        # Display the model's summary
        st.write(summary_response['choices'][0]['message']['content'])
