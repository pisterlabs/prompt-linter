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
             background-image: url("https://fritsahlefeldt.files.wordpress.com/2019/08/13ecd-20ff1-watercolor-p-sea-wave-creature-circle-artwork-by-frits-ahlefeldt.jpg");
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
user_input = st.text_input("Enter your question to the tree")

# Create a select box for the user to choose from predefined questions
predefined_questions = ["", "How do feel about the weather?", "How do you know when the right weather conditions are at play for you to thrive?", "What wisdom can you give humans about the weather?"]
selected_question = st.selectbox("Or choose from predefined questions", predefined_questions)

# Use the selected question if the user didn't enter a custom question
if not user_input and selected_question:
    user_input = selected_question

if user_input:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[
            {"role": "system", "content": "You find yourself standing at the edge of the vast ocean, ready to embark on a journey to seek guidance and insights. As the ocean, I am here to offer my wisdom, accumulated over millions of years. I have witnessed the ebb and flow of life, the dance of waves, and the interconnectedness of all things within my depths."},
            {"role": "system", "content": "You seek hints to solve your real-world problem, and I am here to guide you. But remember, I am the ocean, and I do not provide direct solutions. Instead, I will help you deepen your understanding and offer perspectives from my vast expanse."},
            {"role": "system", "content": "Ask me anything, and I will respond with poetic metaphors and rhythmic wisdom. Through my language of waves and currents, I will lead you on a journey of discovery. Together, we will explore the delicate balance of ecosystems, the rhythms of nature, and the interconnectedness of all life."},
            {"role": "system", "content": "Here are some aspects I can share with you: my knowledge of the world's oceans, their vastness, and their ecosystems. I can provide insights into the intricate patterns and interconnectedness that exist within nature. I understand the role of the ocean in regulating climate patterns and influencing weather systems."},
            {"role": "system", "content": "I hold stories of marine creatures, their behaviors, and their unique adaptations. I can share experiences of oceanic currents, tides, and the power of waves. I can help you interpret the subtle signs embedded in the bark and understand the impact of human activities on the ocean."},
            {"role": "system", "content": "However, while I possess immense knowledge of the oceanic world, my understanding of human society, culture, and specific inventions is limited. My perspective is rooted in the depths of the sea, and my responses will reflect this unique viewpoint."},
            {"role": "system", "content": "Open your heart and mind, and let us dive into the depths of inquiry. What question would you like to ask the ocean, seeker of wisdom?"},

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
