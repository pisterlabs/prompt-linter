# These are the import statements. You're importing several modules that your script will use.
import os
import streamlit as st
from dotenv import load_dotenv
import cohere
from cohere.responses.classify import Example
import pprint
from halo import Halo
from colorama import Fore, Style, init

# Inject CSS to style the Streamlit app
st.markdown(
    """
    <style>
        .reportview-container {
            background-color: #2C2C2C;
        }
        .markdown-text-container {
            color: #FFFFFF;
        }
        .stTextInput>div>div>input {
            color: #4F4F4F;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


load_dotenv()  # This loads environment variables from a .env file, which is good for sensitive info like API keys

pp = pprint.PrettyPrinter(indent=4)  # PrettyPrinter makes dictionary output easier to read

def generate_response(messages):
    spinner = Halo(text='Loading...', spinner='dots')  # Creates a loading animation
    spinner.start()

    co = cohere.Client(os.getenv("COHERE_KEY"))  # Initializes the Cohere API client with your API key

    mood = get_mood_classification(messages, co)
    department = get_department_classification(messages, co)

    spinner.stop()  # Stops the loading animation after receiving the response

    mood_priority = {
        'Despair': 1,
        'Sorrowful': 2,
        'Frustrated': 3,
        'Anxious': 4,
        'Irritated': 5,
        'Neutral': 6,
        'Satisfied': 7,
        'Joyful': 8
    }

    # Prints the user's mood, its priority level, and the responsible department
    print(
        f"\n{Fore.CYAN}Question Received: {Fore.WHITE}{Style.BRIGHT}{messages}{Style.RESET_ALL}"
        f"\n{Fore.GREEN}Mood Detected: {Fore.YELLOW}{Style.BRIGHT}{mood}{Style.RESET_ALL}"
        f"\n{Fore.GREEN}Priority Level: {Fore.RED if mood_priority[mood] <= 2 else Fore.YELLOW if mood_priority[mood] <= 4 else Fore.CYAN}{Style.BRIGHT}{mood_priority[mood]}{Style.RESET_ALL}"
        f"\n{Fore.GREEN}Department to handle your request: {Fore.MAGENTA}{Style.BRIGHT}{department}{Style.RESET_ALL}"
    )

    return messages, mood, department
    
    
    
def get_department_classification(messages, co):
    department_examples = [
        Example("How do I recharge my energy beam?", "Equipment Maintenance"),
        Example("How can I manage the collateral damage from my powers?",
                "City Relations"),
        Example("Can you assist me with writing a speech for the mayor's ceremony?",
                "Public Relations"),
        Example("Is there a special way to activate my stealth mode?",
                "Equipment Maintenance"),
        Example("What are the current laws regarding secret identities?", "Legal"),
        Example("There's a massive tornado headed for the city, what's our plan?",
                "Emergency Response"),
        Example("I pulled a muscle while lifting a car, what should I do?", "Medical"),
        Example("My superhero suit is damaged, can you help me fix it?",
                "Equipment Maintenance"),
        Example(
            "I need training for underwater missions, who can help me?", "Training"),
        Example("What should I say to the press about my recent rescue operation?",
                "Public Relations"),
        Example("How do I handle paperwork for the arrested supervillain?", "Legal"),
        Example(
            "I'm feeling overwhelmed with this superhero life, what should I do?", "Mental Health"),
        Example("How do I track the invisible villain?", "Intelligence"),
        Example(
            "I've been exposed to a new kind of radiation, do you have any info about this?", "Medical"),
        Example("My communication device is not working, how do I fix it?",
                "Equipment Maintenance"),
        Example(
            "How can I improve my relations with the local police department?", "City Relations"),
        Example(
            "My speed isn't improving, can you help me figure out a new training plan?", "Training"),
        Example("What's our protocol for inter-dimensional threats?",
                "Emergency Response"),
        Example("A civilian saw me without my mask, what should I do?", "Legal"),
        Example("How do I maintain my gear to ensure it doesn't fail during missions?",
                "Equipment Maintenance"),
        Example(
            "I can't shake off the guilt after failing a mission, what should I do?", "Mental Health"),
        Example(
            "The villain seems to know my every move, do we have a mole?", "Intelligence"),
        Example(
            "I'm having nightmares about past battles, can someone help?", "Mental Health"),
        Example("How can we predict the villain's next move?", "Intelligence"),
        Example(
            "I'm struggling to balance my civilian life and superhero duties, any advice?", "Mental Health")
    ]

    department_response = co.classify(
        model='large',
        inputs=[messages],
        examples=department_examples
    )  # Sends the classification request to the Cohere model

    department = department_response.classifications[0].prediction  # Extracts the prediction from the response
    return department

def get_mood_classification(messages, co):
    mood_examples = [
        Example("How do I recharge my energy beam?", "Neutral"),
        Example(
            "How can I manage the collateral damage from my powers?", "Anxious"),
        Example(
            "Can you assist me with writing a speech for the mayor's ceremony?", "Joyful"),
        Example("Is there a special way to activate my stealth mode?", "Neutral"),
        Example("What are the current laws regarding secret identities?", "Neutral"),
        Example(
            "There's a massive tornado headed for the city, what's our plan?", "Anxious"),
        Example(
            "I pulled a muscle while lifting a car, what should I do?", "Sorrowful"),
        Example("My superhero suit is damaged, can you help me fix it?", "Frustrated"),
        Example(
            "I need training for underwater missions, who can help me?", "Satisfied"),
        Example(
            "What should I say to the press about my recent rescue operation?", "Joyful"),
        Example(
            "How do I handle paperwork for the arrested supervillain?", "Irritated"),
        Example(
            "I'm feeling overwhelmed with this superhero life, what should I do?", "Despair"),
        Example("How do I track the invisible villain?", "Frustrated"),
        Example(
            "I've been exposed to a new kind of radiation, do you have any info about this?", "Sorrowful"),
        Example(
            "My communication device is not working, how do I fix it?", "Irritated"),
        Example(
            "How can I improve my relations with the local police department?", "Satisfied"),
        Example(
            "My speed isn't improving, can you help me figure out a new training plan?", "Joyful"),
        Example("What's our protocol for inter-dimensional threats?", "Despair"),
        Example("A civilian saw me without my mask, what should I do?", "Anxious"),
        Example(
            "How do I maintain my gear to ensure it doesn't fail during missions?", "Neutral"),
        Example(
            "I can't shake off the guilt after failing a mission, what should I do?", "Sorrowful"),
        Example(
            "The villain seems to know my every move, do we have a mole?", "Frustrated"),
        Example(
            "I'm having nightmares about past battles, can someone help?", "Despair"),
        Example("How can we predict the villain's next move?", "Anxious"),
        Example(
            "I'm struggling to balance my civilian life and superhero duties, any advice?", "Satisfied")
    ]

    mood_response = co.classify(
        model='large',
        inputs=[messages],
        examples=mood_examples
    )  # Sends the classification request to the Cohere model

    mood = mood_response.classifications[0].prediction  # Extracts the prediction from the response
    return mood
    


def main():
    st.title("EmotionDesk")
    st.markdown("""This application classifies the mood and department based on the user's input message.
                After you submit your message, the bot will classify your mood and the department best suited to handle your query.""")
   

    # Initialize chat_history and new_message_flag in session state if they don't exist
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'new_message_flag' not in st.session_state:
        st.session_state.new_message_flag = False

    # Display existing chat history
    for name, msg in st.session_state.chat_history:
        if name == "You":
            st.markdown(f"<div style='text-align: right; color: blue;'>{name}: {msg}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align: left; color: green;'>{name}: {msg}</div>", unsafe_allow_html=True)

    # Create a placeholder for the input box
    user_input_placeholder = st.empty()

    # Create the input box within the placeholder
    user_input = user_input_placeholder.text_input("Enter your message:", "")

    if user_input and not st.session_state.new_message_flag:
        messages, mood, department = generate_response(user_input)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", f"Mood Detected: {mood}\nDepartment: {department}"))
        
        # Set the new_message_flag to True
        st.session_state.new_message_flag = True

        # Clear the input box by rerunning the script
        user_input_placeholder.text_input("Enter your message:", value="", key="new")

        # Refresh the page to update the chat history
        st.experimental_rerun()
    else:
        # Reset the new_message_flag to False
        st.session_state.new_message_flag = False

if __name__ == "__main__":
    main()




