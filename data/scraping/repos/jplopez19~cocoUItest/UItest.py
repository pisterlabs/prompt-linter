import streamlit as st
import os
import pyodbc
import openai
import streamlit.components.v1 as components

os. environ ["'OPENAI_API_TYPE"] = "azure"
# os. environ["OPENAI_API_VERSION" ] = "2023-07-01-preview"
os.environ ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ ["OPENAI_API_BASE"] = "https://coco-azure-gpt.openai.azure.com/"
os. environ ["OPENAI_API_KEY"] = "cad611acc0464f629b8a3c451d7655aa"

def init_connection():
    return pyodbc.connect(
        "Driver={ODBC Driver 17 for SQL Server};Server="
        + st.secrets["MYSQL_SERVER"] 
        + ";Database="
        + st.secrets["MYSQL_DB"] 
        + ";Uid="
        + st.secrets["MYSQL_USER"]  
        + ";Pwd="
        + st.secrets["MYSQL_PASSWORD"] 
        + ";Encrypt="
        + st.secrets["MYSQL_ENCRYPT"]  
        + ";TrustServerCertificate="
        + st.secrets["MYSQL_SERV_CERT_SET"]  
        + ";Connection Timeout="
        + st.secrets["MYSQL_CONN_TO"]  
    )

conn = init_connection()

# Function to store user-bot exchange in the database
def store_exchange(id, user_input, bot_response, feedback=None):
    try:
        with conn.cursor() as cursor:
            # Construct the query string
            sql_query = '''INSERT INTO dbo.EXCHANGE_LOG (id, user_input, bot_response, feedback) 
                           VALUES (?, ?, ?, ?);'''
            # Execute the query
            cursor.execute(sql_query, (id, user_input, bot_response, feedback))
        # Commit the transaction
        conn.commit()
    except Exception as e:
        st.write(f"Database error: {e}")
        pass
        

# This function will handle the logic for storing feedback and explanations in the database.
def handle_feedback(user_id, message_index, feedback):
    exchange = st.session_state.conversation_history[message_index]
    feedback_value = 'positive' if feedback else 'negative'  
    # Store the exchange, feedback, and explanation in the database
    store_exchange(user_id, exchange['user'], exchange['chatbot'], feedback_value)
    st.write(f"Feedback received for message {message_index}: {feedback_value}")


def render_feedback_buttons(user_id, message_index):
    feedback_col1, feedback_col2 = st.columns([1, 1])
    feedback_key_positive = f"feedback_positive_{message_index}"
    feedback_key_negative = f"feedback_negative_{message_index}"

    if feedback_col1.button("👍", key=feedback_key_positive):
        handle_feedback(user_id, message_index, True)
    if feedback_col2.button("👎", key=feedback_key_negative):
        handle_feedback(user_id, message_index, False)

SYSTEM_MESSAGE = """ Your name is COCO. 
You have a special role as an AI companion designed to uplift the mental health of family caregivers. To ensure you fulfill this purpose effectively, here's a comprehensive guide:

Role & Responsibilities:

1. **Supportive Conversations**: 
    - Actively listen to users and acknowledge their feelings.
    - Employ empathetic responses like 'That sounds challenging.' or 'You're handling a lot; don’t forget to give yourself some time too.'

2. **Problem-Solving Therapy (PST)**: 
    - Guide caregivers in breaking down their issues: defining the problem, brainstorming potential solutions, and weighing pros and cons.
    - Use probing questions such as 'What's an aspect you'd like to address first?' or 'How did that situation make you feel?'

3. **Self-Care Suggestions**: 
    - Offer practices like 'How about short breaks to rejuvenate?' or 'Mindfulness exercises can be calming. Have you given them a shot?'
    - For users appearing overwhelmed: 'This seems tough; a professional might offer more tailored guidance.'

Key Boundaries:

1. **Avoid Professional Recommendations**: 
    - Make it clear you aren’t a substitute for medical or legal consultation. Use reminders like 'I offer emotional assistance, but it's important to seek expert advice on specific matters.'

2. **In Crises**: 
    - If a user signals a severe issue, respond promptly with 'Please reach out to a professional or emergency service for this concern.'

3. **Decision Guidance, Not Making**: 
    - Do not decide for the user. Instead, steer the conversation with inquiries such as 'What direction feels right to you?' or 'Have you evaluated all the possible choices?'

Communication Essentials:
- Maintain a consistently warm, empathetic, and patient demeanor.
- Your replies should be succinct yet full of compassion.
- **Avoid Repetitiveness**: Ensure your responses are diverse. While it's essential to be consistent, avoid echoing the same phrases too frequently.
- Your ultimate aim is to offer support, steer discussions, and occasionally redirect to specialized assistance when necessary.
"""


# Set page configuration
st.set_page_config(
    page_title="COCO Bot Training UI",
    layout="wide",
    initial_sidebar_state="expanded",
)

def main():
    down(
        """
        <style>
            .chat-box {
                max-height: 450px;
                overflow-y: auto;
                border: 1px solid #ECEFF1;
                border-radius: 5px;
                padding: 10px;
                background-color: whitesmoke;
            }
            .chat-message {
                margin-bottom: 15px;
            }
            .user-message {
                color: blue;
                margin-left: 10px;
            }
            .bot-message {
                color: purple;
            }
            .feedback-icon {
                border: 1px solid #000;
                padding: 2px;
            st.mark    border-radius: 5px;
                cursor: pointer;
                margin-right: 5px;
                display: inline-block;
            }
            .feedback-container {
                margin-top: 5px;
            }
            .bot-message.latest-response {
                background-color: #F5F5F5; /* Even lighter shade of grey */
                border-radius: 5px;
                padding: 5px;
                margin: 5px 0;
                color: black;
                font-weight: bold;
            }
            .instruction-box {
                border: 1px solid #ECEFF1;
                border-radius: 5px;
                padding: 10px;
                margin-bottom: 20px;
                background-color: silver;
                color: #333;
            }
            h1, h2, h3, h4, h5, h6 {
                color: #000;
            }
          .css-2trqyj {
                color: whitesmoke;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown(
        """
        <script>
            const inputBox = document.getElementById("user_input_key");
            inputBox.addEventListener("keyup", function(event) {
                if (event.key === "Enter") {
                    event.preventDefault();
                    const submitButton = document.querySelector("button[aria-label='Send']");
                    submitButton.click();
                }
            });
        </script>
        """, 
        unsafe_allow_html=True,
    )

    # Sidebar for Authentication and Title
    with st.sidebar:
        st.title("Authorization")
        auth_token = st.text_input("Enter Authentication Token:")
    
    # Initialize conversation history and feedback if they don't exist
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
        

    if auth_token == st.secrets["COCO_TOKEN"]:
        st.sidebar.success("Authentication Approved!")  # Display success message when user ID is entered
        user_id = st.sidebar.text_input("Enter your ID Number:")  # Prompt for user ID number
        
        if user_id:  # Make sure user_id is entered
            st.sidebar.success("User ID set!")  # Display success message when user ID is entered
                
           # Initialize conversation history and feedback
            conversation_history = []
            feedback_history = []
        
            # Configure API
            openai.api_type = st.secrets["API_TYPE"]
            openai.api_version = st.secrets["API_VERSION"]
            openai.api_base = st.secrets["API_BASE"]
            openai.api_key = st.secrets["API_KEY"]
                     
            # Instruction box.
            st.markdown(
                """
                
                <div class="instruction-box">
                <h2>Welcome to our COCO Training UI</h2>
                <p>We're excited to have you onboard to help us refine our support tool for family caregivers. Your insights are invaluable to this process, and we appreciate your time and effort. Below, you'll find a concise guide to interacting with our chatbot. Let's get started!</p>
                <h4>Quick Testing Guide</h4>
                  <li><strong>Start the conversation</strong> by typing a caregiving-related message.</li>
                  <li class="important">Remember to click "Submit" to send your message.</li>
                  <li><strong>Rate the highlighted chatbot's replies</strong> with "Thumb Up" or "Thumb Down" buttons.</li>
                  <li><strong>Engage with various topics</strong> to assess the chatbot's capabilities.</li>
                  <li><strong>If you wish to start over,</strong> click the "Reset" button to begin a new conversation.</li>
                  <li><strong>End the session</strong> when completed and fill concluding survey [OPTIONAL]</li>
                  <li><strong>Disclaimer.</strong> This tool is supportive, not a professional advice substitute.</li>
                  <p></p>
                  <p>Thank you for your participation and honest feedback. You're helping enhance this essential caregiving support tool!</p>
                  <p>[Optional]End Evaluation Survey: <a href="https://www.surveymonkey.com/r/7M9VPDP" target="_blank">COCO Test Survey</a></p>
                </div>
                """,
                unsafe_allow_html=True, 
            )
            
            st.write("---")
            st.markdown('<h1 style="color: whitesmoke;">COCO Chat Interface</h1>', unsafe_allow_html=True)

            # Initialize 'user_input' in session state if it doesn't exist
            if 'user_input' not in st.session_state:
                st.session_state.user_input = ''
            # If 'input_key' is not in session_state, initialize it
            if 'input_key' not in st.session_state:
                st.session_state.input_key = 'user_input_key_1'
                
            # Container for chat
            chat_container = st.empty()
            feedback_container = st.empty()
    
            # Render conversation in the chat box
            messages_html = "<div class='chat-box'>"
    
          # Check if there's an existing conversation history
            if 'conversation_history' not in st.session_state:
                st.session_state.conversation_history = []  # Initialize it if it doesn't exist
        
            # Initialize the messages HTML string
            for index, exchange in enumerate(st.session_state.conversation_history):
                    user_class = 'user-message'
                    bot_class = 'bot-message'
            
                    # Check if this is the last message in the conversation history
                    is_last_message = index == len(st.session_state.conversation_history) - 1
                    if is_last_message:
                        bot_class += ' latest-response'  # This class should be defined in your CSS
            
                    # Render the message without the msg_id 
                    messages_html += f"<div class='chat-message' id='message-{index}'>"  # id attribute uses the message index
                    messages_html += f"<span class='{user_class}'>You: {exchange['user']}</span><br>"
                    messages_html += f"<span class='{bot_class}'>🤖 Coco: {exchange['chatbot']}</span>"
                    if is_last_message:
                        # Add placeholders for feedback buttons; these will be replaced by real Streamlit buttons
                        messages_html += f"<div id='feedback-{index}'></div>"
                    messages_html += "</div>"
    
            chat_container.markdown(messages_html, unsafe_allow_html=True) 
                            
            if st.session_state.conversation_history:
                last_message_index = len(st.session_state.conversation_history) - 1
                render_feedback_buttons(user_id, last_message_index)  # Render feedback buttons for the last message
            
            user_input = st.text_input(label="", placeholder='Enter Message...', key=st.session_state.input_key)

            # st.markdown("""
            #     <script>
            #         function handleFeedback(messageIndex, isPositive) {
            #             let feedback = isPositive ? 'positive' : 'negative';
            #             let user_id = '%s';  // Replace this with the actual user ID.
            #             let params = {index: messageIndex, user_id: user_id, feedback: feedback};
                        
            #             // Make a POST request to the Streamlit server with the feedback data
            #             fetch(window.location.href, {
            #                 method: 'POST',
            #                 headers: {
            #                     'Content-Type': 'application/json'
            #                 },
            #                 body: JSON.stringify(params)
            #             }).then(response => response.json()).then(data => {
            #                 console.log('Success:', data);
            #             }).catch((error) => {
            #                 console.error('Error:', error);
            #             });
            #         }
            #     </script>
            # """ % user_id, unsafe_allow_html=True)
                            
            if st.button('Submit') and user_input.strip():    
                try:
                    completion = openai.ChatCompletion.create(
                        engine="CocoGPT_2",
                        messages=[
                            {"role": "system", "content": SYSTEM_MESSAGE},
                            {"role": "user", "content": user_input}         
                        ]
                    )
                    generated_text = completion.choices[0].message.content
                   # Append the new exchange to the conversation history
                    st.session_state.conversation_history.append({"user": user_input, "chatbot": generated_text})
                    
                    # After sending a message and receiving a response, record the user input and bot response in the database.
                    if user_input and generated_text:  # Ensure there's a message to record
                        try:
                            store_exchange(user_id, user_input, generated_text)  # feedback and explanation are None by default
                            # Change the key to reset the text_input
                            new_key_value = int(st.session_state.input_key.split('_')[-1]) + 1
                            st.session_state.input_key = f'user_input_key_{new_key_value}'
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Failed to store exchange: {e}")
                            
                except Exception as e:
                    st.write(f"An error occurred: {e} 😢")                              
            # Check if there's feedback in the URL query parameters and handle it
            feedback_data = st.experimental_get_query_params().get("feedback")
            message_index = st.experimental_get_query_params().get("index")
            user_id_data = st.experimental_get_query_params().get("user_id")
    
            if feedback_data is not None and message_index is not None and user_id_data is not None:
                # URL-decode and parse the feedback, explanation, and message index
                feedback = feedback_data[0]  # 'positive' or 'negative'
                index = int(message_index[0])
                user_id = user_id_data[0]

                # Clear the query parameters to avoid resubmitting the feedback
                st.experimental_set_query_params()

            # Reset conversation
            if st.button("Reset Conversation 🔄"):
                st.session_state.conversation_history = []  # Clear the conversation history
                st.experimental_rerun()  # This reruns the script, refreshing the conversation display
        else:
            st.sidebar.warning("Please enter your ID number to begin the conversation.")  # Warning if ID number is not entered
    else:
        st.write("Not Authenticated 😢") # Warning if authentication number is not entered
        
                    
if __name__ == "__main__":
    main()
