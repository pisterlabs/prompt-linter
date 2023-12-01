import openai
import streamlit as st
import requests
import base64
from openai import OpenAI

openai.api_key = st.secrets["OPENAI_API_KEY"]

client = OpenAI()



thread = client.beta.threads.create()



# Streamlit Configuration
st.set_page_config(
    page_title="AI Exploration",
    page_icon="https://haloarc.co.uk/wp-content/themes/halo/images/logo.png",
    layout="wide"
)

def get_state_variable(var_name, default_value):
    if 'st_state' not in st.session_state:
        st.session_state['st_state'] = {}
    if var_name not in st.session_state['st_state']:
        st.session_state['st_state'][var_name] = default_value
    return st.session_state['st_state'][var_name]

# Initialize session state for authentication
if 'is_authenticated' not in st.session_state:
    st.session_state.is_authenticated = False

def get_text_file_buffer(text):
    """Generates a text file buffer from a string."""
    import io
    buffer = io.StringIO()
    buffer.write(text)
    buffer.seek(0)
    return buffer

def display_dashboard():
    #st.title("GPT4 - API")


    # Create columns for the image, title, and page selector
    col_img, col_title, col_select = st.columns([2, 9, 3])

    # Upload the image to the left-most column
    with col_img:
        st.image("https://s3-eu-west-1.amazonaws.com/tpd/logos/5a95521f54e2c70001f926b8/0x0.png")


    # Determine the page selection using the selectbox in the right column
    with col_select:
        page_selection = st.selectbox(
            "Select a Page:",
            ["GPT4 - API", "IRS Helper", "DALL·E 3 - API", "Tips and Tricks"],
            index=0,
            key='page_selector'
        )


    # Set the title in the middle column based on page selection
    with col_title:
        if page_selection == "GPT4 - API":
            st.title("GPT4 - API")
        elif page_selection == "IRS Helper":
            st.title("IRS Helper")
        elif page_selection == "DALL·E 3 - API":
            st.title("DALL·E 3 - API")
        elif page_selection == "Tips and Tricks":
            st.title("Tips and Tricks for Prompting")
        elif page_selection == "Crashy":
            st.title("Crashy")



    def gpt4():

        if "openai_model" not in st.session_state:
            st.session_state["openai_model"] = "gpt-4-1106-preview"

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Create two columns
        col1, col2 = st.columns([4, 1])

        # Empty space in the first column
        col1.empty()

        # Add a button to clear the chat in the second column
        if col2.button('Clear Chat'):
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("What is up?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                for response in client.chat.completions.create(
                    model=st.session_state["openai_model"],
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                    stream=True,
                    max_tokens=4000,
                ):
                    


                    # Extract the message content using the proper JSON keys
                    #st.write(response)
                    # Access the first 'Choice' object in the 'choices' list.
                    choice = response.choices[0]  # 'choices' is a list, so you can use index access here.

                    # Access the 'ChoiceDelta' object from 'choice' which contains 'content' field.
                    choice_delta = choice.delta  # 'delta' is an attribute of 'Choice' object.

                    # Access the 'content' field from 'choice_delta'.
                    message_content = choice_delta.content  # 'content' is an attribute of 'ChoiceDelta' object.

                    # Append the extracted content to the 'full_response' string with a newline character
                    full_response += message_content if message_content is not None else ""

                    # Update the placeholder with the 'full_response' using Streamlit's markdown to render it
                    message_placeholder.markdown(full_response + "▌")

                    #content = response.choices
                    #full_response += content.text.value + "\n"  # `.text.value` instead of content["text"]["value"]
                    #full_response += response['choices'][0]['message']['content']
                    #message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            #st.write(thread)
        if st.button('Download Chat Log'):
            chat_log_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
            buffer = get_text_file_buffer(chat_log_str)
            b64 = base64.b64encode(buffer.getvalue().encode()).decode()
            st.markdown(f'<a href="data:file/txt;base64,{b64}" download="chat_log.txt">Download Chat Log</a>', unsafe_allow_html=True)




    # Main GPT-4 powered chat function with Knowledge Retrieval
    def gpthalo():

        st.session_state.assistant_id = "asst_BkdFnOIMY40wFJXyClgN6kAh"

        # Initialize session state for the model, messages, and assistant id if not already present
        if "openai_model" not in st.session_state:
            st.session_state["openai_model"] = "gpt-4-1106-preview"
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "assistant_id" not in st.session_state:
            st.session_state.assistant_id = None
            # Upload a file with an "assistants" purpose


        # Create two columns for chat interface
        col1, col2 = st.columns([4, 1])
        col1.empty()  # Placeholder

        if col2.button('Clear Chat'):
            st.session_state.messages = []

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Input for new messages
        if prompt := st.chat_input("How do I log a support ticket?"):

            # Print the current state of the messages to debug
            #print("Current state of messages before appending new message: ", st.session_state.messages)

            # Append new user message
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Print after appending to check update
            #st.write("State of messages after appending new user message: ", st.session_state.messages)

            #st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
            ]

            #st.write("Messages: ", messages)
            # Construct a string representation of the conversation
            conversation_str = "\n".join([f"{message['role']}: {message['content']}" for message in st.session_state.messages])
            #st.write(conversation_str)  # This will print the entire conversation as a string

            # Create and send a message to the thread
            message = client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=conversation_str
            )
            
            # Create a response using Knowledge Retrieval
            run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id="asst_BkdFnOIMY40wFJXyClgN6kAh",
                #instructions="Excellence Guru's primary function is to provide detailed, specific guidance and information on company processes using the core file named 'Halo New Starters Notion' as its sole information source. In cases where the query extends beyond the scope of the document, the GPT is to acknowledge the limitation and direct the user to consult with a member of the Center of Excellence team for further assistance. The GPT's interactions are grounded in the knowledge contained within this core file, ensuring accuracy and relevance. It will avoid speculation and will not use any information outside of this document. If a search within the document yields no answer, the GPT will state so transparently but will never mention the document's existence."
            )

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

                # Wait for the run to complete and get the response
                while run.status not in ["completed", "failed"]:
                    run = client.beta.threads.runs.retrieve(
                        thread_id=thread.id,
                        run_id=run.id
                    )

                # If the run completed successfully, get the latest messages
                if run.status == "completed":
                    messages = client.beta.threads.messages.list(
                        thread_id=thread.id
                    )
                    for msg in messages:
                        #if msg.role == "assistant":
                            #full_response += msg.content[0].delta.get("content", "")
                        if msg.role == "assistant":  # `.role` instead of msg["role"]
                                for content in msg.content:  # `.content` instead of msg["content"]
                                    if content.type == "text":  # `.type` instead of content["type"]
                                        full_response += content.text.value + "\n"  # `.text.value` instead of content["text"]["value"]
                            
                    message_placeholder.markdown(full_response)
                else:
                    message_placeholder.markdown("Something went wrong. Please try again.")

                # Append assistant's response
                st.session_state.messages.append({"role": "assistant", "content": full_response})

                # Final print to check conversation state
                #st.write("Final state of messages after appending assistant response: ", st.session_state.messages)

                #st.session_state.messages.append({"role": "assistant", "content": full_response})
                #print(st.session_state.messages)

        # Button to download the chat log
        if st.button('Download Chat Log'):
            chat_log_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
            buffer = get_text_file_buffer(chat_log_str)
            b64 = base64.b64encode(buffer.getvalue().encode()).decode()
            st.markdown(f'<a href="data:file/txt;base64,{b64}" download="chat_log.txt">Download Chat Log</a>', unsafe_allow_html=True)






    # Main GPT-4 powered chat function with Knowledge Retrieval
    def crashy():

        st.session_state.assistant_id = "asst_wYvoZYYge5BalhvIKOvUWFpV"

        # Initialize session state for the model, messages, and assistant id if not already present
        if "openai_model" not in st.session_state:
            st.session_state["openai_model"] = "gpt-4-1106-preview"
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "assistant_id" not in st.session_state:
            st.session_state.assistant_id = None
            # Upload a file with an "assistants" purpose


        # Create two columns for chat interface
        col1, col2 = st.columns([4, 1])
        col1.empty()  # Placeholder

        if col2.button('Clear Chat'):
            st.session_state.messages = []

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Input for new messages
        if prompt := st.chat_input("What is up?"):

            # Print the current state of the messages to debug
            #print("Current state of messages before appending new message: ", st.session_state.messages)

            # Append new user message
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Print after appending to check update
            #st.write("State of messages after appending new user message: ", st.session_state.messages)

            #st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
            ]

            #st.write("Messages: ", messages)
            # Construct a string representation of the conversation
            conversation_str = "\n".join([f"{message['role']}: {message['content']}" for message in st.session_state.messages])
            #st.write(conversation_str)  # This will print the entire conversation as a string

            # Create and send a message to the thread
            message = client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=conversation_str
            )
            
            # Create a response using Knowledge Retrieval
            run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id="asst_wYvoZYYge5BalhvIKOvUWFpV",
                #instructions="Excellence Guru's primary function is to provide detailed, specific guidance and information on company processes using the core file named 'Halo New Starters Notion' as its sole information source. In cases where the query extends beyond the scope of the document, the GPT is to acknowledge the limitation and direct the user to consult with a member of the Center of Excellence team for further assistance. The GPT's interactions are grounded in the knowledge contained within this core file, ensuring accuracy and relevance. It will avoid speculation and will not use any information outside of this document. If a search within the document yields no answer, the GPT will state so transparently but will never mention the document's existence."
            )

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

                # Wait for the run to complete and get the response
                while run.status not in ["completed", "failed"]:
                    run = client.beta.threads.runs.retrieve(
                        thread_id=thread.id,
                        run_id=run.id
                    )

                # If the run completed successfully, get the latest messages
                if run.status == "completed":
                    messages = client.beta.threads.messages.list(
                        thread_id=thread.id
                    )
                    for msg in messages:
                        #if msg.role == "assistant":
                            #full_response += msg.content[0].delta.get("content", "")
                        if msg.role == "assistant":  # `.role` instead of msg["role"]
                                for content in msg.content:  # `.content` instead of msg["content"]
                                    if content.type == "text":  # `.type` instead of content["type"]
                                        full_response += content.text.value + "\n"  # `.text.value` instead of content["text"]["value"]
                            
                    message_placeholder.markdown(full_response)
                else:
                    message_placeholder.markdown("Something went wrong. Please try again.")

                # Append assistant's response
                st.session_state.messages.append({"role": "assistant", "content": full_response})

                # Final print to check conversation state
                #st.write("Final state of messages after appending assistant response: ", st.session_state.messages)

                #st.session_state.messages.append({"role": "assistant", "content": full_response})
                #print(st.session_state.messages)

        # Button to download the chat log
        if st.button('Download Chat Log'):
            chat_log_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
            buffer = get_text_file_buffer(chat_log_str)
            b64 = base64.b64encode(buffer.getvalue().encode()).decode()
            st.markdown(f'<a href="data:file/txt;base64,{b64}" download="chat_log.txt">Download Chat Log</a>', unsafe_allow_html=True)









    def chatgpt():

        if "openai_model" not in st.session_state:
            st.session_state["openai_model"] = "gpt-3.5-turbo-1106"

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Create two columns
        col1, col2 = st.columns([4, 1])

        # Empty space in the first column
        col1.empty()

        # Add a button to clear the chat in the second column
        if col2.button('Clear Chat'):
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("What is up?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                for response in openai.ChatCompletion.create(
                    model=st.session_state["openai_model"],
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                    stream=True,
                ):
                    full_response += response.choices[0].delta.get("content", "")
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        if st.button('Download Chat Log'):
            chat_log_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
            buffer = get_text_file_buffer(chat_log_str)
            b64 = base64.b64encode(buffer.getvalue().encode()).decode()
            st.markdown(f'<a href="data:file/txt;base64,{b64}" download="chat_log.txt">Download Chat Log</a>', unsafe_allow_html=True)


    def dalle():


        # Generate an image from a text prompt
        def generate_image(prompt):
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            image_url = response.data[0].url
            return image_url

        # Download the image and display it in Streamlit
        def display_image(url):
            response = requests.get(url)
            response.raise_for_status()
            image_bytes = response.content
            st.image(image_bytes, caption="Generated Image", use_column_width=True)

        # Text input for the prompt
        prompt = st.text_input('Enter your image prompt:')

        # Button to generate and display the image
        if st.button('Generate Image'):
            image_url = generate_image(prompt)
            display_image(image_url)

    def display_faq():
        #st.title("Tips and Tricks for GPT-4 and DALL·E 2 APIs")
        
        st.header("GPT-4 API - FAQ")
        
        st.subheader("1. How to Frame Specific Questions for Accurate Answers?")
        st.write("""
        Being vague will often yield general or ambiguous responses. Specificity narrows down the scope for GPT-4.
        - **Bad**: "Tell me about car repairs."
        - **Good**: "Explain the process of calibrating the front radar on a 2021 VW Golf and why it's important."
        """)

        st.subheader("2. How to Get Simplified Explanations?")
        st.write("""
        If you're looking for easy-to-understand answers, explicitly ask for them.
        - **Example**: "Explain like I'm 5, how does paintless dent repair work?"
        """)
        
        st.subheader("3. How to Receive a Concise Answer?")
        st.write("""
        To receive a summary-like answer that cuts through the fluff, guide the model with cues.
        - **Example**: "Summarize in one sentence, what steps should I take immediately after a minor car collision?"
        """)

        st.subheader("4. How to Ask Follow-up Questions?")
        st.write("""
        GPT-4 can handle a series of related queries. Just ensure your follow-up is directly related to the prior question for context.
        - **Example**: "You mentioned that a damaged frame can be risky. What are the signs of a damaged car frame?"
        """)
        
#        st.header("DALL·E 2 API - FAQ")
        
#        st.subheader("1. How to Get More Detailed Images?")
#        st.write("""
#        DALL·E 2 works best when you're descriptive. It gives the model a better understanding of what you're envisioning.
#        - **Bad**: "Draw a car."
#        - **Good**: "Generate a photorealistic image of a red 2020 Ford Mustang with a cracked windshield."
#        """)

#        st.subheader("2. How to Tweak or Iterate Over Generated Images?")
#        st.write("""
#        A slight change in phrasing can lead to different outputs. Feel free to iterate on your initial idea.
#        - **Example**: "Produce an image of a red 2020 Ford Mustang with a shattered windshield."
#        """)
        
#        st.subheader("3. How to Get Realistic Images?")
#        st.write("""
#        If you're looking for photorealistic outputs, it helps to specify this in your prompt.
#        - **Example**: "Generate a photorealistic image of a car collision involving a white SUV and a blue sedan."
#        """)

#        st.subheader("4. How to Influence the Style or Aesthetic?")
#        st.write("""
#        You can direct the style of the generated image by including artistic descriptors.
#        - **Example**: "Create an image of a damaged car in the style of a vintage 1950s advertisement."
#        """)









    if page_selection == "GPT4 - API":
        gpt4()
    elif page_selection == "Crashy":
        crashy()

    elif page_selection == "IRS Helper":
        gpthalo()
        
    elif page_selection == "DALL·E 3 - API":
        dalle()
    elif page_selection == "Tips and Tricks":
        display_faq()



# Logic for password checking
def check_password():
    if not st.session_state.is_authenticated:
        password = st.text_input("Enter Password:", type="password")


            
        
        if password == st.secrets["db_password"]:
            st.session_state.is_authenticated = True
            st.rerun()
        elif password:
            st.write("Please enter the correct password to proceed.")
            
        blank, col_img, col_title = st.columns([2, 1, 3])

        # Upload the image to the left-most column
        with col_img:
            st.image("https://s3-eu-west-1.amazonaws.com/tpd/logos/5a95521f54e2c70001f926b8/0x0.png")


        # Determine the page selection using the selectbox in the right column
        with col_title:
            #st.title("Created By Halo")
            st.write("")
            st.markdown('<div style="text-align: left; font-size: 40px; font-weight: normal;">Created By Halo*</div>', unsafe_allow_html=True)
            
        blank2, col_img2, col_title2 = st.columns([2, 1, 3])

        # Upload the image to the left-most column
        with col_img2:
            st.image("https://th.bing.com/th/id/OIP.42nU_MRx_INTLq_ejdHxBQHaCe?pid=ImgDet&rs=1")


        # Determine the page selection using the selectbox in the right column
        with col_title2:
            
            #st.title("Powered By IRS")
            st.markdown('<div style="text-align: left; font-size: 30px; font-weight: normal;">Powered By IRS</div>', unsafe_allow_html=True)
        # Fill up space to push the text to the bottom
        for _ in range(20):  # Adjust the range as needed
            st.write("")

        # Write your text at the bottom left corner
        st.markdown('<div style="text-align: right; font-size: 10px; font-weight: normal;">* Trenton Dambrowitz, Special Projects Manager, is "Halo" in this case.</div>', unsafe_allow_html=True)



    else:
        print("Access granted, welcome to the app.")
        display_dashboard()


check_password()

