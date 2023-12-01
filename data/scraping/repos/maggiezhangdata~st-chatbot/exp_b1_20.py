from openai import OpenAI
import streamlit as st
import time
import re  # Import regular expressions

st.title("聊天机器人")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
assistant_id = st.secrets["assistant_id_b1_20"]
speed = 30




if "thread_id" not in st.session_state:
    thread = client.beta.threads.create()
    st.session_state.thread_id = thread.id

if "show_thread_id" not in st.session_state:
    st.session_state.show_thread_id = False

if "first_message_sent" not in st.session_state:
    st.session_state.first_message_sent = False

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"],unsafe_allow_html=True)


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")
st.sidebar.markdown("#### 完成对话后，复制对话编号并粘贴至下方问卷的文本输入框中。\n:star: 请勿将其输入至聊天机器人对话页面。")
st.sidebar.info(st.session_state.thread_id)
st.sidebar.caption("请复制上述对话编号。")

def update_typing_animation(placeholder, current_dots):
    """
    Updates the placeholder with the next stage of the typing animation.

    Args:
    placeholder (streamlit.empty): The placeholder object to update with the animation.
    current_dots (int): Current number of dots in the animation.
    """
    num_dots = (current_dots % 6) + 1  # Cycle through 1 to 3 dots
    placeholder.markdown("回答生成中，请耐心等待" + "." * num_dots)
    return num_dots


# Handling message input and response
max_messages = 30  # 10 iterations of conversation (user + assistant)

if len(st.session_state.messages) < max_messages:
    
    user_input = st.chat_input("")
    if not st.session_state.first_message_sent:
        st.markdown(
            "您可以通过复制粘贴<br>"
            "<span style='color: #8B0000;'>我最近很心烦，请告诉我该怎么办？</span><br>"
            "到下面👇🏻的对话框，开启和聊天机器人的对话，寻求建议和帮助。", unsafe_allow_html=True
        )
    if user_input:
        st.session_state.first_message_sent = True
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            waiting_message = st.empty()  # Create a new placeholder for the waiting message
            dots = 0

            
#------------------------------------------------------------------------------------------------------------------------------#
            def format_response(response):
                """
                Formats the response to handle bullet points and new lines.
                Targets both ordered (e.g., 1., 2.) and unordered (e.g., -, *, •) bullet points.
                """
                # Split the response into lines
                lines = response.split('\n')
                
                formatted_lines = []
                for line in lines:
                    # Check if the line starts with a bullet point (ordered or unordered)
                    if re.match(r'^(\d+\.\s+|[-*•]\s+)', line):
                        formatted_lines.append('\n' + line)
                    else:
                        formatted_lines.append(line)

                # Join the lines back into a single string
                formatted_response = '\n'.join(formatted_lines)

                return formatted_response.strip()
        
            import time
            max_attempts = 2
            attempt = 0
            while attempt < max_attempts:
                try:
                    update_typing_animation(waiting_message, 5)  # Update typing animation
                    # raise Exception("test")
                    message = client.beta.threads.messages.create(thread_id=st.session_state.thread_id,role="user",content=user_input)
                    run = client.beta.threads.runs.create(thread_id=st.session_state.thread_id,assistant_id=assistant_id,)
                    
                    # Wait until run is complete
                    while True:
                        run_status = client.beta.threads.runs.retrieve(thread_id=st.session_state.thread_id,run_id=run.id)
                        if run_status.status == "completed":
                            break
                        dots = update_typing_animation(waiting_message, dots)  # Update typing animation
                        time.sleep(0.3) 
                    # Retrieve and display messages
                    messages = client.beta.threads.messages.list(thread_id=st.session_state.thread_id)
                    full_response = messages.data[0].content[0].text.value
                    full_response = format_response(full_response)  # Format for bullet points
                    chars = list(full_response)
                    # speed = 20  # Display 5 Chinese characters per second
                    delay_per_char = 1.0 / speed
                    displayed_message = ""
                    waiting_message.empty()
                    for char in chars:
                        displayed_message += char
                        message_placeholder.markdown(displayed_message)
                        time.sleep(delay_per_char)  # Wait for calculated delay time
                    break
                except:
                    attempt += 1
                    if attempt < max_attempts:
                        print(f"An error occurred. Retrying in 5 seconds...")
                        time.sleep(5)
                    else:
                        error_message_html = """
                            <div style='display: inline-block; border:2px solid red; padding: 4px; border-radius: 5px; margin-bottom: 20px; color: red;'>
                                <strong>网络错误:</strong> 请重试。
                            </div>
                            """
                        full_response = error_message_html
                        waiting_message.empty()
                        message_placeholder.markdown(full_response, unsafe_allow_html=True)

#------------------------------------------------------------------------------------------------------------------------------#



            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )

else:

    if user_input:= st.chat_input(""):
        with st.chat_message("user"):
            st.markdown(user_input)
        

    
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.info(
                "已达到此聊天机器人的最大对话限制，请复制侧边栏对话编号。将该对话编号粘贴在下面的文本框中。"
            )
    st.chat_input(disabled=True)

    # # Button to copy thread ID
    # if st.button("复制thread_id"):
    #     st.session_state.show_thread_id = True

    # # When thread ID is shown, update the flag to hide the input box
    # if st.session_state.get('show_thread_id', False):
    #     st.session_state['thread_id_shown'] = True  # Set the flag to hide the input box
    #     st.markdown("#### Thread ID")
    #     st.info(st.session_state.thread_id)
    #     st.caption("请复制以上文本框中的thread_id。")



#----------------------------------------------
# else:
#     user_input = st.chat_input("最近还好吗？")
#     st.session_state.messages.append({"role": "user", "content": user_input})

#     # with st.chat_message("user"):
#     #     st.markdown(user_input)

#     with st.chat_message("assistant"):
#         message_placeholder = st.empty()
#         message_placeholder.info(
#             "注意：已达到此聊天机器人的最大消息限制，请点击复制thread_id按钮，复制thread_id。将该thread_id粘贴在下一页的回答中。"
#         )
    

#     if st.button("复制thread_id"):
#         st.session_state.show_thread_id = True

#     if st.session_state.show_thread_id:
#         st.markdown("#### Thread ID")
#         st.info(st.session_state.thread_id)
#         st.caption("请复制以上文本框中的thread_id。")




