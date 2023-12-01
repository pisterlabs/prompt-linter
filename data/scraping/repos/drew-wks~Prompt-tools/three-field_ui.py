import openai
import streamlit as st
# accessible at https://chatbotsys-tasty.streamlit.app/
from trubrics.integrations.streamlit import FeedbackCollector

openai.api_key = st.secrets["OPENAI_API_KEY"]  # sets api key by pulling it from streamlit secrets file
# to run in browser: streamlit run /Users/drew_wilkins/Drews_Files/Drew/Python/pythonProject/WebApp/chatbot_system.py

def main():
    def get_completion(prompt):
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,  # this is the degree of randomness of the model's output
        )
        return response.choices[0].message["content"]

    def get_completion_from_messages(messages,
                                     model="gpt-3.5-turbo",
                                     temperature=0):
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=int(max_tokens),
        )
        return response.choices[0].message["content"]

    collector = FeedbackCollector(
        component_name="default",
        email=st.secrets["TRUBRICS_EMAIL"],  # Retreives my Trubrics credentials from secrets file
        password=st.secrets["TRUBRICS_PASSWORD"],  # https://blog.streamlit.io/secrets-in-sharing-apps/
    )

    st.title("Business Task Assistant")
    st.write(
        "We developed this proof of concept for situations where you would like Chat GPT to perform a task with text supplied by you. By separating the text from the instruction, you can get better results.")
    st.write("Go ahead, give it a try!")



    with st.form("round1"):
        #model = st.selectbox("Select a model:", ("gpt-3.5-turbo", "gpt-3.5-turbo-16k"))  # this works it's just not needed right now
        max_tokens = 2000
        instruction = st.text_area("Instructions the model must follow go here:", value="Example: You are a VP of Product Development. Write a proposal about using AI to improve our firm's decisions-making process for handling new product investments")
        text = st.text_area("(Optional) Add any text on which the instuctions will act:", value="Example: here is a proposal to use an example")
        system_message = st.text_area("(Optional) Specify any conditions on how the instructions are carried out goes here", value="Example: Here are our company values...")
        submit_button1 = st.form_submit_button("Process inputs")

    user_message = f"""{instruction}```{text}```"""

    if submit_button1:
        messages = [
            {'role': 'system','content': system_message},
            {'role': 'user','content': user_message},
        ]
        response = get_completion_from_messages(messages)
        st.write(f"**Response:** {response}")

        st.markdown("""---""")
        st.write("How well did the model do?")
        st.write(f"{model}")

        collector.st_feedback(feedback_type="thumbs",
                              model=model,
                              open_feedback_label="[Optional] How did the model do?",
                              )
if __name__ == '__main__':
    main()
