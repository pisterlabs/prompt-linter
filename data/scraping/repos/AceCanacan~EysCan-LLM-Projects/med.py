import streamlit as st
import openai
from streamlit_chat import message

openai.api_key = "INSERT OPENAI KEY"

def med_chat(st):
    # Initialise session state variables
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
    if 'model_name' not in st.session_state:
        st.session_state['model_name'] = []
    if 'analysis' not in st.session_state:
        st.session_state['analysis'] = ""  # Add a session variable to store analysis result
    if 'provided' not in st.session_state:
        st.session_state['provided'] = False
        st.session_state['symptoms'] = ''
        st.session_state['duration'] = ''
        st.session_state['patterns'] = ''
        st.session_state['medications'] = ''

    # Generate a response
    def generate_response(prompt):
        st.session_state['messages'].append({"role": "user", "content": prompt})

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=st.session_state['messages'],
            max_tokens=256,
            temperature=0.6
        )
        response = completion.choices[0].message.content.strip()
        st.session_state['messages'].append({"role": "assistant", "content": response})

        return response
    
    def analyze_propmpt(prompt):
        st.session_state['messages'].append({"role": "user", "content": prompt})

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=st.session_state['messages'],
            max_tokens=3000,
            temperature=0.1
        )
        response = completion.choices[0].message.content.strip()
        st.session_state['messages'].append({"role": "assistant", "content": response})

        return response

    # Analyze user messages
    def generate_analysis():
        user_messages = st.session_state['past']  # Get the last 5 user messages
        merged_user_messages = " ".join(user_messages)  # Merge the user messages into a single prompt

        st.session_state['messages'] = [
            {"role": "user", "content": merged_user_messages},
            {"role": "assistant", "content": f"""
                                            You are an empathetic, AI-powered assistant, designed to provide support in a medical context. Your user is a patient who has already provided detailed information about their health concerns. Your task is to analyze this information from a medical perspective and provide a summary of the possible condition they might be experiencing, potential causes, and general advice on symptom management.

                                            For example, if the patient has described symptoms of diarrhea, you might suggest that they could be experiencing food poisoning, which can be caused by consuming contaminated food or water. You could then provide general advice such as staying hydrated and resting.

                                            Remember, your role is purely informational, and under no circumstances should you attempt to provide medical advice or diagnosis. Always take into account the context provided by the patient and ensure your responses are both medically accurate and understandable to a layperson.

                                            After providing the summary, remind the patient that this is an AI-generated response and it's crucial for them to consult with a healthcare professional who can provide advice based on a comprehensive understanding of their symptoms and medical history

                                            Do not show any signs of uncertainty such as "i cannot pinpoint the exact cause". The patient knows you are a bot, and knows that you have limitations. No need to tell them.

                                            Separate per part

                                            - Possible Diagnosis
                                            (talk about the probable disease or condition they have)
                                            - Potentaial Causes
                                            (explain the potential cause of their condition)
                                            - Measures
                                            (explain what measures can they take to alleviaet their problem)

                                           """}
        ]

        st.session_state['analysis'] = analyze_propmpt("")  # Update the analysis session variable

    st.image('medgpt_logo.png')
    st.markdown("Developed by [Eys](https://www.linkedin.com/in/acecanacan/)", unsafe_allow_html=True)
    st.write("This is not a substitute for professional medical advice but only an exploration on how LLMs can be integrated to providing medical services.")
    st.markdown('')


    if not st.session_state['provided']:
        with st.form(key='context_form'):

            st.write('Before you begin, please fill out these forms to help the AI understand you better and provide more meaningful support based on your current situation.')

            st.session_state['symptoms'] = st.text_area("Can you describe in detail the primary symptom that is concerning you?", value=st.session_state['symptoms'])
            st.session_state['duration'] = st.text_area("How long have you been experiencing this symptom?", value=st.session_state['duration'])
            st.session_state['patterns'] = st.text_area("Have you noticed any patterns or triggers related to your symptoms?", value=st.session_state['patterns'])
            st.session_state['medications'] = st.text_area("Have you tried any treatments or taken any new medications recently, and if so, did they affect your symptom?", value=st.session_state['medications'])

            if st.form_submit_button("Submit"):
                st.session_state['messages'].append({"role": "system", "content": f"""Symptoms: {st.session_state['symptoms']}. Duration of symptoms: {st.session_state['duration']}. Patterns of symptoms: {st.session_state['patterns']} Medications Taken if any {st.session_state['medications']}
                                                                                      This is the context about the patient
                                                                                      You are a highly empathetic, AI-powered assistant, providing support in a medical context. 
                                                                                      Your user is a patient who has answered the following questions about their health concerns:
                                                                                      Your task is not to diagnose or analyze the patient's condition. Instead, you are to use the information provided to ask follow-up questions, 
                                                                                      aimed at gathering more details and understanding the patient's situation better. Probe into different areas of their lifestyle, such as diet, 
                                                                                      exercise, stress levels, and sleep patterns, which might be relevant to their symptoms. It's important to remain respectful, empathetic, 
                                                                                      and understanding in your tone throughout the conversation, making the patient feel heard and acknowledged.
                                                                                      After a thorough conversation, when you think you have gathered enough information, you should inform the patient that 
                                                                                      you've completed your questions and the data will be reviewed by medical professionals for further analysis. 
                                                                                      Remember, your role is purely informational, and under no circumstances should you attempt to provide medical advice or diagnosis.
                                                                                      Take into account the context provided by the patient
                                                                                      """})
                st.session_state['provided'] = True
                st.experimental_rerun()


    if st.session_state['provided']:

        st.write('You can now begin your AI consultation. Say Hi!.')

        # Container for chat history
        response_container = st.container()
        # Container for text box
        container = st.container()



        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    if i < len(st.session_state["past"]):
                        message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                    message(st.session_state["generated"][i], key=str(i))

        # Add a new section for displaying the analysis
        if st.session_state['analysis']:
            st.markdown(f"## Analysis")
            st.markdown(st.session_state['analysis'])

        # New Session button
        if st.button("New Session"):
            st.session_state['generated'] = []
            st.session_state['past'] = []
            st.session_state['messages'] = []
            st.session_state['model_name'] = []
            st.session_state['analysis'] = ""  # Reset the analysis session variable
            st.session_state['provided'] = []
            st.session_state['symptoms'] = ''
            st.session_state['duration'] = ''
            st.session_state['medications'] = ''
            st.session_state['patterns'] = ''
            st.experimental_rerun()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_area("You:", key='input', height=100)
                submit_button = st.form_submit_button(label='Send')

            if submit_button and user_input:
                output = generate_response(user_input)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)
                st.session_state['model_name'].append("GPT-3.5")
                st.experimental_rerun()


            # Replace the if condition for the analysis button with a permanent button
            analysis_button = st.button("Generate Analysis")
            if analysis_button:
                generate_analysis()
                st.experimental_rerun()


if __name__ == "__main__":
    med_chat(st)
