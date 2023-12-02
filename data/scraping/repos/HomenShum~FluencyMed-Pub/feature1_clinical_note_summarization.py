import openai
import json
import streamlit as st

"""
This program can take patient_id as input and return the corresponding patient's notes from the database.
The notes can be used as input for the GPT model to generate an analysis that is easy to read and useful for categorization tasks.
"""

##### Settings ############################################################################################################

openai.api_key = st.secrets["openai_api_key"]


system_prompt = """ You are an AI assistant specialized in biomedical topics. You are provided with a text description from a patient's screening notes. Analyze the patient's notes and provide information useful for physicians. Here are your instructions:

                    - Highlight conditions, symptoms, medical history, and any other information that can be helpful to the physicians.

                    - Keep in mind that grouping a large number of diseases into manageable categories for statistical analysis and reporting is useful for physicians. 

                    - Ensure the conversation includes information such as the type of disease, cause, location in the body, and patient's age and sex. 

                    - Avoid providing medical advice or diagnostic information. 

                    - Ensure the output is in markdown bullet point format for clarity.

                    - Encourage the user to consult a healthcare professional for advice."""


##### Functions ############################################################################################################

def get_patient_notes(patient_id):
    """Get the patient's notes"""
    if patient_id == "123":
        patient_notes = {
            "patient_id":   patient_id,
            "notes":"""     Hi doctor, I am a 26 year old male. I am 5 feet and 9 inches tall and weigh 255 pounds. 
                            When I eat spicy food, I poop blood. Sometimes when I have constipation as well, 
                            I poop a little bit of blood. I am really scared that I have colon cancer. 
                            I do have diarrhea often. I do not have a family history of colon cancer. 
                            I got blood tests done last night. Please find my reports attached.
                    """
        }
    else:
        return "Patient not found"
    return json.dumps(patient_notes)


def patient_note_analysis(user_prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",  # use your actual model here
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
    )
    return response["choices"][0]["message"]["content"]


##### Main ############################################################################################################
id_search   =   "123"
user_prompt =   f"""Can you analyze patient {id_search}'s notes? \n
                    {get_patient_notes(id_search)} \n
                    Output in markdown bullet points as follows: \n
                    **Patient information:** \n
                    **Symptoms and conditions:** \n
                    **Concerns:** \n
                    **Medical history and investigations:** \n
                    **Medical departments:** \n
                """
patient_note_analysis_output = patient_note_analysis(user_prompt)
# print (patient_note_analysis_output)
