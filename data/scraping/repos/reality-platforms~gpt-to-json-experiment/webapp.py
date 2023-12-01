import streamlit as st
import json
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate


INITIAL_INPUT_TEXT="""Welcome To Simply Dental
THE TOP-REVIEWED DENTAL PRACTICE
IN LOS ALGODONES, MEXICO
Simply Dental is committed to providing world-class cosmetic, restorative, and general dentistry.  Our modern office is operated by a team of highly experienced dentists who provide technology-assisted treatments and superior personalized care during every procedure. That is why patients abroad, as well as in Los Algodones and all over Mexico, recommend Simply Dental to their friends and family. We are proudly Number One when it comes to positive Google reviews.

"""
INITIAL_QUESTIONS_TEXT="Which tags apply to this dental office? Are they trustworthy? How would you rate their friendliness on a scale from 1 to 10? How would you rate their quality on a scale from 1 to 10?"
INITIAL_SAMPLE_JSON='{"trustworthy": boolean, "caters_to_americans": boolean, "friendliness": integer, "high_quality": integer}'

st.title('GPT to JSON')


questions_prompt = PromptTemplate(
    input_variables=["text", "questions", "json"],
    template="Parse the following text: \n{text} \n\n return these questions in form of the json provided: \n\n json: {json} \n questions:\n{questions} \n\n explain your answer before you fill out the json. End your response with the json object. Make sure that all your answers exactly match the json type specified. Label the json object with the label 'json:'",
)
llm = OpenAI(temperature=0)

@st.cache
def get_original_response(input_text: str, questions: str, sample_json: str) -> str:
    try:
        formatted_questions_prompt = questions_prompt.format(
            text=input_text,
            questions=questions,
            json=sample_json
        )
        return llm(formatted_questions_prompt)
    except:
        return 'Failed to fetch response'

def extract_json(original_response: str) -> str:
    try:
        return original_response[original_response.find('json:') + len('json:'):]
    except:
        return 'Failed to fetch json'


input_text = st.text_area("Input text", value=INITIAL_INPUT_TEXT, max_chars=None, key=None, placeholder="We the people, in order to form...", label_visibility="visible")
questions = st.text_area("Questions", value=INITIAL_QUESTIONS_TEXT, max_chars=None, key=None, placeholder="What tags are associated with this text? How would you rate its energy level on a scale from 1 to 10?", label_visibility="visible")
sample_json = st.text_area("Sample JSON", value=INITIAL_SAMPLE_JSON, max_chars=None, key=None, placeholder='{{ "confident": boolean, "intellectual": boolean, "goofy": boolean, "energy_level": number}}.', label_visibility="visible")

displayed_json = st.text('Generating json...')

original_response = get_original_response(input_text, questions, sample_json)
full_response = st.text_area("Full response", value=original_response.strip(), max_chars=None, key=None, placeholder="We the people, in order to form...", label_visibility="visible")

extracted_json = extract_json(original_response)

displayed_json.write(json.dumps(json.loads(extracted_json), indent=2))


