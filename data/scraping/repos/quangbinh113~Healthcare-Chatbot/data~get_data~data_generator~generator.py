import guidance 
import dotenv
import os
import json
from utils import split_to_contexts, format_json, update_data
import time

# load the openai key from .env file
dotenv.load_dotenv()

open_ai_key = os.getenv("OPENAI_API_KEY")
guidance.llm = guidance.llms.OpenAI("gpt-3.5-turbo-0613", api_key=open_ai_key)


# define the guidance program
create_conversation = guidance(
    '''
    {{#system~}}
        You are my AI bot to generate conversation between a chatbot and a patient. You produce each response in the character's perspective.
    {{~/system}}
    
    {{#user~}}
        I want you to generate multiple conversations related to a given context. The conversation is between the patient and the health-care chatbot called Medi. The conversation may be about one of following case or can combine them or based on your imagination:
            The patient has some symptom in the context and needs help, Medi will give advice and diagnosis based on its knowledge obtained from the context.
            The patient just has some questions about the disease mentioned in the context and Medi gives the explanation for these questions.

        You must write the output conversation in a following JSON format:

        {
        “context 0”: patient’s turn,
        “context 1”: Medi’s turn,
        “context 2”: patient’s turn,
        “context 3”: Medi’s turn,
        “context 4”: patient’s turn,
        “response”: Medi’s turn
        }

        The number of words in each turn of Medi and patient must not exceed 15 words no matter what. The conversation need not contain all the information in the context and should be as natural as possible. The replies of Medi and the patient must be simple and comprehensive for each other as much as possible.
        
        The context is: {{context}}
        
        
        From the context, you must output a list of 3 dialogues in the following format:

        [dialogue 1, 
        dialogue 2,
        dialogue 3,
        ]
        This is an example of output: 

        [
        {
        "context 0": "Medi, what causes gout? Is it a type of arthritis?",
        "context 1": "Gout is a type of arthritis caused by the buildup of uric acid crystals in the joints. It can cause sudden attacks of joint pain.",
        "context 2": "Can gout be prevented? How is it treated?",
        "context 3": "Gout can be prevented by avoiding foods high in purines. Treatment may include medications to lower uric acid levels and lifestyle changes.",
        "context 4": "Thank you for the clarification, Medi.",
        "response": "You're welcome! Let me know if you have any more questions."
        },
        {
        "context 0": "Medi, I have a question about back pain. What causes it?",
        "context 1": "Back pain can have various causes, such as muscle strain, herniated discs, or spinal abnormalities.",
        "context 2": "Is there anything I can do to prevent back pain?",
        "context 3": "Maintaining good posture, exercising regularly, and lifting properly can help prevent back pain.",
        "context 4": "Thank you for the information, Doctor. I'll keep that in mind.",
        "response": "You're welcome. Let me know if you have any more questions."
        },
        {
        "context 0": "Hi, Medi.",
        "context 1": "Hello. How can I assist you today?",
        "context 2": "Lately, I've been feeling my heart racing and skipping beats. Should I be concerned?",
        "context 3": "Based on your symptoms, it's possible that you're experiencing arrhythmia. I recommend consulting a healthcare professional for a proper diagnosis.",
        "context 4": "If it is arrhythmia, what are the treatment options available?",
        "response": "Treatment for arrhythmia may include medication, medical procedures, or surgery, depending on the type and severity of the condition."
        }
        ]


    {{~/user}}
    
    {{#assistant~}}
        {{gen 'data' n=15 temperature=1}}
    {{~/assistant}}
    ''',
        )



def generate_conversation(context, disease, save_path):
    while True:
        try: 
            conversations = create_conversation(context=context)["data"]
            time.sleep(10)
            break
        except Exception as e:       
            print('Rate limit exceeded, wait for 60 seconds')
            time.sleep(60)
    path = f'{save_path}/{disease}.json'
    # Check if the file exists
    if not os.path.exists(path):
        with open(path, "w") as file:
            file.write("[]")  # Write an empty JSON object to the file
    update_data(conversations, path)

        
def run(base_path, save_path):
    print("Start generating conversation")
    print("\n")
    
    # get the list of diseases
    diseases = os.listdir(base_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # ignore = ['Arrhythmia', 'Arthritis', 'Asthma', 'Back_pain', 'Cancer', 'Cardiovascular_disease', 'Cervicitis', 'Common_cold', 'Diabetes', 'Encephalitis', 'Gastritis', 'Hepatitis', 'Hepatitis_B', 'Hepatitis_C', 'Hepatitis_E', 'Hypertension', 'Infectious_diseases', 'Inflammatory_bowel_disease', 'Influenza', 'Kidney_disease', 'Meningitis', 'Mental_illness', 'Migraine', 'Obesity']
    for i, disease in enumerate(diseases):
        # if disease in ignore:
        #     continue
        print(f"* The topic {i+1}/{len(diseases)}: {disease}")
        # the path to the file store the knowledge about the disease
        text_path = os.path.join(base_path, disease, f"{disease}.txt")
        contexts = split_to_contexts(text_path)
        for j, context in enumerate(contexts):
            generate_conversation(context, disease, save_path)
            print(f"    - Finished {j+1}/{len(contexts)}")

            
    # format json file stored in the directory to make it more readable
    format_json(save_path)
            
            
if __name__ == "__main__":
    # The path to folder store the raw data
    base_path = '../data_crawler/raw_data'
    # The path to folder store the generated conversation data
    save_path = '../../data/conservation_data'
    run(base_path, save_path)