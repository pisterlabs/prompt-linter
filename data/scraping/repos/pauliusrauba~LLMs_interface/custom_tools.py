import pandas as pd
import pickle
import numpy as np
import ast
import matplotlib.pyplot as plt
import streamlit as st
import PIL
import io
from langchain.utilities import WikipediaAPIWrapper
from langchain.agents import tool
from autoprognosis.utils.serialization import load_from_file
from QRisk_model import QRisk3Model

from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain
from langchain.document_loaders import PyPDFLoader
from autoprognosis.plugins.explainers import Explainers

import pickle

import io
import urllib
import base64

def person_data(name: str) -> pd.DataFrame:
    """Returns the data of a client. Use this when there is a need to 
    extract data of a patient. The function returns a pandas dataframe."""
    return pd.read_csv("./cvd/person_cvd.csv")

@tool
def plot_feature_importance_heart_risk(name: str) -> list:
    """Use this for any question related to plotting the feature importance of heart risk for any patient or any model.
    The input should always be an empty string and this function will always return a tuple that contains the top three risks
    and their associated scores. It will always plot of feature importances. """
    # Assume that clf is the trained random forest model and X_train is the training data
    # Get feature importance
    #clf = get_classifier('')
    #feature_importance = clf.feature_importances_

    # Sort features by importance
    #sorted_idx = np.argsort(feature_importance)

    column_names = np.array([
    'Age of the patient',
    'Sex of the patient',
    'Type of chest pain',
    'Resting blood pressure',
    'Serum cholesterol in mg/dl',
    'Fasting blood sugar > 120 mg/dl',
    'Resting electrocardiographic results',
    'Maximum heart rate achieved',
    'Exercise induced angina',
    'ST depression induced by exercise',
    'The slope of the peak exercise ST segment',
    'Number of major vessels',
    'Thalassemia '
    ])

    # Now plot
    #fig, ax = plt.subplots(figsize=(14, 6))

    # Remove the edges at the top and right
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)

    #pos = np.arange(sorted_idx.shape[0])

    # First, plot the less important features in grey
    #ax.barh(pos[:-3], feature_importance[sorted_idx][:-3], color='grey', align='center')

    # Then, plot the top 3 important features in red
    #ax.barh(pos[-3:], feature_importance[sorted_idx][-3:], color='red', align='center')

    # Set the yticks to be the names of the features
    #ax.set_yticks(pos)
    #ax.set_yticklabels(column_names[sorted_idx], fontsize=14)

    # label the x-axes as 'Importance'
    #ax.set_xlabel('Importance', fontsize=14)

    # title of the plot
    #ax.set_title('Feature importance', fontsize=16)

    #fig.tight_layout()

    # Save it to a BytesIO object
    #buf = io.BytesIO()
    #plt.savefig(buf, format='png')
    #plt.close(fig)
    #buf.seek(0)

    # Convert the BytesIO object to a base64-encoded string
    #image_base64 = base64.b64encode(buf.read()).decode()

    # Create the HTML for the image
    #html_img = f'<img width="100%" height="300" src="data:image/png;base64,{image_base64}"/>'

    # Export the HTML image as a txt file
   # with open("feature_importance.txt", "w") as f:
#    f.write(html_img)
        

    # Now you can use html_img as a response.
    #st.session_state.history.append({"message": html_img, "is_user": False, "info": "Trained using a random forest model."})
    return ""      
    #return column_names[sorted_idx][-3:], feature_importance[sorted_idx][-3:]


@tool
def get_information_on_patient(feature: str) -> str:
    """Use this function to extract a specific piece of information on the patient that is available in the pandas
    dataframe. This function is only used for extracting a single piece of information, not describing the patient.
    The information on the patient that is available is: ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
      'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal'].
    The input to the function is the column name, and the output is a string explaining the value of that
    column name"""
    
    df = person_data('')
    
    info_person = df.loc[0, feature]
    
    return f"The value of {feature} is {info_person}"

@tool
def get_info_from_wkipedia(item: str) -> str:
    """Use this tool for any questions related to overall medical literature and overall knowledge,
    as well as extracting relevant statistics for diseases. The input for this tool is the object of search,
    such as a disease, and the output is wikipedia information for that disease. The language model uses
    this information to answer questions relevant to the person."""
    
    wiki = WikipediaAPIWrapper()
    return wiki.run(item)

@tool
def counterfactual_CVD_risk(features: str) -> str:
    """Use this for any question related to how the cardiovascular risk would change if any of the observed
    characteristics, such as age, would change. The current columns are ['sex', 'age', 'b_atrial_fibr', 'b_antipsychotic_use', 'b_steroid_treat', 
    'b_erectile_disf', 'b_had_migraine', 'b_rheumatoid_arthritis',
      'b_renal', 'b_mental_illness', 'b_sle', 'hypdbin', 'b_diab_type1',
      'hxdiab', 'bmi', 'ethrisk', 'family_cvd', 'chol_ratio', 'sbp', 'sbps5', 'smallbin', 'town_depr_index']. 
    The function changes the required characteristic to the set value and re-runs the risk prediction.
    The function takes a string in the form tuple as an input which is '(feature, value)', such as '(age, 50)'. 
    The function then returns a string explaining the old and new risk predictions, as well as their difference."""
    
    # Get data
    X = pd.read_csv('./cvd/person_cvd.csv')  
    
    feat, value = ast.literal_eval(features)
    X_count = X.copy()
    X_count[feat] = value
    
    # Get classifier
    qrisk_model = QRisk3Model()

    score_old = qrisk_model.predict(X).values[0][0].round(3)
    score_new = qrisk_model.predict(X_count).values[0][0].round(3)

    diff = score_new - score_old

    return "Old risk: " + str(score_old) + "\nNew risk: " + str(score_new), "Difference: " + str(diff)

@tool
def df_to_string(name: str) -> str:
    """Use this function for any questions about different treatment options for a patient. 
    The function takes as input an empty string and returns a string that contains the information about
    the patient. This information is information on the patient's age, sex, chest pain, and others.
    Based on this information, the language model should suggest possible treatment options specific
    to this individual."""
    df = person_data('')

    # Create a dictionary to map column names to more interpretable strings
    column_map = {
        'age': 'The age of the patient is',
        'sex': 'The sex of the patient is',
        'cp': 'The type of chest pain is',
        'trestbps': 'The resting blood pressure (in mm Hg on admission to the hospital) is',
        'chol': 'The serum cholesterol in mg/dl is',
        'fbs': 'The fasting blood sugar > 120 mg/dl is',
        'restecg': 'The resting electrocardiographic results are',
        'thalch': 'The maximum heart rate achieved is',
        'exang': 'Exercise induced angina is',
        'oldpeak': 'ST depression induced by exercise relative to rest is',
        'slope': 'The slope of the peak exercise ST segment is',
        'ca': 'The number of major vessels (0-3) colored by fluoroscopy is',
        'thal': 'Thalassemia is'
    }

    # Convert the sex and fbs columns to more interpretable strings
    df['sex'] = df['sex'].apply(lambda x: 'male' if x == 1 else 'female')
    df['fbs'] = df['fbs'].apply(lambda x: 'true' if x == 1 else 'false')

    # Convert the DataFrame to a string
    df_string = ', '.join([f'{column_map[col]} {df.iloc[0][col]}' for col in df.columns])

    return df_string

@tool
def calculate_Qrisk_score(name: str) -> str:
    """Use this function to calculate the cardiovascular disease risk for a person / calculate the Qrisk score for a person. The input to the function is an empty string.
    The function returns a string containing information about the Q-risk score of a person."""
    
    qrisk_model = QRisk3Model()
    X = pd.read_csv('./cvd/person_cvd.csv')  
    score = qrisk_model.predict(X).values[0][0].round(3)
            
    return f"The Qrisk Risk Score for this person is {score * 100} % by running the full Qrisk3 model. QRisk3 is the recommended CVD risk score in the UK. All the uploaded variables were used in the prediction."

@tool
def get_nice_guidelines(question: str) -> str:
    """Use this function to get the guidelines from NICE on how to treat a person with cardiovascular disease.
     
    The input to the function is a question, such as "What are the guidelines for a person with a 4% probability of cardiovascular disease?"
    The function returns a string containing the guidelines.
    """
    loader = PyPDFLoader("guidelines/Guidelines.pdf")
    pages = loader.load_and_split()
    
    faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())
    docs = faiss_index.similarity_search(question)
    
    # Loop over docs and put in a list
    info = []
    for doc in docs:
        info.append(doc.page_content)
    full_text = '; NEW PAGE: '.join(info)
    
    # Summarize the information
    template = """Using the following information below, you are asked to provide answers to the question: what are the guidelines for a person with a 13.3% probability of cardiovascular disease?

    {full_text}

    Answer:"""

    prompt = PromptTemplate(template=template, input_variables=["full_text"])
    
    # Summarize with an LLM
    llm = OpenAI()
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    answer = llm_chain.run(full_text)
    
    return answer

@tool
def get_qrisk3_information(question: str) -> str:
    """Use this tool to get information about the QRISK3 method for cardiovascular risk prediction.
     
    The input to the function is a question, such as "Why is corticosteroids included in the QRISK3 prediction model?"
    The function returns a string containing the reasons explaining the answer.
    """
    loader = PyPDFLoader("guidelines/QRISK3 Paper.pdf")
    pages = loader.load_and_split()
    
    faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())
    docs = faiss_index.similarity_search(question)
    
    # Loop over docs and put in a list
    info = []
    for doc in docs:
        info.append(doc.page_content)
    full_text = '; NEW PAGE: '.join(info)
    
    # Summarize the information
    template = """Using the following information below, you are asked to provide answers to the question: Why is corticosteroids included in the QRISK3 prediction model?

    {full_text}

    Answer:"""

    prompt = PromptTemplate(template=template, input_variables=["full_text"])
    
    # Summarize with an LLM
    llm = OpenAI()
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    answer = llm_chain.run(full_text)
    
    return answer

@tool
def explain_predictions_diabetes(item: str) -> str:
    """Use this tool to explain the key factors driving the risk prediction for the diabetes model. This explains the most important features for diabetes.
    The tool takes as input an empty string and returns the top features of the risk prediction model. This method is based on the SHAP package using Shapley Values from Game theory"""

    ft_imp = pd.read_csv("./diabetes/feature_importance_data.csv")
    ft_imp = ft_imp['col_name'][:7].values
    return f"The key factors driving the risk prediction for the diabetes model are: {ft_imp}. This is calculated using the SHAP package using Shapley Values from Game theory."

@tool
def calculate_diabetes_risk(time: str) -> str:
    """Use this tool to calculate the risk of Type II diabetes for the user using Autoprognosis 2.
    The input to the function is a string of time when the diabetes risk should be estimated in years (e.g. "5"). This should be a number in years, such as "5".
    The output to the function is a string explaining a person's diabetes risk score.

    The implementation of the diabetes risk score is provided using the autoprognosis package. 
    """

    # Load the model
    diab_model_loc = './diabetes/model_diabetes_xgb.bkp'
    model = load_from_file(diab_model_loc)

    # Load the person data
    df = pd.read_csv("./diabetes/person_diabetes.csv")

    # Get the prediction timeline in years
    if time == "": time = 5
    else: time = int(time)
    pred_horizon = time * 365

    # Get the prediction
    preds = model.predict(df, [pred_horizon]).iloc[0].iloc[0]

    return f"The diabetes risk score for this person is {np.round(preds * 100, 4)} %"

