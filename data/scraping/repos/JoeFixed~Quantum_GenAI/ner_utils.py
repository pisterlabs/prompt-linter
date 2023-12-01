import spacy
import nltk
import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from transformers import (
    MarianMTModel, MarianTokenizer,
    BartForConditionalGeneration, BartTokenizer,
    AutoModelForTokenClassification, AutoTokenizer
)
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_extraction_chain
import pandas as pd
from config import Settings

nltk.download('punkt')
nlp = spacy.load('en_core_web_sm')
# Load NER model


def load_ner_model():
    nlp = spacy.load('en_core_web_sm')
    return nlp

# Perform Named Entity Recognition


def ner_spacy(text):
    doc = nlp(text)
    return doc


def load_ner_model():
    NER_MODEL_PATH = Settings().NER_MODEL_PATH
    tokenizer_NER = AutoTokenizer.from_pretrained(NER_MODEL_PATH)
    model_NER = AutoModelForTokenClassification.from_pretrained(NER_MODEL_PATH)
    return model_NER, tokenizer_NER


def extract_named_entities(text, model_NER, tokenizer_NER):
    custom_labels = ["O", "B-job", "I-job", "B-nationality", "B-person", "I-person", "B-location", "B-time", "I-time", "B-event",
                     "I-event", "B-organization", "I-organization", "I-location", "I-nationality", "B-product", "I-product", "B-artwork", "I-artwork"]

    nltk.download('punkt')
    from nltk.tokenize import word_tokenize

    def _extract_ner(text, model, tokenizer, start_token="▁"):
        tokenized_sentence = tokenizer(
            [text], padding=True, truncation=True, return_tensors="pt")
        tokenized_sentences = tokenized_sentence['input_ids'].numpy()

        with torch.no_grad():
            output = model(**tokenized_sentence)

        last_hidden_states = output[0].numpy()
        label_indices = np.argmax(last_hidden_states[0], axis=1)
        tokens = tokenizer.convert_ids_to_tokens(tokenized_sentences[0])
        special_tags = set(tokenizer.special_tokens_map.values())

        grouped_tokens = []
        for token, label_idx in zip(tokens, label_indices):
            if token not in special_tags:
                if not token.startswith(start_token) and len(token.replace(start_token, "").strip()) > 0:
                    grouped_tokens[-1]["token"] += token
                else:
                    grouped_tokens.append(
                        {"token": token, "label": custom_labels[label_idx]})

        # extract entities
        ents = []
        prev_label = "O"
        for token in grouped_tokens:
            label = token["label"].replace("I-", "").replace("B-", "")
            if token["label"] != "O":

                if label != prev_label:
                    ents.append({"token": [token["token"]], "label": label})
                else:
                    ents[-1]["token"].append(token["token"])

            prev_label = label

        # group tokens
        ents = [{"token": "".join(rec["token"]).replace(
            start_token, " ").strip(), "label": rec["label"]} for rec in ents]

        return ents

    if text.strip() != "":
        sample = " ".join(word_tokenize(text.strip()))

    # Create a list of dictionaries to hold the token and label data
    ent_data = []
    ents = _extract_ner(text=sample, model=model_NER,
                        tokenizer=tokenizer_NER, start_token="▁")
    for ent in ents:
        ent_data.append({"Token": ent["token"], "Label": ent["label"]})

    # Convert the list of dictionaries to a pandas DataFrame
    df_graph = pd.DataFrame(ent_data)
    return df_graph

quantum_df = pd.DataFrame()

def extract_entities_from_text(inp: str) -> pd.DataFrame:
    """
    Extracts named entities from a given input text and organizes them into a DataFrame.

    Args:
        inp (str): The input text from which to extract entities.

    Returns:
        pd.DataFrame: A DataFrame containing extracted entities, organized by type and value.
    """
    # Define the schema

    schema = {
        "properties": {
            "Name": {"type": "string"},
            "Nationality": {"type": "string"},
            "Job": {"type": "string"},
            "Country": {"type": "string"},
            "Person": {"type": "string"},
            "Location": {"type": "string"},
            "Time": {"type": "string"},
            "Event": {"type": "string"},
            "Organization": {"type": "string"},
            "Product": {"type": "string"},
            "Artwork": {"type": "string"},
        },
        "required": [],
    }

    # Initialize the ChatOpenAI model
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo",
                     openai_api_key=Settings().GPT_API_KEY)

    # Create the extraction chain
    chain = create_extraction_chain(schema, llm)

    # Run the extraction chain on the input text
    result = chain.run(inp)

    # Create a list to store the filtered entities
    # Filter the result dictionary
    filtered_entity = {key: value for key, value in result.items() if value}

    # Convert the list of filtered entities to a pandas DataFrame
    df = pd.DataFrame([filtered_entity])

    # Integrate the reformatted_extracted_entities function

    def reformatted_extracted_entities(original_dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the original dataframe of entities into a melted format.

        Args:
            original_dataframe (pd.DataFrame): The original dataframe containing entities.

        Returns:
            pd.DataFrame: A restructured DataFrame with 'Entity Type' and 'Entity Value' columns.
        """

        # Use pandas' melt function to restructure the dataframe
        melted_df = original_dataframe.melt(
            var_name='Entity Type', value_name='Entity Value')

        # Drop any rows that have NaN values in 'Entity Value'
        melted_df.dropna(subset=['Entity Value'], inplace=True)

        # Drop duplicate rows based on both 'Entity Type' and 'Entity Value'
        melted_df.drop_duplicates(
            subset=['Entity Type', 'Entity Value'], inplace=True)

        # Reset the index for aesthetics
        melted_df.reset_index(drop=True, inplace=True)
        melted_df.index = np.arange(1, len(melted_df) + 1)

        return melted_df

    # Apply the reformatted_extracted_entities function
    df = reformatted_extracted_entities(df)
    # Adjust the index to start from 1
    df.reset_index(drop=True, inplace=True)
    df.index = df.index + 1
    quantum_df =df
    return df

def prioritize_entities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prioritizes entities based on a predefined hierarchy and removes duplicates.

    Args:
        df (pd.DataFrame): DataFrame containing entities.

    Returns:
        pd.DataFrame: A DataFrame with prioritized entities and duplicates removed.
    """
    # Define priority for 'Entity Type'
    type_priority = {
        "Country": 1,
        "Name": 2,
        "Nationality": 3,
        "Job": 4,
        "Person": 5,
        "Location": 6,
        "Time": 7,
        "Event": 8,
        "Organization": 9,
        "Product": 10,
        "Artwork": 11
    }

    # Sort the dataframe first by 'Entity Value' and then by priority of 'Entity Type'
    df['Type Priority'] = df['Entity Type'].map(type_priority)
    df = df.sort_values(by=['Type Priority', 'Entity Value'])

    # Drop duplicates based on 'Entity Value', and keep only the first (highest-priority) occurrence
    df = df.drop_duplicates(subset='Entity Value', keep='first')

    # Drop the 'Type Priority' column and reset the index
    df = df.drop(columns='Type Priority')
    df = df.reset_index(drop=True)
    return df

##############################################################

#Quantum Code

##########################

# '''import all the necessary libraries'''
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn import svm

# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import word_tokenize


# from qiskit import *
# # from qiskit_machine_learning.kernels import QuantumKernel
# from sklearn.model_selection import train_test_split


# from qiskit.circuit.library import PauliFeature, AerPauliExpectation, PauliExpectation, CircuitSampler, Gradient
# from qiskit.opflow import PauliFeature, AerPauliExpectation, PauliExpectation, CircuitSampler, Gradient
# from qiskit_machine_learning.algorithms import QSVC
# from qiskit.utils import QuantumInstance
# # from qiskit_machine_learning.kernels import QuantumKernel
# from qiskit import Aer
# from qiskit.opflow import CircuitSampler
# from qiskit.utils import QuantumInstance
# import qiskit
# from qiskit.circuit.library import ZFeatureMap
# from qiskit_machine_learning.kernels import FidelityQuantumKernel

# from qiskit_machine_learning.algorithms import QSVC


# from qiskit import Aer
# from qiskit.utils import QuantumInstance
# from qiskit_machine_learning.algorithms import QSVC


# # Download NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# import spacy
# from spacy import displacy
# nlp = spacy.load('en_core_web_sm')

# def reformatted_extracted_entities_quantum(original_dataframe: pd.DataFrame) -> pd.DataFrame:
#     """
#     Transforms the original dataframe of entities into a melted format.

#     Args:
#         original_dataframe (pd.DataFrame): The original dataframe containing entities.

#     Returns:
#         pd.DataFrame: A restructured DataFrame with 'Entity Type' and 'Entity Value' columns.
#     """

#     # Use pandas' melt function to restructure the dataframe
#     melted_df = original_dataframe.melt(
#         var_name='Entity Type', value_name='Entity Value')

#     # Drop any rows that have NaN values in 'Entity Value'
#     melted_df.dropna(subset=['Entity Value'], inplace=True)

#     # Drop duplicate rows based on both 'Entity Type' and 'Entity Value'
#     melted_df.drop_duplicates(
#         subset=['Entity Type', 'Entity Value'], inplace=True)

#     # Reset the index for aesthetics
#     melted_df.reset_index(drop=True, inplace=True)
#     melted_df.index = np.arange(1, len(melted_df) + 1)

#     return melted_df

# def QML_Classification (original_dataframe: pd.DataFrame) -> pd.DataFrame:

#     # Preprocessing
#     stop_words = set(stopwords.words('english'))
#     lemmatizer = WordNetLemmatizer()

#     # Tokenization, cleaning, stopword removal, and lemmatization
#     original_dataframe['processed_text'] = original_dataframe['Entity Value'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word.lower()) for word in word_tokenize(x) if word.isalnum() and word.lower() not in stop_words]))

#     # Vectorization using TF-IDF
#     vectorizer = TfidfVectorizer()
#     X = vectorizer.fit_transform(original_dataframe['processed_text'])
#     Y = original_dataframe['Entity Type']

#     # Splitting into training and test sets
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

#     # number of steps performed during the training procedure
#     tau = 100
#     # regularization parameter
#     C = 1000

#     # Load the dataset
#     feature_dim = X_train.shape[1]

#     # Set up the quantum feature map
#     feature_map = ZFeatureMap(feature_dimension=X_train.shape[1], reps=1)

#     # Transform the dataset into quantum data
#     quantum_instance = QuantumInstance(Aer.get_backend('statevector_simulator'))

#     q_instance = QuantumInstance(Aer.get_backend('statevector_simulator'), shots=1024)
#     circuit_sampler = CircuitSampler(quantum_instance)

#     # Set up the quantum kernel
#     # quantum_kernel = QuantumKernel(feature_map, pauli_expansion=2, quantum_instance=q_instance)
#     qkernel = FidelityQuantumKernel(feature_map=feature_map)

#     # Build the QSVC model
#     qsvc = QSVC(quantum_kernel=qkernel, C=C)

#     # Fit the QSVC model to the training data
#     qsvc.fit( X_train.toarray(), Y_train.values)

#     # Predict on the test data
#     y_pred = qsvc.predict(X_test.toarray())
#     print("*" * 50)
#     print("\n Predicted Labels:", y_pred," \n\n")

#     # Apply the reformatted_extracted_entities_quantum function
#     df = reformatted_extracted_entities_quantum(quantum_df)
#     # Adjust the index to start from 1
#     df.reset_index(drop=True, inplace=True)
#     df.index = df.index + 1


#    # QML_Classification(df)

#     return df

