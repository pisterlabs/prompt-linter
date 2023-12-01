import os
import cohere
from cohere.responses.classify import Example
import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
import heapq

# Constants
COHERE_API_KEY = os.environ.get('COHERE_API_KEY')


def get_top_disease_with_medicine(description: str) -> list[dict[str, str]]:
    # Initialize Cohere client
    co = cohere.Client(COHERE_API_KEY)

    # Load the medical condition dataset
    df = pd.read_csv("cohere_api/healthHarborModel/datasets/drugs_side_effects_drugs_com.csv",
                     usecols=['medical_condition', 'medical_condition_description', 'drug_name'])

    # Extract drug names and corresponding medical conditions
    df_drugs = df['drug_name']
    df_cond_copy = df['medical_condition']
    dict_des_to_drug = {}

    for idx, row in df_cond_copy.items():
        if row not in dict_des_to_drug:
            dict_des_to_drug[row] = str(df_drugs[idx]).capitalize()

    # Create examples for Cohere API
    examples = []
    x = df[df.columns[:-1]].values
    y = df[df.columns[-1]].values

    over = RandomUnderSampler()
    x, y = over.fit_resample(x, y)
    data = np.hstack((x, np.reshape(y, (-1, 1))))
    transformed_df = pd.DataFrame(data, columns=df.columns)

    for index, row in transformed_df.iterrows():
        examples.append(Example(row["medical_condition_description"], row["medical_condition"]))

    # Perform classification
    inputs = [description]
    response = co.classify(
        inputs=inputs,
        examples=examples,
        # model="embed-multilingual-v2.0" - Model selection comment
    )

    # Extract top disease predictions
    labels_dict = {}
    num_of_suggestions = 3

    for i in range(len(response)):
        for j in response[i].labels:
            label = response[i].labels[j]
            labels_dict[j] = label.confidence

    top = heapq.nlargest(num_of_suggestions, labels_dict.values())
    topnames = []

    for j in range(num_of_suggestions):
        for i in labels_dict.keys():
            if labels_dict[i] == top[j]:
                dictionar = {'rank': j, 'medicine': dict_des_to_drug[i], 'disease': i, 'confidence': labels_dict[i] * 100}
                topnames.append(dictionar)

    return topnames
