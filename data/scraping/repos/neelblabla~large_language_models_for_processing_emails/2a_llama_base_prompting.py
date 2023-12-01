# Import necessary libraries
import pandas as pd
import numpy as np
from langchain.llms import Ollama

# Specify the LLM-model
ollama = Ollama(base_url="http://localhost:11434", model="llama2")

# Import the test data
df = pd.read_csv("llabelled_enron.csv")

# Loop to go through the rows of the test dataset and prompt the LLM
res_list = []

for i, mail in enumerate(df["content"]):
    prompt = mail + ("Read the text until here. Which of the following categories fits best?"
                     "Answer only with the number of the category you think is fitting, nothing else!"
                     "Categories: "
                     "1.1 Company Business, Strategy, etc. (elaborate in Section 3 [Topics]), "
                     "1.2 Purely Personal, "
                     "1.3 Personal but in professional context (e.g., it was good working with you), "
                     "1.4 Logistic Arrangements (meeting scheduling, technical support, etc), "
                     "1.5 Employment arrangements (job seeking, hiring, recommendations, etc), "
                     "1.6 Document editing/checking (collaboration)")

    result = ollama(prompt)

    res_list.append(result)
    
    # Stopping condition
    if i > 200:
        break

# Include nans for the length of thee dataset and merge
res_list_nan = res_list + [np.nan] * (len(df) - len(res_list))
df["llama output"] = res_list_nan

# Save the results
df.to_csv("llama_labels.csv")