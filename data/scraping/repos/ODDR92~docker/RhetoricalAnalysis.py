import pandas as pd
import openai
from dotenv import load_dotenv
import os
import textwrap

# Load your OpenAI API key from an environment variable
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the data
df = pd.read_csv('ceo_interviews.csv')

# Create a new DataFrame with alternating rows merged
df['Title'] = df['Title'].iloc[::2].reset_index(drop=True)
df['Content'] = df['Content'].iloc[1::2].reset_index(drop=True)

# Drop any rows that still have missing data
df = df.dropna()

# Define a function to use GPT-3 for rhetorical analysis
def rhetorical_analysis(text):
    # Chunk the text into smaller pieces of approximately 3000 characters each
    chunks = textwrap.wrap(text, width=3000)

    analysis_results = []
    for chunk in chunks:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k-0613",
            messages=[
                {"role": "system", "content": "This is a rhetorical analysis assistant."},
                {"role": "user", "content": chunk},
            ],
            temperature=0.3,
            max_tokens=13000
        )
        analysis_results.append(response['choices'][0]['message']['content'].strip())
    
    # Combine the analysis results from all chunks
    return " ".join(analysis_results)

# Apply the rhetorical analysis to the 'Content' column
df['Rhetorical_Analysis'] = df['Content'].apply(rhetorical_analysis)

# Print the rhetorical analysis
print(df['Rhetorical_Analysis'])