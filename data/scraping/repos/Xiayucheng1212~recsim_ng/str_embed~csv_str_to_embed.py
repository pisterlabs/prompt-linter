# imports
import pandas as pd
import tiktoken
import openai
from openai.embeddings_utils import get_embedding
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

openai.api_key = "your-openai-key"
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  
max_tokens = 8000  
top_n = 20
input_datapath = "./data/csv_rawdata.csv"  
output_datapath = "./data/embeddings.csv"

df = pd.read_csv(input_datapath)
df['category_encoded'] = label_encoder.fit_transform(df['category'])
df = df.head(top_n)
df["combined"] = (
    "Title: " + df.headline.str.strip() + "; Content: " + df.short_description.str.strip()
)
print(df["combined"])
df["embedding"] = df.combined.apply(lambda x: get_embedding(x, engine=embedding_model))

df = df.drop(["date","link","authors","category","headline","short_description","combined"], axis='columns')
df.to_csv(output_datapath)
