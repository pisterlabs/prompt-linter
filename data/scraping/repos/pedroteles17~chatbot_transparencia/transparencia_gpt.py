#%%
import os
from tqdm import tqdm
import pandas as pd
from langchain.document_loaders import PyPDFLoader
import re
from openai import OpenAI

#%%
# Load PDFs and split into pages
files_names = os.listdir("decisoes_cgu_pdfs")

pdfs_text = []
for file in tqdm(files_names, total=len(files_names)):
    try:
        loader = PyPDFLoader(f"decisoes_cgu_pdfs/{file}")
        pages = loader.load_and_split()
        pdfs_text += pages
    except Exception as e:
        print(f"Error in file {file}: {e}")

# %%
# Save pages as txt
pdf_text_df = pd.DataFrame(pdfs_text)\
    .rename(columns={0: "page_content", 1: "metadata"})\
    .assign(
        page_content = lambda x: x["page_content"].apply(lambda x: re.sub(r'[\s\n]+', " ", x[1])),
        metadata = lambda x: x["metadata"].apply(lambda x: x[1]),
        data = lambda x: x.apply(lambda x: {"content": x["page_content"], "metadata": x["metadata"]}, axis=1)
    )\
    .drop(columns=["page_content", "metadata"])\
    .to_csv("decisoes_cgu.txt", index=False, header=False)

# %%
client = OpenAI()

# Upload the file to OpenAI
file = client.files.create(
  file=open("decisoes_cgu.txt", "rb"),
  purpose='assistants'
)

# Create an assistant using the file ID
assistant = client.beta.assistants.create(
  instructions="Você é um chatbot útil que responde a perguntas sobre decisões de transparência tomadas pelo governo brasileiro. Use sua base de conhecimento para responder da melhor forma às perguntas.",
  model="gpt-3.5-turbo-1106", # gpt-3.5-turbo-1106 or gpt-4-1106-preview
  tools=[{"type": "retrieval"}],
  file_ids=[file.id]
)

# %%
