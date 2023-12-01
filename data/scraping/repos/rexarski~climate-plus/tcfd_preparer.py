from openai.error import OpenAIError
import configparser
import os
from utils import (
    embed_docs,
    get_answer,
    parse_pdf,
    search_docs,
    text_to_docs,
)
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import glob

config = configparser.ConfigParser()
config.read("config.ini")
os.environ["OPENAI_API_KEY"] = config.get("API", "openai_api_key")
os.environ["OPENAI_MODEL"] = config.get("API", "model_engine")

path = "data/pdf"
pdf_files = glob.glob(os.path.join(path, "*.pdf"))
valid_indices = np.array(
    sorted([int(os.path.splitext(os.path.basename(f))[0]) for f in pdf_files])
)
df_tcfd = pd.read_csv("data/tcfd.csv")
df_tcfd = df_tcfd.loc[df_tcfd.index.isin(valid_indices)]
df_tcfd = df_tcfd.reset_index().rename(columns={"index": "file_id"})
df_tcfd = (
    df_tcfd.assign(disclosure=df_tcfd["Recommended Disclosure"].str.split("; "))
    .explode("disclosure")
    .reset_index(drop=True)
)
df_tcfd["disclosure"] = (
    df_tcfd["disclosure"]
    # .str.replace(r"\s[abcd]\)$", "", regex=True)
    .str.strip()
)
df_tcfd = (
    df_tcfd.drop(columns=["Recommended Disclosure"])
    .drop_duplicates()
    .reset_index(drop=True)
)
df_tcfd = df_tcfd[
    ["file_id", "disclosure", "Company", "Industry", "Region", "Year"]
]
# print(df_tcfd.disclosure.value_counts())
df_tcfd["disclosure"] = df_tcfd["disclosure"].str.replace(
    "All disclosures",
    "Risk Management,Strategy,Metrics and Targets,Governance",
)
df_tcfd = (
    df_tcfd.assign(disclosure=df_tcfd["disclosure"].str.split(","))
    .explode("disclosure")
    .reset_index(drop=True)
)

# dictionary of questions:

questions_dict = {
    "Governance a)": "Does the company describe the board’s or a board committee’s oversight of climate-related risks or opportunities?",
    "Governance b)": "Does the company describe management’s or a management committee’s role in assessing and managing climate-related risks or opportunities?",
    "Strategy a)": "Does the company describe the climate-related risks or opportunities it has identified?",
    "Strategy b)": "Does the company describe the impact of climate-related risks and opportunities on its businesses, strategy, or financial planning?",
    "Strategy c)": "Does the company describe the resilience of its strategy, taking into consideration different climate-related scenarios, including a 2°C or lower scenario?",
    "Risk Management a)": "Does the company describe its processes for identifying and/or assessing climate-related risks?",
    "Risk Management b)": "Does the company describe its processes for managing climate-related risks?",
    "Risk Management c)": "Does the company describe how processes for identifying, assessing, and managing climate-related risks are integrated into overall risk management?",
    "Metrics and Targets a)": "Does the company disclose the metrics it uses to assess climate-related risks or opportunities?",
    "Metrics and Targets b)": "Does the company disclose Scope 1 and Scope 2, and, if appropriate Scope 3 GHG emissions?",
    "Metrics and Targets c)": "Does the company describe the targets it uses to manage climate-related risks or opportunities?",
}

df_tcfd["question"] = df_tcfd["disclosure"].map(questions_dict)

print(df_tcfd)

# prep the dataframe with file_id and disclosure only
fid_list = df_tcfd["file_id"].to_list()
dis_list = df_tcfd["disclosure"].to_list()
que_list = df_tcfd["question"].to_list()
flist = []
dlist = []
qlist = []
answer_list = []
for i in tqdm(range(len(fid_list))):
    file_id = fid_list[i]
    disclosure = dis_list[i]
    question = que_list[i]
    docs = []
    with open(f"data/pdf/{file_id}.pdf", "rb") as file:
        doc = None
        if file is not None:
            if file.name.endswith(".pdf"):
                try:
                    doc = parse_pdf(file)
                except TypeError:
                    print(
                        f"data/pdf/{file_id}.pdf has TypeError! need some investigation"
                    )
                    continue
            else:
                raise ValueError("File type not supported!")
        docs.append(doc)
        text = text_to_docs(docs)
        try:
            index = embed_docs(text)
        except OpenAIError as e:
            print(e._message)
        d = disclosure
        sources = search_docs(index, d)
        try:
            answer = get_answer(sources, d, question)
        except OpenAIError as e:
            print(e._message)
        temp_list = answer["output_text"].split("\n")
        flist.extend([file_id] * len(temp_list))
        dlist.extend([disclosure] * len(temp_list))
        qlist.extend([question] * len(temp_list))
        answer_list.extend(temp_list)
        # print(f"{file_id} {disclosure} is completed.")
        # if len(answer_list) == len(flist) == len(dlist) == len(qlist):
        #     print("All good!")

df = pd.DataFrame({"question": qlist, "text": answer_list, "label": dlist})

# In some rare occasions, the answer could be empty. Remove those rows:
df = df[df["text"] != ""]
# also remove rows with empty questions:
df = df[df["question"] != ""]

df.to_csv("data/tcfd_output.csv", index=False)
