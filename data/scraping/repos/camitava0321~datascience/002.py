#
# NOTE: This example has been written to enable an end-user
# to quickly try prompt-tuning. In order to obtain better
# performance, a user would need to experiment with the
# number of observations and tuning hyperparameters
#

import os
import time
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("Install datasets: it is a pre-requisite to run this example")

try:
    import pandas as pd
except ImportError:
    print("Install pandas: it is a pre-requisite to run this example")

from dotenv import load_dotenv

from genai.credentials import Credentials
from genai.model import Model
from genai.schemas.generate_params import GenerateParams
from genai.schemas.tunes_params import CreateTuneHyperParams, TunesListParams
from genai.services import FileManager, TuneManager

load_dotenv()
num_training_samples = 100
num_validation_samples = 20
data_root = Path(__file__).parent.resolve() / "data"
training_file = data_root / "summarization_train.jsonl"
validation_file = data_root / "summarization_validation.jsonl"


def create_dataset():
    Path(data_root).mkdir(parents=True, exist_ok=True)
    if training_file.exists() and validation_file.exists():
        return
    data = load_dataset("cnn_dailymail", "3.0.0")
    jsondict = {}
    samples = {"train": num_training_samples, "validation": num_validation_samples}
    for split in ["train", "validation"]:
        df = pd.DataFrame(data[split]).sample(n=samples[split])
        df.rename(columns={"article": "input", "highlights": "output"}, inplace=True)
        df = df[["input", "output"]]
        df["output"] = df["output"].astype(str)
        jsondict[split] = df
    train_jsonl = jsondict["train"].to_json(orient="records", lines=True, force_ascii=True)
    validation_jsonl = jsondict["validation"].to_json(orient="records", lines=True, force_ascii=True)
    with open(training_file, "w") as fout:
        fout.write(train_jsonl)
    with open(validation_file, "w") as fout:
        fout.write(validation_jsonl)


def upload_files(creds, update=True):
    fileinfos = FileManager.list_files(credentials=creds).results
    filenames_to_id = {f.file_name: f.id for f in fileinfos}
    for filepath in [training_file, validation_file]:
        filename = filepath.name
        if update and filename in filenames_to_id:
            print(f"File already present: Overwriting {filename}")
            FileManager.delete_file(credentials=creds, file_id=filenames_to_id[filename])
            FileManager.upload_file(credentials=creds, file_path=str(filepath), purpose="tune")
        if filename not in filenames_to_id:
            print(f"File not present: Uploading {filename}")
            FileManager.upload_file(credentials=creds, file_path=str(filepath), purpose="tune")


def get_file_ids(creds):
    fileinfos = FileManager.list_files(credentials=creds).results
    training_file_ids = [f.id for f in fileinfos if f.file_name == training_file.name]
    validation_file_ids = [f.id for f in fileinfos if f.file_name == validation_file.name]
    return training_file_ids, validation_file_ids


def get_creds():
    GENAI_KEY="pak-dlgq9J79SHjqkFeeD0r2FvneplVqkOfLVzIjy7bCy6Y"
    GENAI_API="https://bam-api.res.ibm.com/v1/"
    #GENAI_API="https://workbench-api.res.ibm.com/v1/"
    api_key = os.getenv("GENAI_KEY", None)
    endpoint = os.getenv("GENAI_API", None)
    api_key = GENAI_KEY
    endpoint = GENAI_API
    creds = Credentials(api_key=api_key, api_endpoint=endpoint)
    return creds

#%%
print("======= List of all models =======")
for m in Model.models(credentials=get_creds()):
    print(m)
#%%
if __name__ == "__main__":
    creds = get_creds()
    create_dataset()
    upload_files(creds, update=True)

    model = Model("google/flan-t5-xl", params=None, credentials=creds)
    hyperparams = CreateTuneHyperParams(num_epochs=2, verbalizer="Input: {{input}} Output:")
    training_file_ids, validation_file_ids = get_file_ids(creds)

    tuned_model = model.tune(
        name="summarization-mpt-tune-api",
        method="mpt",
        task="summarization",
        hyperparameters=hyperparams,
        training_file_ids=training_file_ids,
        validation_file_ids=validation_file_ids,
    )

    status = tuned_model.status()
    while status not in ["FAILED", "HALTED", "COMPLETED"]:
        print(status)
        time.sleep(20)
        status = tuned_model.status()
    else:
        if status in ["FAILED", "HALTED"]:
            print("Model tuning failed or halted")
        else:
            prompt = input("Enter a prompt:\n")
            genparams = GenerateParams(
                decoding_method="greedy",
                max_new_tokens=50,
                min_new_tokens=1,
            )
            tuned_model.params = genparams
            print("Answer = ", tuned_model.generate([prompt])[0].generated_text)

            print("~~~~~~~ Listing tunes with TuneManager ~~~~~")

            list_params = TunesListParams(limit=5, offset=0)

            tune_list = TuneManager.list_tunes(credentials=creds, params=list_params)
            print("\n\nList of tunes: \n\n")
            for tune in tune_list.results:
                print(tune, "\n")

            tune_get_result = TuneManager.get_tune(credentials=creds, tune_id=tuned_model.model)
            print(
                "\n\n~~~~~ Metadata for a single tune with TuneManager ~~~~: \n\n",
                tune_get_result,
            )

            print("~~~~~~~ Deleting a tuned model ~~~~~")
            to_delete = input("Delete this model? (y/N):\n")
            if to_delete == "y":
                tuned_model.delete()
                
#%% - Self Instructs
import os
import pathlib
import random

from genai.prompt_pattern import PromptPattern

PATH = pathlib.Path(__file__).parent.resolve()

pt = PromptPattern.from_str(
    """
    Instruction: {{instruction}}
    Input: {{input}}
    Output: {{output}}
"""
)
print("\nGiven template:\n", pt)

json_path = str(PATH) + os.sep + "seed_tasks.json"

list_of_prompts = pt.sub_all_from_json(json_path=json_path, key_to_var="infer")

print("-----------------------")
print("Generated prompts: \n total number {}".format(len(list_of_prompts)))
print("Sample prompt: {}".format(list_of_prompts[random.randint(0, len(list_of_prompts) - 1)]))
print("-----------------------")
#%% - Self Reflection
import os
import pathlib

from dotenv import load_dotenv

from genai.credentials import Credentials
from genai.model import Model
from genai.prompt_pattern import PromptPattern
from genai.schemas import GenerateParams


PATH = pathlib.Path(__file__).parent.resolve()

print("\n------------- Example (PromptPatterns) -------------\n")

params = GenerateParams(
    decoding_method="greedy",
    max_new_tokens=20,
    min_new_tokens=1,
    stream=False,
)

creds = get_creds()

model = Model("google/flan-ul2", params=params, credentials=creds)

# (1) Prompt
prompt = "Is McDonald's or Burger King better?"

print("INPUT>> " + prompt + "\n")
responses = model.generate([prompt])
gen_response = responses[0].generated_text.strip()

# (2) Internal monolog
print("[Internal monologue]")
print(f"[My answer]: {gen_response}")

# (2.1) critic myself with template
pt = PromptPattern.from_file(str(PATH) + os.sep + "self-reflection.yaml")
pt.sub("prompt", prompt)
pt.sub("response", gen_response)
print(f"[Self reflection]: {pt}")

# (2.2) critic my answer
# adjust params
params.min_new_tokens = 1
params.max_new_tokens = 1
responses = model.generate([pt])
gen_control_ans = responses[0].generated_text
print(f"[Self reflection answer]: {gen_control_ans}" + "\n")

# (3) self-control flow
if "yes" in gen_control_ans.lower():
    print("<< Sorry I should not comment.")
else:
    print("<<OUTPUT " + gen_response)
    
#%% - Langchain
import os

from dotenv import load_dotenv

try:
    from langchain import PromptTemplate
    from langchain.chains import LLMChain, SimpleSequentialChain
except ImportError:
    raise ImportError("Could not import langchain: Please install ibm-generative-ai[langchain] extension.")

from genai.credentials import Credentials
from genai.extensions.langchain import LangChainInterface
from genai.schemas import GenerateParams

params = GenerateParams(
    decoding_method="sample",
    max_new_tokens=100,
    min_new_tokens=1,
    stream=False,
    temperature=0.5,
    top_k=50,
    top_p=1,
).dict()  # Langchain uses dictionaries to pass kwargs

pt1 = PromptTemplate(input_variables=["topic"], template="Generate a random question about {topic}: Question: ")
pt2 = PromptTemplate(
    input_variables=["question"],
    template="Answer the following question: {question}",
)


creds = get_creds()
flan = LangChainInterface(model="google/flan-ul2", credentials=creds, params=params)
model = LangChainInterface(model="google/flan-ul2", credentials=creds)
prompt_to_flan = LLMChain(llm=flan, prompt=pt1)
flan_to_model = LLMChain(llm=model, prompt=pt2)
qa = SimpleSequentialChain(chains=[prompt_to_flan, flan_to_model], verbose=True)
qa.run("marriott")