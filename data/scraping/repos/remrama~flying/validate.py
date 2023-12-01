
import openai
import pandas as pd

FINE_TUNED_MODEL = ""
df = pd.read_table("dreams.tsv")

df["raw"] = df["raw"].str.strip().add("\n\n###\n\n")

# openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.Model.list()

## Add the ending token as a stop sequence during inference.
def predict(prompt):
    return openai.Completion.create(
        model=FINE_TUNED_MODEL,
        prompt=prompt,
        # max_tokens=2049-8,  # model-specific limit minus current prompt length? defaults to 16
        # see 
        temperature=0,
        # logprobs=None,
        suffix=,
        stop=" END",
    )
