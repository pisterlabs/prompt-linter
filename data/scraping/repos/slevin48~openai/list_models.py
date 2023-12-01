# %%
from openai import OpenAI
import os, toml
secrets = toml.load("../.streamlit/secrets.toml")
os.environ['OPENAI_API_KEY'] = secrets['OPEN_AI_KEY']
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
)
# %%
models = list(client.models.list())
models_name = [model.id for model in models]
# selected_model = models_name[0]
# %%
with open('models.txt', 'w') as f:
  for model in models_name:
    f.write(model + '\n')