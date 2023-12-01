from dotenv import load_dotenv
import streamlit as st
import json
import pandas as pd
import redis
import os
import openai


st.set_page_config(page_title="Smart Shopping List", page_icon="🛒")


st.title("🛒 Smart Shopping List")

load_dotenv()
r_host = os.environ.get('RD_HOST')
r_port = os.environ.get('RD_PORT')
r_pass = os.environ.get('RD_PASS')


@st.cache_data
def redis_call(host, port, password):

    r = redis.Redis(
        host=host,
        port=port,
        password=password)

    keys = r.keys()
    values = r.mget(keys)

    data = {}

    for key, value in zip(keys, values):
        data[f"{key.decode()}"] = f"{value.decode()}"

    return data


data = redis_call(r_host, r_port, r_pass)


def redis2df(redis_json):
    df = pd.DataFrame()

    for key, value in redis_json.items():
        if key != 'key':
            temp_df = pd.DataFrame.from_dict(
                json.loads(value), orient='index').T
            temp_df.index = [key]
            df = pd.concat([df, temp_df])
        else:
            pass
    return df


df = redis2df(data)


# Load the existing index set from a text file
try:
    with open('assets/grocery_set.txt', 'r') as f:
        index_set = set([line.strip().lower() for line in f])
except FileNotFoundError:
    index_set = set()

# Add new items to the index set
new_items = set(df.index) - index_set
index_set.update(new_items)

# Save the updated index set to a text file
with open('assets/grocery_set.txt', 'w') as f:
    for item in index_set:
        f.write(str(item).lower() + '\n')

# Check for missing items and generate a markdown checklist
missing_items = []
for idx in df.index:
    if idx.lower() not in index_set:
        missing_items.append(idx.lower())

missing_items = list(set(missing_items))


with open("./assets/recipes.txt", "r") as f:
    data = f.read()

data = data.replace("'", "\"").strip()
list_of_json = data.split('\n\n')

recipes = []

for j in list_of_json:
    d = json.loads(j)
    recipes.append(d['recipe_name'])


# @st.cache_data
def list_generator(recipes, df, missing_items):
    response = openai.Completion.create(
        # model="text-ada-001",
        model="text-davinci-003",
        prompt=f"A person likes these recipes in this list {recipes}, he has these items {set(df.index)} in his home. These items {missing_items} were in his home but now finished. What other items he may need to buy the next time he goes to the grocery store? Give me a the items as a markdown list. Do not add too many items. only the one he does not have in his home and the most essential items he might need beside those.",
        temperature=0.7,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response


st.subheader("Items to buy:")


# if missing_items:
#     for item in missing_items:
#         st.markdown(f"- {item.title()}")
# else:
#     st.markdown("No items to buy.")
raw_output = list_generator(recipes, df, missing_items)
# missing_items[0:0] = raw_output['choices'][0]['text']

# list_string = "\n".join([f"- {i}" for i in raw_output['choices'][0]['text']])

# markdown_string = f"### Ingredients:\n\n{list_string}"

st.markdown(raw_output['choices'][0]['text'])

with st.form("my_form", clear_on_submit=True):
    st.write("Add new items")
    item = st.text_input("Item Name")

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
