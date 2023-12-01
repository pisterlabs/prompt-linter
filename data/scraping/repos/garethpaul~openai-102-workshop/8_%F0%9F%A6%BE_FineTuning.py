import streamlit as st
import pandas as pd
import os
st.markdown("# Fine-tuning")

st.markdown(
    "**Demo**: We will build a tool for this demo to create descriptions of imaginary superanimals. In the end, the tool will receive the age and power of the superhero, and it will automatically produce a description of our superanimal.")
st.markdown("# See [Colab File]() to follow along.")
st.markdown("# Step 1. Generate data")
with st.expander("Show Code"):
    st.code("""
import os
import openai
import pandas as pd
import time
import random

l_animal = ['cat', 'dog', 'chicken']
l_power = ['invisibility', 'read in the thoughts', 'turning lead into gold', 'immortality', 'telepathy', 'teleport', 'flight']

f_prompt = "Imagine a complete and detailed description of a {animal} highly fictional character who has the superpower of {power}. Write out the entire description in a maximum of 100 words in great detail:"
f_sub_prompt = "{animal}, {power}"

df = pd.DataFrame()
for animal in l_animal:
    for power in l_power:
        for i in range(3): ## 3 times each
            prompt = f_prompt.format(animal=animal, power=power)
            sub_prompt = f_sub_prompt.format(animal=animal, power=power)
            print(sub_prompt)

            for j in range(10):  # max number of retries
                try:
                    response = openai.Completion.create(
                        model="text-davinci-003",
                        prompt=prompt,
                        temperature=1,
                        max_tokens=500,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0
                    )
                    finish_reason = response['choices'][0]['finish_reason']
                    response_txt = response['choices'][0]['text']
                    break  # successful request, break the retry loop
                except:
                    sleep_time = (2 ** j) + random.random()  # exponential backoff with jitter
                    print(f"Rate limit hit. Retrying in {sleep_time} seconds")
                    time.sleep(sleep_time)
                    continue
                else:
                    raise  # re-throw the exception if it's not a rate limit error

            new_row = {
                'animal': animal,
                'power': power,
                'prompt': prompt,
                'sub_prompt': sub_prompt,
                'response_txt': response_txt,
                'finish_reason': finish_reason
            }
            new_row = pd.DataFrame([new_row])
            df = pd.concat([df, new_row], axis=0, ignore_index=True)

        df.to_csv("out_openai_completion.csv")
    
    """)

    st.code("""# Then we convert out_openai_completion.csv to the format that corresponds to
# key-value pair (prompt -> completion).
prepared_data = df.loc[:,['sub_prompt','response_txt']]
prepared_data.rename(columns={'sub_prompt':'prompt', 'response_txt':'completion'}, inplace=True)
prepared_data.to_csv('prepared_data.csv',index=False)
    """)

    st.code("""
    !openai tools fine_tunes.prepare_data --file prepared_data.csv --quiet
    !mv prepared_data_prepared.jsonl animal_data.jsonl
    !openai api fine_tunes.create --training_file animal_data.jsonl --model davinci --suffix 'SuperAnimals'
    """)
# get public/prepared_data.csv
# get current file path and get "prepared_data.csv"
file_path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(file_path, "prepared_data.csv")
# comma as separator
df = pd.read_csv(file_path, sep=",")
# split the prompt into two colum
df['completion'] = df['completion'].str.strip()

st.write(df)


st.markdown("# Step 2. Prepare the data")
with st.expander("Show Code"):
    st.code("""openai api fine_tunes.create --training_file prepared_data_prepared.jsonl --model davinci --suffix 'SuperHero'""")

# get the file path from the subdirectory "public" e.g. one level up you are in /pages and need to get to ../public
# Get the current file path
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go one level up by getting the parent directory
parent_dir = os.path.dirname(current_dir)

# Join the parent directory with the "public/prepared_data_prepared.jsonl" path
file_path = os.path.join(parent_dir, "public", "prepared_data_prepared.jsonl")

# Print the resulting file path
# open the file and output to st.code
st.markdown("### JSONL Training File")
st.markdown("prepared_data_prepared.jsonl")
with open(file_path, "r") as f:
    st.code(f.read())

st.write("# Step 3. Fine-tune the model")
with st.expander("Show Code"):
    st.code("""openai api fine_tunes.create --training_file prepared_data_prepared.jsonl --model davinci --suffix 'SuperHero'""")

st.write("# Using the Model")
query = st.text_input("Enter your query")
if query:
    import openai
    response = openai.Completion.create(
        model='davinci:ft-twilio:gpjsuperhero-2023-05-30-22-14-10',
        prompt=[query],
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["END"]
    )
    st.write(response)
    st.write(response['choices'][0]['text'])
