import cohere
import pandas as pd
import replicate
import streamlit as st
from cohere.classify import Example

COHERE_KEY = st.secrets["cohere_key"]
co = cohere.Client(f'{COHERE_KEY}')

# using this script as an example: https://github.com/lablab-ai/streamlit-cohere-boilerplate/blob/main/myapp.py

if ('output' not in st.session_state):
    st.session_state['output'] = []

if ('history' not in st.session_state):
    st.session_state['history'] = []

if ('image' not in st.session_state):
    st.session_state['image'] = []

if ('previous_prompts' not in st.session_state):
    st.session_state["previous_prompts"] = []

# function to accept a prompt and classify it?
df = pd.read_csv("lexica_prompts_df.csv")

MODEL_NAME = "da1454fc-2118-4174-b26a-5173f4fc79a6-ft"
replicate_model = replicate.models.get("stability-ai/stable-diffusion")


def preprocess_df(df):
    for k in df.keyword.unique():
        df["prompts_without_keywords"] = df.prompt.apply(lambda x: x.replace(k, ""))
    return df


# def classify_prompts(df, prompts):
#     exs = [Example(p, l) for p, l in zip(df.prompts_without_keywords, df.keyword)]
#     # given a prompt or a list of ones, classifies them, and returns values
#     print(prompts)
#     response = co.classify(inputs=prompts, examples=exs)
#     return response


def classify_prompts(df, prompts):
    response = co.classify(model='40e50ea7-0586-4f53-ab2d-96296048993f-ft',
    inputs=prompts)
    return response

def create_variations(prompt):
    # given a prompt and a keyword, creates variations of the prompts using the keyword
    generated = co.generate(model=MODEL_NAME, prompt=prompt, num_generations=5,
                            temperature=0.9, max_tokens=50)
    return generated


def create_prompt(input_prompt, keyword):
    initial_prompt = f"Rephrase this prompt with more details, in a list:\n {input_prompt} \n"
    end_prompt = "More details:"
    final_prompts = initial_prompt + end_prompt
    return final_prompts


def make_and_grade_variations(df, input_prompt, num_options=3):
    # try to guess intent of the prompt being made
    # implement more generations
    if len(input_prompt) == 0:
        return
    # generate the image
    st.session_state.image = replicate_model.predict(prompt=input_prompt,
                                                     seed=777) + st.session_state.image
    keyword = classify_prompts(df, [input_prompt]).classifications[0].prediction
    generated = create_variations(create_prompt(input_prompt, keyword))
    list_of_gens = []

    for x in range(0, num_options):
        # will create num_options * 5 options
        generated = create_variations(create_prompt(input_prompt, keyword))
        gens_texts = [x.text for x in generated.generations]
        list_of_gens = gens_texts + list_of_gens

    # list_of_gens = [x.text for x in generated.generations]
    labels = classify_prompts(df, list_of_gens)
    # find prompts with highest probability
    print(labels.classifications)
    filtered_prompts = []

    # pull out the classifications
    # pull out only the predictions that have our keyword
  
    for output, prompt in zip(labels.classifications, list_of_gens):
        if output.prediction == keyword:
            filtered_prompts.append([prompt, output.confidence])

    #return_dict = {"prompts": list_of_gens, "probs": probs}
    return_dict = {"prompts": [x[0] for x in filtered_prompts], "probs": [x[1] for x in filtered_prompts]}
    return_df = pd.DataFrame(return_dict)
    return_df.sort_values(by=["probs"], ascending=False).reset_index(drop=True)
    # st.write(return_df)
    st.session_state.output = list(return_df.prompts)[:10]


KNOWN_ART_STYLES = {
    "cottage core": ":mushroom:",
    "cyberpunk": ":sunglasses:",
    "landscapes": ":sunrise_over_mountains:",
    "water colors": ":rainbow:",
    "steampunk": ":steam_locomotive:",
    "architecture": ":house:"
}

st.title(':art: Perfect Prompt!')
st.subheader('Get your prompt perfect before image generation, and save time!')

flavor_text = """
Have you ever used an image generation model like Stable Diffusion, but found that 
using one takes forever? Never fear, Perfect Prompt to the rescue! Perfect Prompt combines classification 
 and generation large language models to help you come up with your... Perfect Prompt.
 
 Currently, Perfect Prompt works with six art styles: cyberpunk, cottage core, landscapes, 
  steampunk, architecture,  and water colors. 
  
 1. Type a prompt, and we match it to an art style! 
 2. Then, the model uses a generator finetuned on over a thousand Stable Diffusion 
 prompts to provide you keyword suggestions for your next prompt.\n
 3. The generations are then filtered again by the classifier, and the top
 suggestions are shown to you!\n
 4. Take a few, throw them into your prompt, and repeat! Enjoy! \n
 
 Finally, please use this tool ethically and follow guidance mentioned in the 
 [Replicate deployment](https://replicate.com/stability-ai/stable-diffusion). Also be aware that currently, any images
 made are logged to Arjun Patel's dashboard in Replicate, and may be viewed there. Thank you! 
 """
with st.expander("Instructions, click here!"):
    st.markdown(flavor_text)

background_info = """
This app was made with :heart: by Arjun Patel, a data scientist with an entrepreneurial spirit and a 
burning desire to make cool stuff with NLP and generative models! You can contact me via 
[Linkedin](https://www.linkedin.com/in/arjunkirtipatel/) or via email at arjunkirtipatel@gmail.com. 

I made this model after seeing how difficult it was to iterate quickly using open source, free image 
generation tools. Using Stable Diffusion on a Gradio/Streamlit app felt so powerful, but due to not
paying for the GPUs, it can take up to a minute of time to get an image back. The other option is to 
pay, which might not be scalable or easy for beginners to do and set up. I wanted to build a solution
to quickly iterating on prompts for image generation, allowing for time and cost efficient image generation!

The data sources include [Lexica](https://lexica.art), 
a fantastic search engine for tons of Stable Diffusion Prompts. I used 
their API to retrieve training data for each category of art style for Perfect Prompt, as well as a prompt dump
posted by the platform for generation training data. Thanks to Lexica for making these freely available!

The classification and generation tasks are done using [Cohere](https://cohere.ai), as this app is a submission to a 
hackathon hosted by [lablab.ai](https://lablab.ai). Thanks to both for the resources to build this cool app!
"""
with st.expander("How was this made?"):
    st.markdown(background_info)

df = preprocess_df(df)
input = st.text_area('Enter your prospective prompt here', height=100, value="")
if input != "":
    st.session_state.previous_prompts = st.session_state.previous_prompts + [input]
    button_click = st.button('Generate Variations', on_click=make_and_grade_variations(df, input_prompt=input))
    st.header("We think you are trying to make art in the following style...")
    predicted_class = classify_prompts(df, [input]).classifications[0].prediction
    st.subheader(predicted_class + KNOWN_ART_STYLES[predicted_class])

if st.session_state.image != []:
    st.image(st.session_state.image[0])

st.subheader("Suggestions will populate below. Try using keywords from them, and add them to your prompt!")
tab1, tab2, tab3, tab4 = st.tabs(["Output", "Generated Prompt History", "Input Prompt History", "Image History"])
with tab1:
    st.write(st.session_state.output)

st.session_state.history = st.session_state.history + st.session_state.output
with tab2:
    # with st.expander("History of output"):
    st.write(st.session_state.history)
st.session_state.previous_prompts = st.session_state.previous_prompts + [input]
with tab3:
    st.write(st.session_state.previous_prompts)

with tab4:
    if st.session_state.image != "":
        st.image(st.session_state.image)
