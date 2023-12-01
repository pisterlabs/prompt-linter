import openai
import streamlit as stream

stream.set_page_config(page_title="Items Classification")
stream.title("Item Classifier Model")

prompt = """
Facebook: Social media, Technology
LinkedIn: Social media, Technology, Enterprise, Careers
Uber: Transportation, Technology, Marketplace
Unilever: Conglomerate, Consumer Goods
Mcdonalds: Food, Fast Food, Logistics, Restaurants
"""

# Get your API key from "https://beta.openai.com/account/api-keys"
openai.api_key = "API KEY"

def gpt3(prompt, engine = 'davinci', temperature = 0, top_p = 1.0, frequency_penalty = 0.0, presence_penalty = 0.0, stop_seq = ["\n"]):
    response = openai.Completion.create(
        prompt = prompt,
        engine = engine,
        max_tokens = 64,
        temperature = temperature,
        top_p = top_p,
        frequency_penalty = frequency_penalty,
        presence_penalty = presence_penalty,
        stop = stop_seq,
    )
    print(response)
    result = response.choices[0]['text']
    print(result)
    return result

try:
    form_application = stream.form(key = 'form')
    command = form_application.text_input("Enter Item "+":")
    submit = form_application.form_submit_button('Submit')

    if submit:
        stream.header("**Results**")
        prompt += command
        result = gpt3(prompt)
        stream.header(result)
except Exception as exp:
    stream.success(f'Something Went Wrong!😁 {exp}')

footer="""
<style>
a:link,
a:visited{
    color: blue;
    background-color: transparent;
    text-decoration: underline;
}

a:hover,
a:active {
    color: red;
    background-color: transparent;
    text-decoration: underline;
}

.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: white;
    color: black;
    text-align: center;
}
</style>

<div class="footer">
<p>Developed with ❤ by <a style='display: block; text-align: center;' href="https://github.com/chirag-goel360" target="_blank"> Chirag Goel </a></p>
</div>
"""
stream.markdown(footer,unsafe_allow_html = True)