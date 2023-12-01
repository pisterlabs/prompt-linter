import gradio as gr
import openai
# import myseckey
from fpdf import FPDF
import requests
import tempfile
from PIL import Image
from io import BytesIO
import re

openai.api_key = "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"

def get_images(prompt):

    #print("------------ ", prompt) )

    response = openai.Image.create(
        prompt=prompt,
        n=5,
        size= "256x256"
    )
    #print (response)

    res =[]
    n=len (response.data)
    for i in range(n):
        res.append(response.data[i].url)
    #print(res)
    image_url=res [0]
    response = requests.get(image_url)

# with open("C:\\Users\\priyansh.varshney\\Desktop\\multiuser chatbot\\image1.jpg", "wb") as file:

# file.write(response.content)

# image_url=res [1]

# response = requests.get(image_url)



# with open("C:\\Users/priyansh. varshney\\Desktop\\multiuser chatbot\\image2.jpg", "wb") as file:

#

#file.write(response.content)

# image_url=res [2]

# response = requests.get(image_url)

# with open("C:\\Users\\priyansh.varshney\\Desktop\\multiuser chatbot\\image3.jpg", "wb") as file: # file.write(response.content)

# image_url= res[3]

# with open("C:\\Users\\priyansh.varshney\\Desktop\\multiuser chatbot\\image4.jpg", "wb") as file:

# file.write(response.content)

# image_url=res [4]

# response = requests.get(image_url)

# with open("C:\\Users/priyansh.varshney\\Desktop\\multiuser chatbot\\image5.jpg", "wb") as file:

# file.write(response.content)

    return res
def get_completion (prompt):
    message = [{'role': 'user', 'content': prompt}]
    response = openai.ChatCompletion.create(
        model = 'gpt-3.5-turbo',
        messages = message

)

    return response.choices[0].message.content

import math

def insert_url(code, urls):

    pattern = r'<img\s+[^>]src\s=\s*["\'] ?([^"\'>]+)["\"]?[^>]*>'
    urls_iterator = iter(urls)
    modified_html = re.sub (pattern, lambda match: '<img src="{}">'.format(next(urls_iterator)), code)
    return modified_html

def fun (topic, language):
    prompt = f"""You are an AI bot create 3 to 4 paragraph on '{topic} by proper headings and formatting."""

    example = """<!DOCTYPE html>

<html>
<head>
<title>Newsletter</title>

<style>

body {

font-family: Arial, sans-serif; background-color: #f5f5f5;

.container {

max-width: 600px;

margin: 0 auto;

padding: 20px;

background-color: #ffffff;

border: 1px solid #dddddd;

}.header {

text-align: center;

margin-bottom: 20px;

}

.content {

margin-bottom: 20px;

}

ير I

.article {

display: flex;

margin-bottom: 20px;

border-bottom: 1px solid #dddddd;

padding-bottom: 20px;

}

article img {

border-radius: 30%;

max-width: 100%;

margin-right: 5%;

margin-left: 5%;

height: auto;

margin-bottom: 10px;

}.article h2 {

font-size: 20px; margin-bottom: 10px;

.article p {

font-size: 14px; line-height: 1.5;

I

.footer {

text-align: center;

font-size: 12px;

color: #777777;

}

</style>

</head>9

<body>

<div class="container">

<div class="header">

<h1>Newsletter</h1>

</div>

<div class="content">

<div class="article">

<img src="image1.jpg" alt="Image 1">

<div>

<h2>Breaking News: Air Pollution Crisis</h2>

an

population

<p>A recent study conducted by environmental experts reveals that air pollution levels have reached all-time high in major cities. The hazardous air quality poses severe health risks to the

and has become a cause for concern.</p>

</div>

</div>

<div class="article">

<div>

<h2>Effects on Marine Life</h2>

<p>Scientists have discovered alarming evidence of pollution's devastating impact on marine life.

The

increasing pollution in oceans and water bodies is causing immense harm to aquatic ecosystems, leading to the loss of various species and disruption of the delicate balance.</p></div>

<img src="image2.jpg" alt="Image 2">

</div>

<div class="article">

<img src="image3.jpg" alt="Image 3">

<div>

<h2>Combatting Pollution: Sustainable Solutions</h2>

<p>In the face of the pollution crisis, communities worldwide are embracing sustainable practices

and

innovations to reduce pollution. From implementing renewable energy sources to promoting

recycling

and waste management, these initiatives are crucial for creating a cleaner and healthier future

for

all.</p>

</div>

</div>

<div class="article">

<div>

<h2>Combatting Pollution: Sustainable Solutions</h2>

<p>In the face of the pollution crisis, communities worldwide are embracing sustainable practices

and

innovations to reduce pollution. From implementing renewable energy sources to promoting

recyclingand waste management, these initiatives are crucial for creating a cleaner and healthier future

for

all.</p>

</div>

<img src="image4.jpg" alt="Image 3">

</div>

<div class="article">

<img src="image5.jpg" alt="Image 3">

<div>

<h2>Combatting Pollution: Sustainable Solutions</h2>

<p>In the face of the pollution crisis, communities worldwide are embracing sustainable practices

and

I innovations to reduce pollution. From implementing renewable energy sources to promoting and waste management, these initiatives are crucial for creating a cleaner and healthier future

recycling

for

all.</p>

</div>

</div>

</div>

<div class="footer">

<p>&copy; 2023 Newsletter. All rights reserved.</p>

</div>

</div></html>"""

    oneshot = f""" Your task is to create a html code for newsletter of the '{topic}' in the format of given' (example) it should not return exactly the same html, the content should be different."""

#oneshot = """Your task is to create a HTML code for newsletter of the '{topic}' in the format of given '{example}', it should not return exactly the same html and after that translate the content to Spanish language."

# oneshot = f""" Your task is to create a html code for newsletter of the '{topic}' in the format of given' {example) in {language}' it should not return exactly the same html, the content should be different."""

#oneshot = f""" Your task is to create a html code for newsletter of the '{topic}' with relatable content, that html code should contains images tag as well but leave that images tag as blank, the code should be well formatted and good looking." www

# oneshot = f"""Generate HTML content for a newsletter on the topic of "{topic}" in '(language). The newsletter should follow the '{example}' format provided in the example below, but the content should be unique and relevant to the chosen topic."

# oneshot = f""" Generate HTML content for a newsletter on the topic of "{topic}" in "(language)". The newsletter should include relevant and unique content related to the chosen topic.

# For example, you can include a welcome message, introduce the topic, provide recent updates or developments, share informative articles or resources, and conclude with a closing message and company information.

# Ensure that the generated HTML code follows proper format such as "{example}""" 

    print (oneshot)
    temp = get_completion (oneshot)
    links = get_images (topic)

    # prompt2 = """Translate the '{temp}' pragraph tag content from english to '{language}"""
    # temp2 = get_completion (prompt2)
    print("#########################################################")
    print("translated text", temp)
    htmlcode=insert_url(temp,links)
    print("After", links)
    print (htmlcode)
    #generate_pdf(temp, links)
    with open("C:\\Users\\DELL\\OneDrive\\Desktop\\News Letter generation\\output.html", "w") as file:
        file.write(htmlcode)
    return htmlcode

demo=gr.Interface(
    fn = fun,
    inputs = ["text"],

    outputs = "text"
)
demo.launch(share=False)