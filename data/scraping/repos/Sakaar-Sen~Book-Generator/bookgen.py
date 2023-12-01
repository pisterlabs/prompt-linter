from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
import asyncio
import codecs
import markdown
import re
import aspose.words as aw
import pdfkit
import time
import requests
import dotenv
import json
import os

dotenv.load_dotenv()

huggingFaceAPiKey = os.getenv("HUGGING_FACE_APIKEY")

repoID="mistralai/Mistral-7B-Instruct-v0.1"

llmmodel = HuggingFaceHub(repo_id=repoID, model_kwargs={"max_new_tokens": 1000, "temperature": 0.7},huggingfacehub_api_token=huggingFaceAPiKey)

def getChapters(topic_name):
    template = """You are an instructor and given the topic, generate names of the chapters neccessary to teach the topic. Create a maximum of 10 chapters. The last chapter should be "Summary". Do not respond with anything after writing the 10 chapters. All the chapters should be unique.

Response should be list of strings with the following format:
```
1. 
2. 
3. 
4. 
5.
6.
...
```

Topic: {topic_name}\n"""


    prompt = PromptTemplate(template=template, input_variables=["topic_name"])

    chain = LLMChain(llm=llmmodel, prompt=prompt)

    IndexRes = chain.run(topic_name).strip()
    IndexResList = IndexRes.split("\n")
    IndexResList = [x[3:] for x in IndexResList]
    IndexResList = [x.strip() for x in IndexResList]
    IndexResList = [x for x in IndexResList if x != ""]

    return IndexResList



async def async_generate(chain, topic_name, current_chapter, next_chapter):
    resp = await chain.arun({"topic_name": topic_name, "next_chapter": next_chapter, "current_chapter": current_chapter})
    return resp


async def generate_concurrently(topic_name, IndexResList):
    template2 = """You are an instructor and are writing a book about {topic_name}. 

The book content you write should be in markdown format.

The current chapter is {current_chapter}. The next chapter is {next_chapter}.
Generate content for ONLY for this chapter. It should be around 200 words. Always start with a heading like # {current_chapter} 

Content for {current_chapter} in markdown format staring with chapter name: """

    prompt2 = PromptTemplate(template=template2, input_variables=["topic_name", "current_chapter", "next_chapter"])
    ChapterChain = LLMChain(llm=llmmodel, prompt=prompt2)
    tasks = [async_generate(ChapterChain, topic_name, IndexResList[i], IndexResList[i+1] if i+1 < len(IndexResList) else "does not exist. This is the last chapter. Write \"Thanks for reading\" at the end") for i in range(len(IndexResList))]
    R = await asyncio.gather(*tasks)

    return R


async def generate_image_prompt(topic_name):
    template3 = "Generate a suitable image generation prompt for the book cover of {topic_name}. Keep the prompt short and concise. Book cover should be modern and elegant. \nPrompt: "

    prompt3 = PromptTemplate(template=template3, input_variables=["topic_name"])
    ImageChain = LLMChain(llm=llmmodel, prompt=prompt3)
    resp = await ImageChain.arun({"topic_name": topic_name})
    return resp


def generate_image(prompt, topic_name):
    modelID = "stabilityai/stable-diffusion-xl-base-1.0"
    API_URL = "https://api-inference.huggingface.co/models/" + modelID
    headers = {"Authorization": f"Bearer {huggingFaceAPiKey}"}
    data = json.dumps(prompt)
    response = requests.request("POST", API_URL, headers=headers, data=data)
    data = response.content
    path = processTopicName(topic_name) + ".jpg"
    with open(path, 'wb') as f:
        f.write(data)
        print("Saved image to " + path)


def processResponse(R):
    FinalResponseList = R
    FinalResponseList = [x.replace("=", "") for x in FinalResponseList]
    FinalResponseList = [x.replace("-", "") for x in FinalResponseList]
    FinalResponseList = [x.replace("`", "") for x in FinalResponseList]

    for i in FinalResponseList:
        content = i[:10].replace("\n", "").strip()
        if content[0] != "#":
            content = "\n\n# " + content
            FinalResponseList[FinalResponseList.index(i)] = content + i[10:]

    FinalResponseList = [x.replace("  ", " ") for x in FinalResponseList]
    return FinalResponseList


def processTopicName(topic_name):
    pattern = r'^[a-zA-Z0-9_.-]+$'
    topic_name = re.sub(pattern, "", topic_name)
    topic_name = re.sub(r'\t', '_', topic_name)
    return topic_name


# old html formatting fn
def backupSaveHtml(topic_name, FinalResponseList, IndexResList):
    for i in range(len(FinalResponseList)):
        FinalResponseList[i] = markdown.markdown(FinalResponseList[i])

    FinalResponse = ""
    for i in range(len(FinalResponseList)):
        FinalResponse += FinalResponseList[i]

        prefixhtml = f"""
<head>
    <style>
        body {{
            font-family: 'Trebuchet MS', 'Lucida Sans Unicode', 'Lucida Grande', 'Lucida Sans', Arial, sans-serif;
            font-size: x-large;
        }}
    </style>
</head>

<h1 style="text-align: center;">
    {topic_name}
</h1>

<h2 style="text-align: center;">
    Index
</h2>

<ol style="page-break-after: always;">

"""

    for i in range(len(IndexResList)):
        prefixhtml += f"""<li>{IndexResList[i]}</li>\n"""

    prefixhtml += "</ol>\n\n"

    FinalResponse = prefixhtml + FinalResponse
    
    topic_name = processTopicName(topic_name)

    if os.path.exists(f"{topic_name}.html"):
        os.remove(f"{topic_name}.html")

    with open(f"{topic_name}.html", "w", encoding="utf-8") as f:
        f.write(FinalResponse)


def htmlToPdf(topic_name):
    path_wkhtmltopdf = "C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe"
    config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)
    options = {'enable-local-file-access': None}

    topic_name = processTopicName(topic_name)

    if os.path.exists(f"{topic_name}.pdf"):
        os.remove(f"{topic_name}.pdf")

    pdfkit.from_file(f"{topic_name}.html", f"{topic_name}.pdf", configuration=config, options=options)


def escape_equal_sign(text):
    return text.replace("=", "\\=")



def saveHtml(topic_name, FinalResponseList, IndexResList):
    md = markdown.Markdown(extensions=['markdown.extensions.nl2br'])
    FinalResponseList = [md.convert(escape_equal_sign(text)) for text in FinalResponseList]

    body_content = "".join(FinalResponseList)

    imagePath = f"{processTopicName(topic_name)}.jpg"

    prefix_html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{
            font-family: 'Arial', sans-serif; 
            font-size: x-large;
            margin: 0; 
        }}
        pre {{
            white-space: pre-wrap;
            background-color: #f0f0f0;
            padding: 5px; 
            font-family: 'Arial', sans-serif; 
        }}
    </style>
</head>
<body>


<h1 style="text-align: center;">
    {topic_name}
</h1>

<img src="{imagePath}" alt="Front Page Image" style="margin-top: 100px;">


<h2 style="text-align: center; page-break-before: always;">
    Index
</h2>

<ol style="page-break-after: always; ">
"""

    prefix_html += "\n".join([f"<li>{item}</li>" for item in IndexResList])

    prefix_html += "</ol>\n\n"

    final_html = prefix_html + f"<pre>{body_content}</pre>\n</body>\n</html>"

    topic_name = processTopicName(topic_name)
    file_path = f"{topic_name}.html"

    if os.path.exists(file_path):
        os.remove(file_path)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(final_html)

def generateQuestions(topic_name):
    template = """You are an instructor and given the topic, generate MCQ to test the students understanding of the topic. Create a maximum of 5 questions. All the questions should be unique. The difficulty of the questions should rise as the question number increases. 

Format Instructions:

"""

def main(topic_name):
    try:
        IndexResList = getChapters(topic_name)
        print(IndexResList)

        R = asyncio.run(generate_concurrently(topic_name, IndexResList))
        print(R)

        FinalResponseList = processResponse(R)
        print(FinalResponseList)

        prompt = asyncio.run(generate_image_prompt(topic_name))
        print(prompt)

        generate_image(prompt, topic_name)
        print("Image saved")

        saveHtml(topic_name, FinalResponseList, IndexResList)
        print("HTML saved")

        htmlToPdf(topic_name)
        print("PDF saved")

        return True
    except Exception as e:
        print(e)
        return False

if __name__ == "__main__":
    s = time.perf_counter()
    main("Quantum Computing")
    elapsed = time.perf_counter() - s
    print(f"PDF created in {elapsed:0.2f} seconds.")