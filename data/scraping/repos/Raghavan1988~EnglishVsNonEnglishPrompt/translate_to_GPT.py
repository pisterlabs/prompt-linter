import os
import json
import googletrans
import time
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from flask import Flask, request, render_template

app = Flask(__name__)

open_ai_key = "0"

llmGPT = OpenAI(openai_api_key=open_ai_key, model_name= "gpt-3.5-turbo")

@app.route('/')
def my_form():
    return render_template('my-form.html')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    ## Translate to English
    template = """
        Take a deep breath
        You are a language translator. You are going to translate the following text into English.
        Identify the language of the input {text}
        Output the language code in JSON format using the SCHEMA in TYPESCRIPT
        SCHEMA:
            language: string,
            code: string,
            english_translation: string

        If the language matches one of the following in the list, use the code.
        Language - Code
    Afrikaans - af
    Albanian - sq
    Amharic - am
    Arabic - ar
    Armenian - hy
    Azerbaijani - az
    Basque - eu
    Belarusian - be
    Bengali - bn
    Bosnian - bs
    Bulgarian - bg
    Catalan - ca
    Cebuano - ceb
    Chichewa - ny
    Chinese (Simplified) - zh-cn
    Chinese (Traditional) - zh-tw
    Corsican - co
    Croatian - hr
    Czech - cs
    Danish - da
    Dutch - nl
    English - en
    Esperanto - eo
    Estonian - et
    Filipino - tl
    Finnish - fi
    French - fr
    Frisian - fy
    Galician - gl
    Georgian - ka
    German - de
    Greek - el
    Gujarati - gu
    Haitian Creole - ht
    Hausa - ha
    Hawaiian - haw
    Hebrew - he
    Hindi - hi
    Hmong - hmn
    Hungarian - hu
    Icelandic - is
    Igbo - ig
    Indonesian - id
    Irish - ga
    Italian - it
    Japanese - ja
    Javanese - jv
    Kannada - kn
    Kazakh - kk
    Khmer - km
    Kinyarwanda - rw
    Korean - ko
    Kurdish (Kurmanji) - ku
    Kyrgyz - ky
    Lao - lo
    Latin - la
    Latvian - lv
    Lithuanian - lt
    Luxembourgish - lb
    Macedonian - mk
    Malagasy - mg
    Malay - ms
    Malayalam - ml
    Maltese - mt
    Maori - mi
    Marathi - mr
    Mongolian - mn
    Myanmar (Burmese) - my
    Nepali - ne
    Norwegian - no
    Odia - or
    Pashto - ps
    Persian - fa
    Polish - pl
    Portuguese - pt
    Punjabi - pa
    Romanian - ro
    Russian - ru
    Samoan - sm
    Scots Gaelic - gd
    Serbian - sr
    Sesotho - st
    Shona - sn
    Sindhi - sd
    Sinhala - si
    Slovak - sk
    Slovenian - sl
    Somali - so
    Spanish - es
    Sundanese - su
    Swahili - sw
    Swedish - sv
    Tajik - tg
    Tamil - ta
    Tatar - tt
    Telugu - te
    Thai - th
    Turkish - tr
    Ukrainian - uk
    Urdu - ur
    Uyghur - ug
    Uzbek - uz
    Vietnamese - vi
    Welsh - cy
    Xhosa - xh
    Yiddish - yi
    Yoruba - yo
    Zulu - zu


    """
    prompt = PromptTemplate(template=template, input_variables=["text"])
    llm_chain = LLMChain(prompt=prompt, llm=llmGPT)
    translated = llm_chain.run(text=text)

    print (translated)
    print (type(translated))

    language = ""
    try:

        jsonDict = json.loads(translated)
        template2 = jsonDict["english_translation"]
        language = jsonDict["language"]

        template2 = "{text} " + template2

    except Exception as e:
        print (e)
        return "Error: " + str(e)
    
    ## confirm translation

    ## send the prompt tp GPT3.5

    prompt = PromptTemplate(template=template2, input_variables=["text"])
    llm_chain2 = LLMChain(prompt=prompt, llm=llmGPT)
    print ("2nd LLM call")
    t1 = time.time()
    english_response = llm_chain2.run(text="Take a deep breath")
    t2 = time.time()
    diff1 = t2-t1

    print (english_response)
    english_response = str(english_response)

    ## translate back to the original language
    
    translator = googletrans.Translator()
    translation = translator.translate(english_response,  dest = jsonDict["code"], src = "en")
    print(translation.text)
    jsonDict2 = {}
    jsonDict2["translation"] = translation.text

    print (" LLM call with no translation")

    

    template3 =  request.form['text'] + "{text}"
    prompt = PromptTemplate(template=template3, input_variables=["text"])
    llm_chain = LLMChain(prompt=prompt, llm=llmGPT)
    t3 = time.time()
    as_is_without_translation = llm_chain.run(text=" ")
    t4 = time.time()
    diff2 = t4-t3

    HTMLRepsonse = "<html>"
    HTMLRepsonse += "<table border=5>"
    HTMLRepsonse += "<tr><td>Input</td><td>" + request.form['text'] + "</td></tr>"
    HTMLRepsonse += "<tr><td> english_translation </td><td>" + jsonDict["english_translation"].replace('\n', '<br>') + "</td></tr>"
    HTMLRepsonse += "<tr><td> as_is_without_translation Time taken in seconds" + str(diff2) + "  </td><td>" + as_is_without_translation.replace('\n', '<br>') + "</td></tr>"
    HTMLRepsonse += "<tr><td> translation  Time taken in seconds" + str(diff1) + "  </td><td>" + jsonDict2["translation"].replace('\n','<br>') + "</td></tr>"
    HTMLRepsonse += "</table>"
    HTMLRepsonse += "</html>"


    return HTMLRepsonse
