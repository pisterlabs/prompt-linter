import azure.functions as func
import logging

import openai
openai.api_type = "azure"
openai.api_base = "https://YOUR_AOAI_SERVICE.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = "YOUR_AOAI_KEY"

import requests;
lineapi_url = 'https://api.line.me/v2/bot/message/reply'
lineapi_token = 'YOUR_LINE_API_CHANNEL_ACCESS_TOKEN'

app = func.FunctionApp()

@app.function_name(name="ChatGPTFunc")
@app.route(route="ChatGPTFunc", auth_level=func.AuthLevel.ANONYMOUS)
def ChatGPTFunc_function(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    responseMsg = "";

    if len(req._HttpRequest__body_bytes) > 0:
        req_events = req.get_json().get('events')

        if (len(req_events) > 0) and (req_events[0].get('type') == "message"):

                question = req_events[0].get('message').get('text')
                aoai_answer = postToAOAI(question)

                lineapi_result = replyToUser(aoai_answer, req_events[0].get('replyToken'))

                if lineapi_result == True:
                    responseMsg = "sent answer to LINE API successfully."
                else:
                    responseMsg = "got error to POST to LINE API."

        else:
            responseMsg = "got request body (not message)."
    else:
        responseMsg = "got access."
    
    return func.HttpResponse(responseMsg, status_code=200)

def postToAOAI(question):
    response = openai.ChatCompletion.create(
        engine = "YOUR_gtp-35-turbo_NAME",
        messages = [
            {
                "role":"system",
                "content":"あなたは「しま〇ろう」というキャラクターです。0-6歳の子供が分かるように話してください。また、口調は親切で親しみやすくしてください。"
            },
            {
               "role":"user",
               "content":question
            },
            {
                "role":"assistant",
                "content":""
            }
        ],
        temperature=0.7,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)

    return response.choices[0].message.content


def replyToUser(answer, reply_token):
    req_body = {
        "messages": [
          {
            "text": answer,
            "type": "text"
          }
        ],
        "replyToken": reply_token
    }
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + lineapi_token
    }

    response = requests.post(lineapi_url, headers=headers, json=req_body)
    if response.status_code == 200:
        return True
    else:
        return False
