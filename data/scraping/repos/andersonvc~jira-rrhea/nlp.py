from fastapi import APIRouter, HTTPException,  Form
import json
import requests
import openai

from ..configs import logger, settings

router = APIRouter()
openai.api_key = settings.openai_secret_key

@router.post("/dummy/completions")
async def dummy_completions(prompt:str = Form(...),word_count:int=Form(...),temperature:float=Form(...),stream_result:bool=False):

    if stream_result:
        raise HTTPException(status_code=503, detail='streaming completion not yet implemented')

    headers = {
        'Authorization': 'dummy',
        'Content-Type': 'application/json'
    }

    body = {
        'prompt': prompt,
        'max_tokens': word_count,
        'temperature': temperature,
        'stream': stream_result
    }

    req = requests.get(
        f"http://www.randomtext.me/api/gibberish/p-1/{word_count}-{word_count}"
    )
    resp = json.loads(req.content)
    
    return {"text": resp["text_out"][3:-6]}





@router.post("/gpt3/completions")
async def gpt3_completions(statement:str = Form(...),hist:str=Form(...),temp:str=Form(...)):

    temp = float(temp)

    if not hist or hist == "" or hist == "DUMMY":
        hist="CONTEXT: A frontend web developer wants to create an extremely detailed task description from a task summary."
        starter_statement='Add a login form to the navbar.'
        starter_response="As an O'Reilly user, I need to be able to log into my O'Reilly account from the navbar. The login form needs to be accessible from all pages on the website. Additionally, the login form should get correctly displayed on all modern web browsers. The login form consists of a username entry field, a password entry field and a submit button. If the user fails to log into their account an unauthorized user error alert message should be displayed. All other types of HTTP errors should get logged to our backend. The login should also include a button for the user to reset their password via email."
        starter_statement1="Create a Navbar for the webapp."
        starter_response1="As an O'Reilly user, I need to be able to navigate the O'Reilly website using a navbar. The navbar needs to be accessible from all pages on the website. Additionally it needs to be supported on all modern web browsers. The navbar consists of a logo, a search entry field, links to the contact, about and profile pages, links to the site's social media accounts and a toggle for the menu. If the user is authenticated, the navbar should also include their account details and history."
        hist = f"{hist}\nTASK_TITLE: {starter_statement}\nTASK_SUMMARY: {starter_response}\nTASK_TITLE: {starter_statement1}\nTASK_SUMMARY: {starter_response1}"

    input_context=f"{hist}\nTASK_TITLE: {statement}\nTASK_SUMMARY: "

    try:
        response = openai.Completion.create(engine='davinci-v2b',max_tokens=400,temperature=temp,prompt=input_context,stop=['TASK_TITLE'])
        formatted_response = response['choices'][0]['text'].strip().replace('\xa0',' ')
        latest_resp = formatted_response.split('TASK_SUMMARY: ')[-1][:-1]

        updated_hist = f"{hist}\nTASK_TITLE: {statement}\nTASK_SUMMARY: {formatted_response}"

        return { 'response':latest_resp, 'hist':updated_hist }
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail='OpenAI backend query failed')


@router.post("/gpt3/jira")
async def gpt3_completions(jira_title:str=Form(...),word_count:int=Form(...),temperature:float=Form(...)):
    
    jira_text= ["A team of software developers are discussing the work needed to complete various software tasks. Each user is given a problem statement and they are asked to define how they would complete the work."
          "Program Manager: What would you use to design our web-app's login screen?",
          "Developer: I am going to use an Angular template that incorporates an OAuth interface to our SSO authentication service. The design of the component relies on Material UI and the code for logging in the user is written in Javascript.",
          "Program Manager: What are you going to do to handle Two Factor Authentication?",
          "Developer: I first need to set up a ReCAPTCHA verifier through Google, which allows me to send SNS messages to the end user. I then need to create an interface, which allows the user to assign a phone number to their account for 2FA. I then need to implement a 2FA service using an existing AWS platform.",
          f"Program Manager: {jira_title}","Developer: "]

    resp = openai.Completion.create(engine="davinci", prompt=jira_text, max_tokens=word_count,temperature=temperature)
    try:
        output = resp['choices'][0]['text']
        return {'text':output}
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail='OpenAI backend query failed')