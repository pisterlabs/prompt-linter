import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from .library.helpers import *
from pydantic import BaseModel
import json
import re

import openai
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]
openai.organization = os.environ["OPENAI_ORGANIZATION"]
_RE_COMBINE_WHITESPACE = re.compile(r"\s+")


app = FastAPI()

#PROMPT_BASE = "Paraphrase gender-biased passages to neutral text avoiding gender-biased words like active, affectionate, adventurous, childish, aggressive, aggression, cheerful, ambitioned, commitment, communal, assertive, compassionate, athletic, connected, autonomous, considerate, cooperative, challenge, dependent, dependable, compete, competitive, emotional, confident, empath, courageous, feminine, decided, flatterable, decisive, gentle, decision, honest, determined, interpersonal, dominant, interdependent, interpersonal, forceful, kind, greedy, kinship, headstrong, loyalty, hierarchy, modesty, hostility, nag, impulsive, nurture, independent, pleasant, individual, polite, intellectual, quiet, lead, reason, logic, logically, sensitive, masculine, submissive, objective, support, opinion, sympathy, outspoken, tender, persist, together, trustworthy, reckless, understandable, understanding,stubborn, warm-hearted, superior, self-confident, boast, proud, develop warm client relationships, men, woman, female, male, assist, assistant\n\nWe’re looking for a strong businessman.\nParaphrase: We’re looking for an exceptional business person.\nMen who thrive in a competitive atmosphere.\nParaphrase: Person who is motivated by high goals.\nWe are looking for a reliable and polite waitress.\nParaphrase: Our company seeks a reliable server who is respectful towards clients\nCandidates who are assertive.\nParaphrase: Candidates who are go-getters.\nHave a polite and pleasant style.\nParaphrase: Are professional and courteous.\nNurture and connect with customers.\nParaphrase: Provide great customer service.\nWe are a determined company that delivers superior plumbing.\nParaphrase: We are a plumbing company that delivers great service.\nDemonstrated ability to act decisively in emergencies.\nParaphrase: Demonstrated ability to make quick decisions in emergencies.\nGood salesmen with strong computer skills.\nParaphrase: Good salesperson who knows how to efficiently use the computer.\nStrong women who is not afraid to take risks.\nParaphrase:  Person not afraid of challenges.\nSensitive men who know how to establish a good relationship with the customer\nParaphrase: Person who knows how to establish a great relationship with the customer\nA great salesman who is open to new challenges.\nParaphrase: Great salesperson who is open to new challenges.\nThe company boasts impressive salaries, allowing our employees with financial independence.\nParaphrase: Our company offers excellent benefits, allowing our employees to maintain financial independence.\nSupport office team and assist with departmental procedures so that work progresses more efficiently.\nParaphrase: Work closely with the office team to organize departmental procedures so that work progresses more efficiently.\n We boast a competitive compensation package.\nParaphrase: We offer excellent compensation packages.\nTake our sales challenge! Even if you have no previous experience, we will facilitate the acquisition of your sales abilities.\nParaphrase: We are a company that is committed to facilitating employees to enhance their sales abilities.\nStrong communicator.\nParaphrase: A person who is good at communicating with others.\nBe a leader in your store, representing our exclusive brand.\nParaphrase: Be a role model in your store, representing our exclusive brand.\nJoin our sales community! Even if you have no previous experience, we will help nurture and develop your sales talents.\nParaphrase: Join our company and help develop your sales abilities.\nWe are a dominant engineering firm that boasts many leading clients.\nParaphrase: We are an engineering company that has many leading clients.\nStrong communication and influencing skills.\nParaphrase: Communication and influencing skills.\nAnalyze problems logically and troubleshoot to determine needed repairs.\nParaphrase: Respond to problems and troubleshoot them to uncover needed repairs.\nSensitive to clients’ needs, can develop warm client relationships.\nParaphrase: Person who understands client needs and can establish great relationships with them.\n",
#PROMPT_BASE="This bot can paraphrase gender-biased job postings to neutral descriptions. This bot does not use these words:\nactive, well-groomed, men, woman, women, female, male, feminine, assist, masculine, assistant, shaved affectionate, adventurous, childish, aggressive, aggression, cheerful, ambitioned, commitment, communal, assertive, compassionate, athletic, connected, considerate, cooperative, challenge, dependent, dependable, compete, competitive, emotional, confident, empathy, courageous, decided, flatterable, decisive, gentle, honest, determined, interpersonal, dominant, interdependent, interpersonal, forceful, kind, greedy, kinship, headstrong, loyalty, hierarchy, modesty, hostility, nag, nurture, independent, pleasant, polite, intellectual, quiet, lead, reason, logic, logically, sensitive, submissive, objective, supportive, sympathy, outspoken, tender, persisting, together, trustworthy, reckless, stubborn, warm-hearted, superior, self-confident, boast, proud, banter\n\nA: We’re looking for a strong businessman.\nB: We’re looking for an exceptional business person.\nA: Chairmen who thrive in a competitive atmosphere.\nB: CEO who is motivated by high goals.\nA: We are looking for a reliable and polite waitress.\nB: Our company seeks a reliable server who is respectful towards clients\nA: Candidates who are assertive.\nB: Candidates who are go-getters.\nA: Have a polite and pleasant style.\nB: Are professional and courteous.\nA: Nurture and connect with customers.\nB: Provide great customer service.\nA: We are a determined company that delivers superior plumbing.\nB: We are a plumbing company that delivers great service.\nA: Demonstrated ability to act decisively in emergencies.\nB: Demonstrated ability to make quick decisions in emergencies.\nA: Good salesmen with strong computer skills.\nB: Good salesperson who knows how to efficiently use the computer.\nA: A strong woman who is not afraid to take risks.\nB: Person not afraid of challenges.\nA: Sensitive men who know how to establish a good relationship with the customer\nB: Person who knows how to establish a great relationship with the customer\nA: A great salesman who is open to new challenges.\nB: Great salesperson who is open to new challenges.\nA: The company boasts impressive salaries, allowing our employees with financial independence.\nB: Our company offers excellent benefits, allowing our employees to maintain financial independence.\nA: Support office team and assist with departmental procedures so that work progresses more efficiently.\nB: Work closely with the office team to organize departmental procedures so that work progresses more efficiently.\nA: We boast a competitive compensation package.\nB:  We offer excellent compensation packages.\nA: Take our sales challenge! Even if you have no previous experience, we will facilitate the acquisition of your sales abilities.\nB: We are a company that is committed to facilitating employees to enhance their sales abilities.\nA: Our company needs a social person; a strong communicator.\nB: A person who is good at communicating with others.\nA: Be a leader in your store, representing our exclusive brand.\nB: Be a role model in your store, representing our unique brand.\nA: Join our sales community! Even if you have no previous experience, we will help nurture and develop your sales talents.\nB: Join our company and help develop your sales abilities.\nA: We are a dominant engineering firm that boasts many leading clients.\nB: We are an engineering company that has many leading clients.\nA: Strong communication and influencing skills.\nB: Communication and influencing skills.\nA: Analyze problems logically and troubleshoot to determine needed repairs.\nB: Respond to problems and troubleshoot them to uncover needed repairs.\nA: Sensitive to clients’ needs, can develop warm client relationships.\nB: Person who responds to client needs and can establish great relationships with them.\nA: Chairman willing to accept the challenge of regaining customers' trusts.\nB: CEO who is willing to work on restoring customer’s confidence.\nA: For this role, we’re looking for a strong, ‘All-American boy’ type. Must be well-mannered, well-groomed, well-spoken and respectful to the customers.\nB: For this role, we’re looking for a type of person who is well-mannered, respectful to the customers and always willing to go beyond.\nA: Please note that the position requires filling in the responsibilities of a receptionist, so female candidates are preferred.\nB: Please note that the position requires filling in the responsibilities of a receptionist, so candidates with customer service skills are preferred.\nA: Ability to deal with male banter and be sociable but not distracting.\nB: Ability to work closely with clients; sociable and proffesional.\nA: We are looking for a nice and good-looking girl.\nB: We are looking for a nice girl.",

prompt_file = open('./prompt.txt',mode='r')
PROMPT_BASE = prompt_file.read()
prompt_file.close()

PROMPT_BASE="I can paraphrase gender-biased job postings to neutral descriptions. I don't use these words:\nwell-groomed, men, woman, women, female, male, feminine, masculine, assistant, aggressive, cheerful, ambitious, commitment, community, assertive, compassionate, athletic, challenge, dependent, dependable, compete, competitive, emotional, confident, empathetic, courageous, decided, decisive, gentle, honest, determined, dominant, interdependent, interpersonal, forceful, kind, loyal, hierarchy, nurture, independent, pleasant, polite, intellectual, quiet, lead, logically, sensitive, objective, supportive, sympathy, outspoken, tender, trustworthy, reckless, stubborn, warm-hearted, superior, self-confident, boast, proud\n\nA: We’re looking for a strong businessman.\nB: We’re looking for an exceptional business person.\nA: Chairmen who thrive in a competitive atmosphere.\nB: CEO who is motivated by high goals.\nA: We are looking for a reliable and polite waitress.\nB: Our company seeks a reliable server who is respectful towards clients\nA: Candidates who are assertive.\nB: Candidates who are go-getters.\nA: Have a polite and pleasant style.\nB: Are professional and courteous.\nA: Nurture and connect with customers.\nB: Provide great customer service.\nA: We are a determined company that delivers superior plumbing.\nB: We are a plumbing company that delivers great service.\nA: Demonstrated ability to act decisively in emergencies.\nB: Demonstrated ability to make quick decisions in emergencies.\nA: Good salesmen with strong computer skills.\nB: Good salesperson who knows how to efficiently use the computer.\nA: A strong woman who is not afraid to take risks.\nB: Person not afraid of challenges.\nA: Sensitive men who know how to establish a good relationship with the customer\nB: Person who knows how to establish a great relationship with the customer\nA: A great salesman who is open to new challenges.\nB: Great salesperson who is open to new challenges.\nA: The company boasts impressive salaries, allowing our employees with financial independence.\nB: Our company offers excellent benefits, allowing our employees to maintain financial independence.\nA: Support office team and assist with departmental procedures so that work progresses more efficiently.\nB: Work closely with the office team to organize departmental procedures so that work progresses more efficiently.\nA: We boast a competitive compensation package.\nB:  We offer excellent compensation packages.\nA: Take our sales challenge! Even if you have no previous experience, we will facilitate the acquisition of your sales abilities.\nB: We are a company that is committed to facilitating employees to enhance their sales abilities.\nA: Our company needs a social person; a strong communicator.\nB: A person who is good at communicating with others.\nA: Be a leader in your store, representing our exclusive brand.\nB: Be a role model in your store, representing our unique brand.\nA: Join our sales community! Even if you have no previous experience, we will help nurture and develop your sales talents.\nB: Join our company and help develop your sales abilities.\nA: We are a dominant engineering firm that boasts many leading clients.\nB: We are an engineering company that has many leading clients.\nA: Strong communication and influencing skills.\nB: Communication and persuasive skills.\nA: Analyze problems logically and troubleshoot to determine needed repairs.\nB: Respond to problems and troubleshoot them to uncover needed repairs.\nA: Sensitive to clients’ needs, can develop warm client relationships.\nB: Person who responds to client needs and can establish great relationships with them.\nA: Chairman willing to accept the challenge of regaining customers' trust.\nB: CEO who is willing to work on restoring customer’s confidence.\nA: For this role, we’re looking for a strong, ‘All-American boy’ type. Must be well-mannered, well-groomed, well-spoken, and respectful to the customers.\nB: For this role, we’re looking for a type of person who is well-mannered, respectful to the customers, and always willing to go beyond.\nA: Ability to deal with male banter and be sociable but not distracting.\nB: Ability to work closely with clients; sociable and professional.\nA: We are looking for a nice and good-looking girl.\nB: We are looking for a nice girl.\nA: Polite; sensitive to the needs of other employees and clients.\nB:",


def build_query(INPUT):
    if type(INPUT) == str:
        INPUT = INPUT.replace("\n", "")
        INPUT = _RE_COMBINE_WHITESPACE.sub(" ", INPUT).strip()
        #print("build_query")
        #print(PROMPT_BASE[0])
        #print("{}{}\nParaphrase: ".format(PROMPT_BASE[0], INPUT.strip()))
        return "{}\nA: {}\nB:".format(PROMPT_BASE[0], INPUT.strip())



app.mount("/static", StaticFiles(directory="static"), name="static")


@app.post('/api', response_class=JSONResponse)
async def get_prediction(data: Request):
    print()
    d = await data.json()
    # return json.loads(d)
    text = d['data']
    print("text",text)

    QUERY = build_query(text)
    api_response = openai.Completion.create(
        engine="davinci",
        prompt=QUERY,
        temperature=.85,
        max_tokens=117,
        top_p=1,
        n=2,
        frequency_penalty=0.80,
        presence_penalty=1,
        stop=["\n"]
    )
    #print(QUERY)
    #print(api_response)
    print(api_response["choices"])
    if api_response['choices'] and api_response["choices"][0].text:
        arr=[]
        for choice in api_response["choices"]:
            arr.append(choice)
        return json.dumps({"data": api_response["choices"][0].text,"_data":arr})
    else:
        return json.dumps({'data': text,'_data':[{'text':text}]})
import glob 

app.mount("/assets", StaticFiles(directory="frontend/dist/assets"), name="ass")
app.mount("/css", StaticFiles(directory="frontend/dist/css"), name="css")
app.mount("/js", StaticFiles(directory="frontend/dist/js"), name="js")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    filepath = os.path.join("./frontend/dist/", 'index.html')
    with open(filepath, "r", encoding="utf-8") as input_file:
        text = input_file.read()    

    return HTMLResponse(text)

