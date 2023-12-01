import os
import openai

openai.api_key = os.getenv("OPENAI_KEY")
total = 0

recent = []

from fastapi import FastAPI, Response
from typing import Optional
from pydantic import BaseModel, Field

class Complaint(BaseModel):
    complaint: str
    into_void: Optional[bool] = False
    v: Optional[bool] = Field(alias = 'validate', default=True) #"validate" shadows pydantic name

app = FastAPI()

@app.get("/")
async def root():
    r = """<p>Hi I made an API</p>

<h1>About</h1>
<p>This uses my personal free OpenAI tokens and davinci 3 pls don't make too many requests</p>

<p>commiserate before complaining by GETing <a href="commiserate">/commiserate</a></p>

<p>complain: POST to /complain with {"complaint": "X sucks", "into_void": "False", "validate": "True"}</p>

<h1>Sample</h1>
<p> curl -X POST -H "Content-Type: application/json" /

-d '{"complaint":"The guitar is killing my fingers and no one cares, they just want me to keep hurting myself by playing for them", "validate": "True", "into_void": "False"}' /
http://127.0.0.1:8000/complain

</p>

<h1>Helper Func for Generating CURL Requests</h1>
<pre class="tab">
comp = input("What is your complaint: ").replace("'", "").replace("\"", "")
void = input("Do you want other people to be able to commiserate with this (y/n): ")
if "y" in void:
         void = "True"
else:
    void = "False"
val = input("Do you want to be validated by an AI (y/n): ")
if "y" in val:
    val = "True"
else:
    val = "False"
print(f""\"curl -X POST -H "Content-Type: application/json" / -d '{"complaint":"{comp}", "validate": "{val}", "into_void": "False"}' / http://127.0.0.1:8000/complain"\"")
</pre>

"""
    return Response(content=r, media_type="text/html")

#commiserate before you complain
@app.get("/commiserate")
async def get_resp():
    global recent
    if len(recent) > 0:
        return recent[-1]
    else:
        return "Someone has a problem somewhere (this API has not registered anything recently though)"

@app.post("/complain")
async def create_complaint(c: Complaint):
    global recent
    global total
    if not c.into_void:
        if len(recent) > 100:
            recent = recent[80:]
        recent.append(c.complaint)
    if c.v:
        total += 1
        if total > 500:
            return "sorry, don't want to spend too much on this, service is now down because too many people have made requests"
        response = openai.Completion.create(
          model="text-davinci-003",
          prompt=f"{c.complaint} Villainize the problem and make me feel good about myself.",
          temperature=0.7,
          max_tokens=100,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0
        )
        return response["choices"][0]["text"][2:]
