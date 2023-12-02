import cohere
from cohere.responses.classify import Example

co = cohere.Client('iB2NBRMvpjLHLWjaUGrDL2kLsgtkCiq24xKFksSx') #This is your trial API key

toxicInputs = ["this game sucks, you suck",  "stop being a dumbass","Let's do this once and for all","This is coming along nicely"  ]
toxicExamples = [
  Example("you are hot trash", "Toxic"),  
  Example("go to hell", "Toxic"),
  Example("get rekt moron", "Toxic"),  
  Example("get a brain and use it", "Toxic"), 
  Example("say what you mean, you jerk.", "Toxic"), 
  Example("Are you really this stupid", "Toxic"), 
  Example("I will honestly kill you", "Toxic"),  
  Example("yo how are you", "Benign"),  
  Example("I'm curious, how did that happen", "Benign"),  
  Example("Try that again", "Benign"),  
  Example("Hello everyone, excited to be here", "Benign"), 
  Example("I think I saw it first", "Benign"),  
  Example("That is an interesting point", "Benign"), 
  Example("I love this", "Benign"), 
  Example("We should try that sometime", "Benign"), 
  Example("You should go for it", "Benign")
]

testToxic = ["kill yourself", "nice try"]
response = co.classify(model='large', inputs=testToxic,  examples=toxicExamples)
# print(response.classifications)



# print('The confidence levels of the labels are: {}'.format(response.classifications))


#takes in a single message as String, returns boolean isMisogynistic,int confidence (% value, eg 95%)
def classifyMessage(message):
    returnList = []
    response = co.classify(model='06151efa-9a0a-42bb-b210-10da4bfe20d4-ft',inputs=[message])
    allInfo = str(response.classifications[0]).split(",")
    isMiso = allInfo[0][allInfo[0].find("\"")+1:]
    isMiso = isMiso[:len(isMiso)-1]
    mConfidence = allInfo[1][allInfo[1].find(":")+2:]
    mConfidence  = int(float(mConfidence)*100)
    returnList.append(isMiso)
    returnList.append(mConfidence)

    response = co.classify(model='large', inputs=[message],  examples=toxicExamples)
    allInfo = str(response.classifications[0]).split(",")
    isToxic = allInfo[0][allInfo[0].find("\"")+1:]
    isToxic = isToxic[:len(isToxic)-1]
    tConfidence = allInfo[1][allInfo[1].find(":")+2:]
    tConfidence  = int(float(tConfidence)*100)
    returnList.append(isToxic)
    returnList.append(tConfidence)
    
    return returnList
 
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get('/get-result')
async def getResult(msg: str):
    return classifyMessage(msg)


# def checkMessageHistory(history):


#test



