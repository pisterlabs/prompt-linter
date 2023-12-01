import cohere
co = cohere.Client('dMIKaNIyifhwVK3lOPYRjtjPh6X5RH4ZJuso9vsW')

#Twilio setup
from twilio.rest import Client
account_sid = 'ACac339753b0a113287560d97e558baf64'
auth_token = 'b427b5aedb62d82ac96dc6498c048a21'
client = Client(account_sid, auth_token)

response = co.generate(
  model='small',
  prompt='We are at TAMU Datathon',
  max_tokens=40,
  temperature=0.9,
  k=0,
  p=0.75,
  frequency_penalty=0,
  presence_penalty=0,
  stop_sequences=[],
  return_likelihoods='NONE')

messageBody = response.generations[0].text

message = client.messages.create(body=messageBody,from_='+14452567085',to='+14096738804')


import sched, time
s = sched.scheduler(time.time, time.sleep)
def sendRandomText(sc): 
    response = co.generate(
    model='small',
    prompt='We are at TAMU Datathon',
    max_tokens=40,
    temperature=0.9,
    k=0,
    p=0.75,
    frequency_penalty=0,
    presence_penalty=0,
    stop_sequences=[],
    return_likelihoods='NONE')

    messageBody = response.generations[0].text

    message = client.messages.create(body=messageBody,from_='+14452567085',to='+14096738804')
    sc.enter(300, 1, sendRandomText, (sc,))

s.enter(300, 1, sendRandomText, (s,))
s.run()
