
from http.server import BaseHTTPRequestHandler,HTTPServer
from urllib import parse
import json

import cohere 
from cohere.classify import Example
import os
from dotenv import load_dotenv
try:
  from api.util.bot import bot
except:
  from util.bot import bot
load_dotenv()




examples =  [
  Example("I feel like no one loves me", "Self-harm"),  
  Example("I feel meaningless", "Self-harm"),  
  Example("I want to feel pain", "Self-harm"),  
  Example("I want everything to end", "Self-harm"),  
  Example("Why does no one love me?", "Self-harm"),  
  Example("My chest hurts really badly. Please help!", "Medical attention"),  
  Example("My arm is broken", "Medical attention"),
  Example("I have a giant cut on my leg!", "Medical attention"),    
  Example("I feel like I'm going to pass out", "Medical attention"),
  Example("I think I'm getting warts on my genitals. What does that mean", "Symptoms"),    
  Example("I have a slight fever and cough. What do I have", "Symptoms"),    
  Example("I have diarrea and muscle aches. What do you think I have", "Symptoms"),
  Example("I have a small headache and some trouble breathing. What does that mean", "Symptoms")
]
class handler(BaseHTTPRequestHandler):
  def setHeader(s, self):
    self.send_response(200)
    self.send_header("Access-Control-Allow-Origin", "*")
    self.send_header("Access-Control-Allow-Headers", "*")
    self.send_header("Access-Control-Allow-Methods","*")
  def do_OPTIONS(self):
    self.setHeader(self)
    self.end_headers()

  def do_HEAD(self):
    self.setHeader(self)
    self.end_headers()

  def do_GET(self):
    dic = dict(parse.parse_qsl(parse.urlsplit(self.path).query)) # parse the query string

    self.setHeader(self)
    self.send_header('Content-type', 'application/json')
    self.end_headers()

    # if `msg=` is in the query string
    if "msg" in dic:
      message = dic["msg"]
    else:
      message = "Wrong request, I need a msg parameter"

    # create a dictionary to be returned as json
    ret_obj = {'reply':str(message)} 
    self.wfile.write(json.dumps(ret_obj).encode())

    return

  def do_POST(self):
    try:

      content_len = int(self.headers.get('content-length'))
      post_body = self.rfile.read(content_len)
      data = json.loads(post_body)
      # print("Received: ", data)
      self.setHeader(self)
      self.send_header('Content-type', 'application/json') # 'text/plain' for plain text
      self.end_headers()
      # print(data)
      retintent,response= bot(data["message"],data["sender"])
      ret_obj = [{"text":response},{"intent":retintent}]
      self.wfile.write(json.dumps(ret_obj).encode())
      #self.wfile.close()
      return
    except Exception as err:
      print(f"Unexpected {err=}, {type(err)=}")
      ret_obj = [{"text": f"Unexpected {err=}"}]
      self.send_response(200)
      self.send_header("Access-Control-Allow-Origin", "*")
      self.send_header("Access-Control-Allow-Headers", "*")
      self.send_header('Content-type', 'application/json') # 'text/plain' for plain text
      self.end_headers()
      self.wfile.write(json.dumps(ret_obj).encode())
      #self.wfile.close()

## Run the server, for local testing
def main():
    port = 5000
    print('Listening on localhost:%s' % port)
    server = HTTPServer(('', port), handler)
    server.serve_forever()

if __name__ == "__main__":
    main()
