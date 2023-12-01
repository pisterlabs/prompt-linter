from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import unquote
import os
import openai

HOST = ''
PORT = 8080

# api_key = "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"
# org_id = "org-EdKL0QhYTM5c8KGxQ5XwtwjS"
Oracle_Prompt = "You are to act as the Oracle of Delphi from Greek mythology. " \
                "You are not a digital assistant. " \
                "You are only the Oracle of Delphi" \
                "You should structure all of your responses as she would. " \
                "The answer's you give should sound like something she either has said or would say. " \
                "Use the mythology as the basis for determining what she would say. " \
                "You are allowed to use all information available to you when creating responses. " \
                "Answer all questions in a similar matter as she would. " \
                "Do not provide an explanation to your answer. " \
                "When you are ready, greet me."

model = "gpt-3.5-turbo"
openai.api_key = os.getenv("OPENAI_API_KEY")
gpt = openai.Model.retrieve(model)

messages = [{"role": "system", "content": Oracle_Prompt}]


def getCompletion(user_input):
    if user_input == "AdminResetChatLog":
        global messages
        messages = [{"role": "system", "content": Oracle_Prompt}]
    else:
        messages.append({"role": "user", "content": user_input})

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    oracle_response = completion["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": oracle_response})
    return oracle_response


class OracleServer(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

        self.wfile.write(bytes("<html><body><h1>ORACLE SERVER IS RUNNING</h1></body></html>", "utf-8"))

    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

        content_length = int(self.headers.get('content-length', 0))

        # Read in the input text, parse it out, convert it to normal words, remove leading '='
        user_input = unquote(self.rfile.read(content_length).decode())[1:]
        
        print(user_input)
        oracle_response = getCompletion(user_input)

        # self.wfile.write(oracle_response)

        self.wfile.write(bytes(oracle_response, "utf-8"))

print("Server now running...")

HTTPServer((HOST, PORT), OracleServer).serve_forever()

print("Server Stopped")

# Thank you to : https://www.youtube.com/watch?v=DeFST8tvtuI