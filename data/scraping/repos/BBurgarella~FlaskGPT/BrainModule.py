import os
import openai
import requests
import json

"""
Copyright (c) 2023 Boris Burgarella

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
def opeanai_api_request(model="gpt-3.5-turbo",
                        messages=None,
                        temperature=1,
                        top_p=1,
                        n=1,
                        presence_penalty=0,
                        frequency_penalty=0):
    
    url = "https://api.openai.com/v1/chat/completions"
    api_key = os.getenv("OPENAI_API_KEY")

    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {api_key}",
    }

    data = {
        "model": model,
        "messages": messages,
        "temperature":temperature,
        "top_p":top_p,
        "n":n,
        "presence_penalty":presence_penalty,
        "frequency_penalty":frequency_penalty
    }

    return requests.post(url, headers=headers, json=data).json()

def load_agent(file):
    with open(file, 'r') as f:
    # Load the contents of the file into a dictionary
        data = json.load(f)
    return data

class chatGPT():

    def __init__(self, agent=None):
        # Get the API key
        openai.api_key = os.getenv("OPENAI_API_KEY")
        # Load the system prompt from a file
        if agent is None:
            self.agent_data = load_agent("agents/Generic.json")
        else:
            self.agent_data = load_agent(f"agents/{agent}.json")
        
    def update_agent(self, filename):
        self.agent_data = load_agent(filename)

    def thinkAbout(self, message, conversation, model="gpt-3.5-turbo", debug=False):
        # Check if the user's message is valid using OpenAI's Moderation API
        try:
            response = openai.Moderation.create(
                input=message
            )
        except:
            response = {"results":[{"flagged":False}]}

        valid_message = not response["results"][0]["flagged"]

        if model == "":
            model = "gpt-3.5-turbo"
        
        if valid_message:
            # Format the user's message and add it to the conversation
            FormattedMessage = {"role": "user", "content": message}
            conversation.append(FormattedMessage)
            
            # Generate a response using OpenAI's GPT-3 model
            try:
                response = response = opeanai_api_request(model="gpt-3.5-turbo",
                                                            messages=conversation,
                                                            temperature=self.agent_data["temperature"], 
                                                            frequency_penalty=self.agent_data["frequency_penalty"],
                                                            presence_penalty = self.agent_data["presence_penalty"],
                                                            top_p=self.agent_data["top_p"])
            except:
                try:
                    response = opeanai_api_request(model="gpt-3.5-turbo",
                                                            messages=conversation,
                                                            temperature=self.agent_data["temperature"], 
                                                            frequency_penalty=self.agent_data["frequency_penalty"],
                                                            presence_penalty = self.agent_data["presence_penalty"],
                                                            top_p=self.agent_data["top_p"])
                    response['choices'][0]['message']['content'] += ("(Generated by gpt 3.5 as you do not have access to gpt4)")
                except openai.error.AuthenticationError as e:
                    response = e.__str__()
                    conversation.append({"role": "assistant", "content":response})
                    return conversation
            
            if "error" in response.keys():
                    conversation.append({"role": "assistant", "content":response["error"]["message"]})
                    return conversation                

            # Format the response and add it to the conversation
            conversation.append({"role": "assistant", "content":response['choices'][0]['message']['content']})
            
            # Write the conversation to a log file
            if debug:
                with open("logs.txt", "w", encoding="utf-8") as file:
                    for i in conversation:
                        file.write(str(i)+"\n")
                    
            return conversation
        
        else:
            # If the user's message is not valid, add an error message to the conversation
            FormattedMessage = {"role": "user", "content": message}
            conversation.append(FormattedMessage)
            conversation.append({"role": "assistant", "content":"Je suis désolé, ceci est un message non valide car contraire aux termes et conditions"})
        
        # Return the conversation with or without the response depending on whether the message was valid
        return conversation