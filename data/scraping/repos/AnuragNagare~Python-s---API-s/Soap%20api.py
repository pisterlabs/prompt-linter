#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from spyne import Application, rpc, ServiceBase, Unicode
from spyne.protocol.soap import Soap11
from spyne.server.wsgi import WsgiApplication
import requests

OPENAI_API_KEY = "Your Api key"
OPENAI_API_ENDPOINT = "https://api.openai.com/v1/engines/davinci/completions"

class OpenAIService(ServiceBase):
    @rpc(Unicode, _returns=Unicode)
    def generate_response(ctx, prompt):
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "prompt": prompt,
            "max_tokens": 50  # Adjust as needed
        }

        response = requests.post(OPENAI_API_ENDPOINT, json=payload, headers=headers)

        if response.status_code == 200:
            response_data = response.json()
            if 'choices' in response_data:
                answer = response_data['choices'][0]['text'].strip()
                return answer
            else:
                return "Error: Unable to extract valid response from OpenAI"
        else:
            return f"Error: OpenAI API error - {response.content}"

application = Application([OpenAIService], 'OpenAI',
                          in_protocol=Soap11(validator='lxml'),
                          out_protocol=Soap11())

if __name__ == '__main__':
    from wsgiref.simple_server import make_server

    wsgi_app = WsgiApplication(application)

    server = make_server('0.0.0.0', 8000, wsgi_app)
    print("SOAP Server started on http://localhost:8000")
    server.serve_forever()

