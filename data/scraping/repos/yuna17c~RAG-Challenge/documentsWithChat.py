import cohere
import requests
import json

co = cohere.Client('DTguE3BfBw8Eq38DvDaaifavZq5h5qm52bFsD4MG')

documents = [
    {
      "title": "grands pingouins",
      "snippet": "mon pipi est rouge",
      "url": "www.google.ca"
    },
]
message = "quel est le plus gros animal"

response = co.chat(
	message,
	model="command",
    documents=documents,
	temperature=0.9
)

print(response.text)