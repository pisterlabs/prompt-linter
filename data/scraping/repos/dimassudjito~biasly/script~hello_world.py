import cohere
import time

import cohere
co = cohere.Client('kcLpZwryzepeisB5zpCSQtEVLeWdUVlGlUMFi2Db')

message = "Hello World!"

response = co.chat(
	message, 
	model="command", 
	temperature=0.9
)

answer = response.text
print(answer)