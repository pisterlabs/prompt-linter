from langchain.llms import Replicate

llm = Replicate(model="replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1")
print(llm)

response = llm("What is the adapter design pattern?")
print(response)