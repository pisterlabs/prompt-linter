import openai, numpy as np

prompt = input("Enter a string to create an embedding vector for: ")
response = openai.Embedding.create(
	input = prompt,
	engine = "text-similarity-davinci-001")

print("\n")	
print(response)

print("\nLet's find the similarity score between 'potato' and 'rhubarb'.")

response = openai.Embedding.create(
	input=["potato", "rhubarb"],
	engine="text-similarity-davinci-001")
	
potato = response['data'][0]['embedding']
rhubarb = response['data'][1]['embedding']

simScore = np.dot(potato, rhubarb)

print("\nScore is " + str(simScore) + "\n")

print("How about 'potato' and 'The starship Enterprise'")

response = openai.Embedding.create(
	input=["potato", "The starship Enterprise"],
	engine="text-similarity-davinci-001")
	
potato = response['data'][0]['embedding']
enterprise = response['data'][1]['embedding']

simScore = np.dot(potato, enterprise)

print("\nScore is " + str(simScore) + "\n")

