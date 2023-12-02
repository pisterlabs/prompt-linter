import cohere 
f = open('rewriteParagraph2.txt', 'r')
#then to store
content = f.read()
# print(content)

co = cohere.Client('') 
response = co.generate( 
  model='xlarge', 
  prompt=content, 
  max_tokens=100, 
  temperature=0.9, 
  k=0, 
  p=0.75, 
  frequency_penalty=0, 
  presence_penalty=0, 
  stop_sequences=[], 
  return_likelihoods='NONE') 
resText = response.generations[0].text
print("=====================================")
print(resText)
print("=====================================")
# resText = resText.split('--')[0]
# resList = resText.strip().split(',')
# resList = [e.strip() for e in resList]
# print(resList)