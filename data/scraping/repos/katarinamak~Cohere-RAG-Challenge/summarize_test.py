# import cohere 
# co = cohere.Client('rGjz0KNIMSReCgEyzpEUDQpYzxSoXb85RjjdyAel') # This is your trial API key
# response = co.summarize( 
#   text='text summary',
#   length='auto',
#   format='auto',
#   model='command',
#   additional_command='',
#   temperature=0.3,
# ) 
# print('Summary:', response.summary)


import cohere
co = cohere.Client('rGjz0KNIMSReCgEyzpEUDQpYzxSoXb85RjjdyAel')


with open('subtitles.txt', 'r') as file:
    input_text = file.read()


response = co.summarize(
  text=input_text,
)
print(response)