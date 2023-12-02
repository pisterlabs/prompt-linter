import cohere
co = cohere.Client('MP8mdQwd6kphSnGGlODq58uWJIggO71u6myCGhKF') # This is your trial API key

text = open("question.txt", "r")
text = text.read()
#Response
response = co.generate(
  model='command',
  prompt=f"You are an experienced scientist reading a document.  \
    You are tasked with identifying and extracting all the technical keywords, phrases, and concepts. \
    Be precise with your answer. Your output should be a machine-readable JSON list with a list of all the technical keywords, \
    phrases, and concepts with this format:\n{{\n    \"technical keywords\": [\n\n    ],\n    \"technical phrases\": [\n\n    ], \
    \"technical concepts\": [\n\n    ]\n}}\nDo not include any text other than a Python JSON list.\n\nGiven the \
    information above, extract all the technical keywords, phrases, and concepts from the below text:\n\n{text}",
  max_tokens=2148,
  temperature=0.4,
  k=0,
  stop_sequences=[],
  return_likelihoods='NONE')
print('Prediction: {}'.format(response.generations[0].text))

text.close()