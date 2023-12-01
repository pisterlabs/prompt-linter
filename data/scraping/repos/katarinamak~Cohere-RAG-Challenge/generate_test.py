import cohere
co = cohere.Client('rGjz0KNIMSReCgEyzpEUDQpYzxSoXb85RjjdyAel') # This is your trial API key
response = co.generate(
  model='command',
  prompt='what is the best university in waterloo?\n',
  max_tokens=300,
  temperature=0.9,
  k=0,
  stop_sequences=[],
  return_likelihoods='NONE')
print('Prediction: {}'.format(response.generations[0].text))
