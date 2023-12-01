import cohere
co = cohere.Client('ArCSzp9peaChzIXMLkNILdyifB0JzifENzD5DSwb')
response = co.generate(
  model='command-xlarge-nightly',
  prompt= 'in five sentences, answer this: ' + input(''),
  max_tokens=300,
  temperature=0,
  k=0,
  p=0.75,
  stop_sequences=[],
  return_likelihoods='NONE')
print('Answer: {}'.format(response.generations[0].text))