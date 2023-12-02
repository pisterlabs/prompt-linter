import api_key

flickr_api_key = api_key.flickr_api_key()
flickr_api_secret = api_key.flickr_api_secret()

YOUR_CLARIFAI_API_KEY = api_key.clarify_apikey()
YOUR_APPLICATION_ID = api_key.clarify_APPLICATION_ID()


from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import service_pb2_grpc
stub = service_pb2_grpc.V2Stub(ClarifaiChannel.get_grpc_channel())
from clarifai_grpc.grpc.api import service_pb2, resources_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2

metadata = (("authorization", f"Key {YOUR_CLARIFAI_API_KEY}"),)

import flickrapi
import urllib

flickr=flickrapi.FlickrAPI(flickr_api_key, flickr_api_secret, cache=True)

keyword = 'beach'

photos = flickr.walk(text=keyword,
                     tag_mode='all',
                     tags=keyword,
                     extras='url_c',
                     per_page=100,           # may be you can try different numbers..
                     sort='relevance')

urls = []
for i, photo in enumerate(photos):
    url = photo.get('url_c')
    urls.append(url)

    # get 50 urls
    if i > 50:
        break

for k,i in enumerate(urls):
    print(k,i)


keywords = []
for i in range(10):
    t = urls[i]
    SAMPLE_URL = t

    request = service_pb2.PostModelOutputsRequest(
        # This is the model ID of a publicly available General model. You may use any other public or custom model ID.
        model_id="general-image-recognition",
        user_app_id=resources_pb2.UserAppIDSet(app_id=YOUR_APPLICATION_ID),
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(image=resources_pb2.Image(url=SAMPLE_URL))
            )
        ],
    )
    response = stub.PostModelOutputs(request, metadata=metadata)

    if response.status.code != status_code_pb2.SUCCESS:
        print(response)
        raise Exception(f"Request failed, status code: {response.status}")

    for concept in response.outputs[0].data.concepts:
        print("%12s: %.2f" % (concept.name, concept.value))
        if(concept.value>0.9):
            keywords.append(concept.name)

keywords = list(set(keywords))
print(keywords)



openai_key = api_key.openai_key()

import os
import openai
openai.api_key = openai_key

prompting_text = '''
write a long postcard to friends associate with following notes
'''+', '.join(keywords)

import translators as ts
response = openai.Completion.create(
  model="text-davinci-002",
  prompt=prompting_text,
  temperature=1,
  max_tokens=1024,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0,
  stop=['/n/n/n/n']
)

print(response.choices[0].text)

prompting_text = '''
write a sentence using all the following words
['''+', '.join(keywords)+']'

response = openai.Completion.create(
  model="text-davinci-002",
  prompt=prompting_text,
  temperature=1,
  max_tokens=1024,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0,
  stop=['/n/n/n/n']
)
print(response.choices[0].text)
t = ts.google(response.choices[0].text,from_language='en',to_language='ru')
print(t)
