import streamlit as st
from requests_toolbelt.multipart.encoder import MultipartEncoder
import requests
from PIL import Image
import io
import ZSL



st.title('Zero Shot Topic Classification')

# fastapi endpoint
url = 'http://fastapi:8000'
endpoint = '/segmentation'

st.write('''Recently, the NLP science community has begun to pay increasing attention to zero-shot and few-shot applications, such as in the paper from OpenAI introducing GPT-3. This demo shows how 🤗 Transformers can be used for zero-shot topic classification, the task of predicting a topic that the model has not been trained on.''')  # description and instructions

# image = st.file_uploader('insert image')  # image upload widget
#
#
# def process(image, server_url: str):
#
#     m = MultipartEncoder(
#         fields={'file': ('filename', image, 'image/jpeg')}
#         )
#
#     r = requests.post(server_url,
#                       data=m,
#                       headers={'Content-Type': m.content_type},
#                       timeout=8000)
#
#     return r
#
#
# if st.button('Get segmentation map'):
#
#     if image is None:
#         st.write("Insert an image!")  # handle case with no image
#     else:
#         segments = process(image, url+endpoint)
#         segmented_image = Image.open(io.BytesIO(segments.content)).convert('RGB')
#         st.image([image, segmented_image], width=300)  # output dyptich

##### New code
user_input = st.text_area("insert text", '')
labels = ['soccer', 'programming', 'sport', 'education', 'jewish', 'israel', 'palestine', 'islam', 'football',
          'health care', 'movies']
st.write('this text is about:', ZSL.print_similarities(user_input, labels))
