"""Python file to serve as the frontend"""

import pandas as pd
import PyPDF2
import streamlit as st
from clarifai.auth.helper import ClarifaiAuthHelper
from clarifai.client import create_stub
from clarifai.modules.css import ClarifaiStreamlitCSS
from geopy.geocoders import Nominatim
from langchain import LLMChain, PromptTemplate
from langchain.llms import Clarifai


from utils.prompts import NER_LOC_PROMPT
from utils.upload_utils import post_texts_with_geo, split_into_chunks, word_counter

st.set_page_config(page_title="Upload App", page_icon=":robot:")

ClarifaiStreamlitCSS.insert_default_css(st)

USER_ID = "openai"
APP_ID = "chat-completion"
MODEL_ID = "GPT-3_5-turbo"

auth = ClarifaiAuthHelper.from_streamlit(st)
stub = create_stub(auth)
userDataObject = auth.get_user_app_id_proto()
st.title("Upload PDF as text chunks")
st.markdown(
    "This will chunk up the PDF into text pages and upload them to our platform. This also fills in the `data.metadata.source = {name of document}`\
        and `data.metadata.page_number = {page number of document}`.")

geolocator = Nominatim(user_agent="test")

text_chunk_size = st.number_input(
    "Text chunk size", min_value=100, max_value=3000, value=300, step=100)
uploaded_file = st.file_uploader("Upload a PDF", type="pdf", key="qapdf")


def visitor_body(text, cm, tm, fontDict, fontSize):
  y = tm[5]
  if y > 50 and y < 770:
    parts.append(text)


if uploaded_file:
  reader = PyPDF2.PdfReader(uploaded_file)
  try:
    document_title = reader.metadata.title.split(".")[0]
  except AttributeError:
    document_title = uploaded_file.name.split(".")[0]

  text_chunks = []
  page_number_list = []
  page_chunk_number_list = []
  prev_page_text = ""
  for page_idx, page in enumerate(reader.pages):
    parts = []
    page.extract_text(visitor_text=visitor_body)
    current_page_text = "".join(parts)
    page_text = prev_page_text + current_page_text

    # Check if text is smaller the text_chunk_size, if so, add it to the previous page text holder
    if (word_counter(page_text) < text_chunk_size) and (page_idx != len(reader.pages) - 1):
      prev_page_text += current_page_text
      continue
    else:
      prev_page_text = ""

    page_text_chunks = split_into_chunks(page_text, text_chunk_size)
    text_chunks.extend(page_text_chunks)
    page_number_list.extend([page_idx] * len(page_text_chunks))
    page_chunk_number_list.extend([idx for idx in range(len(page_text_chunks))])
    print("length of text chunks", len(text_chunks))

  # # Save list to a txt file for debugging
  # with open(f"{uploaded_file_name}.txt", "w") as f:
  #     for item in text_chunks:
  #         f.write("%s\n" % item)

  print("len(text_chunks): ", len(text_chunks))
  print("len(page_number_list): ", len(page_number_list))
  assert len(text_chunks) == len(
      page_number_list), "Text chunks and page numbers should be the same length"

  metadata_list = [{
      "source": f"{document_title}",
      "text_length": word_counter(text_chunks[idx]),
      "page_number": page_number_list[idx],
      "page_chunk_number": page_chunk_number_list[idx],
  } for idx in range(len(text_chunks))]

  metadata_df = pd.DataFrame(metadata_list)
  metadata_sorted_df = metadata_df.sort_values(["page_number", "page_chunk_number"])
  st.dataframe(metadata_sorted_df)

  auth = ClarifaiAuthHelper.from_streamlit(st)
  pat = auth._pat
  llm_chatgpt = Clarifai(pat=pat, user_id=USER_ID, app_id=APP_ID, model_id=MODEL_ID)
  prompt = PromptTemplate(template=NER_LOC_PROMPT, input_variables=["page_content"])
  llm_chain = LLMChain(prompt=prompt, llm=llm_chatgpt)

  geo_points_list = []
  with st.expander("Show Locations Found"):
    for idx, text_chunk in enumerate(text_chunks):
      chat_output = llm_chain(text_chunk)
      st.write(chat_output["text"])

      try:
        geo_points = {}
        location = geolocator.geocode(chat_output["text"], language="en", timeout=None)
        st.info(f"{location.address} {location.latitude, location.longitude}")
        geo_points["lat"] = location.latitude
        geo_points["lon"] = location.longitude
        geo_points_list.append(geo_points)
        metadata_list[idx]["location"] = str(location.address)

      except AttributeError:
        st.warning("No location found")
        geo_points = {}
        geo_points["lat"] = None
        geo_points["lon"] = None
        geo_points_list.append(geo_points)

  #post_texts_with_geo used to upload input in the app along with metadata and ids.
  post_texts_with_geo(st, stub, userDataObject, text_chunks, metadata_list, geo_points_list)
  st.success("Done!")
  st.balloons()
