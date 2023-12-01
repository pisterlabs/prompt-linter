"""
  The mdsModel will return a MDS summary of the various articles that are passed into the model.
"""

from idl import *
import numpy as np
import logging
from logtail import LogtailHandler
import os
import openai
import logging
import nltk
import tensorflow_hub as hub
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

embedding_model = None
model_id = "google/pegasus-multi_news"

openai.api_key = "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"
module = "https://tfhub.dev/google/universal-sentence-encoder/4"

handler = LogtailHandler(source_token="tvoi6AuG8ieLux2PbHqdJSVR")
logger = logging.getLogger(__name__)
logger.handlers = [handler]
logger.setLevel(logging.INFO)


def process_paragraph(paragraphs_raw):
  """
    This function processes the paragraph length
  """
  paragraphs = []
  for paragraph in paragraphs_raw:
    if len(paragraph) > 1500:
      sentences = nltk.tokenize.sent_tokenize(paragraph)
      index = 0
      short_paragraph = ""

      while index < len(sentences):
        while (index < len(sentences) and len(short_paragraph) < 1500) :
          short_paragraph += sentences[index]
          index += 1
        paragraphs.append(short_paragraph)
        short_paragraph = ""
    else:
      paragraphs.append(paragraph)

  paragraphs = [paragraph for paragraph in paragraphs if paragraph != '' and len(nltk.tokenize.sent_tokenize(paragraph)) >= 2]

  paragraphs_original = [paragraph for paragraph in paragraphs_raw if paragraph != '' and len(nltk.tokenize.sent_tokenize(paragraph)) >= 2]

  if len(paragraphs) != len(paragraphs_original):
    logger.info("processed paragraphs: ")
    logger.info([len(paragraph) for paragraph in paragraphs])

  return paragraphs


def get_mds_summary_v3_handler(getMDSSummaryRequest):
  """
    This approach will first extract the top passages per article then concatenate all of them and pass it into pegasus multi news for eval.
  """

  global embedding_model
  if embedding_model == None:
    print("Embedding model is none in add document")
    embedding_model = hub.load(module)

  articles = getMDSSummaryRequest.articles

  try:
    nltk.data.find('tokenizers/punkt')
  except LookupError:
    nltk.download('punkt')

  paragraphs = articles.split("\n")

  articleParagraphs = process_paragraph(paragraphs)

  if len(articleParagraphs) == 0 or len(articleParagraphs[0]) < 100:
    logger.warn("No paragraphs for summary")
    return GetMDSSummaryAndTitleResponse(
      summary="",
      title="",
      error= None,
    )

  if len(paragraphs) <= 1:
    return GetMDSSummaryAndTitleResponse(
      summary=paragraphs[0],
      title="",
      error= None,
    )

  embeddedParagraphs = []
  for para in articleParagraphs:
    if para != '':
      embeddedParagraphs.append(embedding_model([para]))
  embeddedParagraphs = np.squeeze(embeddedParagraphs)

  # Create a matrix of facts and compute the dot product between the matrices
  dot_products = np.dot(embeddedParagraphs, embeddedParagraphs.T)
  if len(dot_products) > 1:
    dot_product_sum = sum(dot_products)
  else:
    dot_product_sum = dot_products[0]

  # Take the indices of the top 10 passages so far
  topParagraphIndices = np.argpartition(dot_product_sum, -1)[-10:]

  topParagraphs = [articleParagraphs[index] for index in topParagraphIndices]

  topParagraphs = " ".join(topParagraphs)
  logger.info("INPUT PARAGRAPHS: " + str(topParagraphs))


  # Pass the combined paragraphs to the pegasus model
  model_id_multi_news = "google/pegasus-multi_news"
  device = "cuda" if torch.cuda.is_available() else "cpu"
  tokenizer = PegasusTokenizer.from_pretrained(model_id_multi_news)
  model = PegasusForConditionalGeneration.from_pretrained(model_id_multi_news).to(device)
  batch = tokenizer(topParagraphs, truncation=True, padding="longest", return_tensors="pt").to(device)
  translated = model.generate(**batch, no_repeat_ngram_size=5, num_beams=5, max_length=150,early_stopping=True, repetition_penalty=100.0)
  summary = tokenizer.batch_decode(translated, skip_special_tokens=True)

  logger.info("OUTPUT PEGASUS SUMMARY: " + str(summary[0]))

  if getMDSSummaryRequest.include_title:

    model_id_xsum = "google/pegasus-xsum"
    tokenizer = PegasusTokenizer.from_pretrained(model_id_xsum)
    model = PegasusForConditionalGeneration.from_pretrained(model_id_xsum)
    batch = tokenizer(topParagraphs, truncation=True, padding="longest", return_tensors="pt").to(device)
    translated = model.generate(**batch, no_repeat_ngram_size=5, num_beams=5, early_stopping=True, repetition_penalty=100.0)
    title = tokenizer.batch_decode(translated, skip_special_tokens=True)

    logger.info("OUTPUT TITLE: " + str(title[0]))

    return GetMDSSummaryAndTitleResponse(
      summary=summary[0],
      title=title[0],
      error= None,
    )

  return GetMDSSummaryAndTitleResponse(
    summary=summary[0],
    title="",
    error= None,
  )


def get_mds_summary_v2_handler(getMDSSummaryRequest):
  """
    Get the MDS summary using a more standard approach.  This will include practically doing extraction from the legit concatenation of all the article text in order to pick out the most legit sentences from each of the articles. This way you can guarantee that the text actually makes sense, it is constant, it is quick, it is factual "no chance of making things up" and you aren't fing relying on so much GPT credits and get repetitive content out.
  """

  global embedding_model
  if embedding_model == None:
    print("Embedding model is none in add document")
    embedding_model = hub.load(module)

  articles = getMDSSummaryRequest.articles

  try:
    nltk.data.find('tokenizers/punkt')
  except LookupError:
    nltk.download('punkt')

  paragraphs = articles.split("\n")

  articleParagraphs = process_paragraph(paragraphs)

  if len(articleParagraphs) == 0 or len(articleParagraphs[0]) < 100:
    logger.warn("No paragraphs for summary")
    return GetMDSSummaryResponse(
      summary="",
      error= None,
    )

  if len(paragraphs) <= 1:
    return GetMDSSummaryResponse(
      summary=paragraphs[0],
      error= None,
    )

  embeddedParagraphs = []
  for para in articleParagraphs:
    if para != '':
      embeddedParagraphs.append(embedding_model([para]))
  embeddedParagraphs = np.squeeze(embeddedParagraphs)

  # Create a matrix of facts and compute the dot product between the matrices
  dot_products = np.dot(embeddedParagraphs, embeddedParagraphs.T)
  dot_product_sum = sum(dot_products)

  topParagraphIndices = np.argpartition(dot_product_sum, -1)[-1:]

  topParagraphs = [articleParagraphs[index] for index in topParagraphIndices]
  mdsSummary = " ".join(topParagraphs)

  return GetMDSSummaryResponse(
    summary=mdsSummary,
    error= None,
  )


def get_mds_summary_handler(getMDSSummaryRequest):
  """
    Get the MDS summary and return it
  """
  articles = getMDSSummaryRequest.articles

  # TODO: Actually implement your model here!!!
  # This is just a placeholder for initial launch ease

  prompt = articles[:2000] + "\n\ntl;dr"
  summary = openai.Completion.create(
    engine="davinci",
    prompt=prompt,
    temperature=0.3,
    max_tokens=60,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
  )

  return GetMDSSummaryResponse(
    summary=summary.choices[0].text,
    error= None,
  )



