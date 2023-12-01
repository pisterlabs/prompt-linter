import cohere
from tqdm import tqdm
import datasets
import torch
import wikipediaapi
import json
import numpy as np
import sys

co = cohere.Client('<key>')

class Chat:
  """
  Class for chatting with the model with persistent chat history.
  """

  def __init__(self):
    """
    Initialize with an empty chat history.
    """

    self.chat_history = []

  def __call__(self, *args, **kwargs):
    """
    Chat with the model. This function is a wrapper around co.chat that
    automatically persists the chat history.
    """

    # Prompt the model
    prompt = args[0]
    response = co.chat(prompt, *args[1:], **kwargs, chat_history=self.chat_history)

    # Update the chat history
    self.chat_history.append({"role": "USER", "message": prompt})
    self.chat_history.append({"role": "CHATBOT", "message": response.text})

    # Return the response
    return response.text

def join(title, lyrics):
  """
  Join the title and lyrics of a song.
  """

  return f"{title}:\n{lyrics}"

def summarize(title, lyrics):
  """
  Return a summarized version of the song.
  """

  summary = co.summarize(join(title, lyrics)).summary

  return summary.strip()  

def summarize_if_possible(text):
  try:
    return co.summarize(text).summary
  except Exception as e:
    print("Couldn't summarize text:", e)
    return text

def download_wikipedia_summary(title):
  """
  Download the summary of a wikipedia article.
  """

  wiki_wiki = wikipediaapi.Wikipedia(language='en', user_agent='jmcs/0.0')

  page = wiki_wiki.page(title)
  
  if page.exists():
      return page.summary
  else:
      return None

def filter_top_k(song_summary, sentence_list, num_keep):
  """
  Return the top k sentences from a list of sentences.
  """

  # Compute the embeddings for the sentences
  embeddings = np.array(co.embed(sentence_list + [song_summary]).embeddings)

  # Compute the distance between the song summary and each sentence
  song_summary = embeddings[-1]
  sentence_embeddings = embeddings[:-1]
  
  distances = np.linalg.norm(song_summary - sentence_embeddings, axis=-1)

  # Keep the top k sentences
  keep_indices = distances.argsort()[:num_keep]

  return keep_indices

def get_wikipedia_articles(song_summary, num_articles = 10, num_keep = 3):
  """
  Return a list of wikipedia articles related to the song.
  """

  chat = Chat()

  # Ask the chat model for a list of wikipedia articles related to the song
  freeform_wiki = chat(
      f"Consider the following song summary. List wikipedia articles that are relevant to its subject. Include at least {num_articles} articles. Song: Genocide. Summary: {song_summary}",
  )

  # Ask the chat model to reformat its previous output as a bracketed list
  bracketed_wiki = chat(
     "Provide the previous list of wikipedia titles in a simple array format. Example: ['song number 1', 'second song', 'sample song']."  
  )

  if not ("[" in bracketed_wiki and "]" in bracketed_wiki):
    print("Couldn't extract wikipedia articles.")
    return {'Error': 'failed'}
  
  # Extract the list from the bracketed string
  extracted_list = bracketed_wiki[bracketed_wiki.find("["):bracketed_wiki.find("]")+1]
  extracted_list = json.loads(extracted_list.replace("'", '"'))

  titles = []
  summaries = []

  # Download the wikipedia summaries for each article
  for title in tqdm(extracted_list):
    content = download_wikipedia_summary(title)

    if content is not None:
      content = summarize_if_possible(content)
      titles.append(title)
      summaries.append(content)

  # Use embeddings to filter the top k articles
  keep_indices = filter_top_k(song_summary, summaries, num_keep)
  best_articles = {titles[i]: summaries[i] for i in keep_indices}

  return best_articles



def get_related_media(song_summary, num_media = 10, num_keep = 3):
  """
  Return a list of books related to the song.
  """

  chat = Chat()

  # Ask the chat model for a list of books and movies related to the song
  response = chat(
    f"""Based on this information, refer to a list of other media, such as books or movies, that cover the same theme as the song. In total, generate at least {num_media} media. 
    The original song description:
    {song_summary}.
    Remember, refer to other media that cover the same theme as the song. Provide a short description of each media, and why it is relevant to the song. Only include media that exists in the real world. Classify them between "BOOK" and "MOVIE". Include these keywords. For example:
    BOOK: Book name - Book description.
    MOVIE: Name of the movie - Description of the movie.
    ...
    In total, generate at least {num_media} media.""",
    temperature=0.1
  )

  if not ("BOOK:" in response and "MOVIE:" in response):
    print("Couldn't extract books and movies.")
    return {'Error': 'failed'}

  # Extract book and movie lines
  lines = response.split("\n")
  book_lines = [line.split('BOOK:')[1] for line in lines if "BOOK:" in line]
  movie_lines = [line.split('MOVIE:')[1] for line in lines if "MOVIE:" in line]

  # Select the top k books and movies
  book_indices = filter_top_k(song_summary, book_lines, num_keep)
  movie_indices = filter_top_k(song_summary, movie_lines, num_keep)

  # Extract the book and movie titles
  book_lines = [book_lines[i].replace('"', '') for i in book_indices]
  books = {book_line.split(" - ")[0].strip(): book_line.split(" - ")[1].strip() for book_line in book_lines if len(book_line.split(" - ")) == 2}

  movie_lines = [movie_lines[i].replace('"', '') for i in movie_indices]
  movies = {movie_line.split(" - ")[0].strip(): movie_line.split(" - ")[1].strip() for movie_line in movie_lines if len(movie_line.split(" - ")) == 2}

  return books, movies

def get_related_books(song_summary):
  """
  Return a list of books related to the song.
  """

  return get_related_media(song_summary)[0]

def get_related_movies(song_summary):
  """
  Return a list of movies related to the song.
  """

  return get_related_media(song_summary)[1]