import wikipediaapi
import faiss
import openai
import os

openai.api_key = "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"

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

# wiki_wiki = wikipediaapi.Wikipedia('MyProjectName (merlin@example.com)', 'en')

# page_py = wiki_wiki.page('Arizona_State_University')

# Firt, download the yt video
ytVideo = input("Enter the youtube video url: ")

# Delete all old files
# os.system("rm -rf *.wav")
# os.system("rm -rf *.mp3")
# os.system("rm -rf *.mp4")

# Download
# os.system("youtube-dl -x " + ytVideo)

audio_file= open("Motion Planning Algorithms (RRT, RRT_, PRM) - [MIT 6.881 Final Project]-gP6MRe_IHFo.m4a", "rb")
transcript = openai.Audio.transcribe("whisper-1", audio_file)
text = transcript['text']

# text = page_py.text

lines = text.split(".")

# Now, vectorize all lines
embeddings = model.encode(lines)

# Now, build our index
index = faiss.IndexFlatIP(embeddings[0].reshape(-1).shape[0])
index.add(embeddings)

while True:
  # Index our question
  q = input("Enter a question: ")

  # Find top indexes
  out = index.search(model.encode(q).reshape(1, -1), 10)

  # Now, create a list containg all indices and the surrounding N context lines
  N = 5
  idxs = []
  for idx in out[1].reshape(-1):
    idxs += [ idx + i for i in range(-N, N+1) ]

  idxs = list(set(idxs))
  idxs.sort()

  # At this point, we have all context lines. Feed all into GPT4, with the original
  # question
  context = "We will give you some context and a question. Answer the question as best as you can with the givens.\nContext:\n"

  for idx in idxs:
    context += lines[idx] + "\n"

  context += "\nQuestion: " + q + "\nAnswer:"

  # Now, feed into GPT4
  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{
      "role" : "user", "content" : context
    }],
    temperature=0.7,
    max_tokens=1000
  )

  print(response.choices[0]['message']["content"])