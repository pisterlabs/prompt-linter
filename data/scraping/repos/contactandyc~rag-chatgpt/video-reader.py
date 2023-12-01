import sys, os, locale
from langchain.document_loaders import YoutubeLoader
import spacy

def getpreferredencoding(do_setlocale = True):
    return "UTF-8"

locale.getpreferredencoding = getpreferredencoding

nlp = spacy.load("en_core_web_sm")

# Use the YoutubeLoader to load and parse the transcript of a YouTube video
def load_text_from_video(vid):
    loader = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=" + vid, add_video_info=True)
    video = loader.load()
    return video

datadir = sys.argv[1]
if datadir[-1] == '/':
    datadir = datadir[0:-1]

os.makedirs(datadir, exist_ok=True)

videos = ['VMj-3S1tku0', 'PaCmpygFfXo']
if len(sys.argv) > 2:
    videos = sys.argv[2:]

text_file = open(datadir + '/input.txt', 'w')

for vid in videos:
    doc = load_text_from_video(vid)
    text = doc[0].page_content
    sentences = [sent.text for sent in nlp(text).sents]
    text_file.write(doc[0].metadata["title"])
    text_file.write('\n\n')
    for s in sentences:
        text_file.write(s)
        text_file.write('\n')

text_file.close()
