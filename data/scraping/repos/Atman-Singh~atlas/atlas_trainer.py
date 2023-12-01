from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from gensim.test.utils import common_texts
import gensim
import os

# uses a directory of text files to train the LLM
def main():
    tokens = []
    load_dotenv()

    # text to be read
    text = ''

    # get words from directory
    directory = r'C:\Users\Atman S\Documents\GM0 RSTs\\'
    for files in os.listdir(directory):
        with open(directory + files, encoding="utf8") as f:
            text += f.read()
            
    # split extracted text into digestable chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(filter(text))

    # encode chunks
    for i in chunks:
        tokens += gensim.utils.simple_preprocess(i)
    tokens = [tokens]
        
    model = gensim.models.Word2Vec(tokens, min_count=1)
    
    # train model
    model.build_vocab(tokens, progress_per=10)
    model.train(tokens, total_examples=model.corpus_count, epochs=model.epochs)
    print("done training")   

    # save settings to atlas.model
    model.save("atlas.model")

# deletes unwanted characters
def filter(text):
    text = text.replace("-", "")
    text = text.replace("^", "")
    text = text.replace("\n", "")
    text = text.replace("Â°", " degrees")
    text = text.replace("\n", "")
    text = text.replace("~", "")
    text = text.replace("=", "")
    return text

# run main method
if __name__ == '__main__':
    main()
    