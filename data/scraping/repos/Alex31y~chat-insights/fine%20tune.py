import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from langchain.document_loaders import PyPDFLoader
#from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredFileLoader
from langchain.utilities import WikipediaAPIWrapper
import re
import numpy as np
import openai
import os
from sentence_transformers import SentenceTransformer, InputExample, losses, models, datasets, util
import torch
torch.cuda.is_available()
import threading
from tqdm import tqdm

chunk_size = 500
n_chunks = 15
model = SentenceTransformer('all-MiniLM-L6-v2')
#model = SentenceTransformer("bert model")
#model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')
#model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
embeddings_file = f"docs\embeddings.npz"

def preprocess(text):
    text = text.replace('\n', ' ')
    text = re.sub('\s+', ' ', text)
    return text

def to_text(file_paths, url):
    text = ""
    if url:
        wikipedia = WikipediaAPIWrapper()
        pages = wikipedia.run(file_paths)
        #print(pages)
        page = preprocess(pages)
        text += page
    elif file_paths:
        for file in file_paths:
            print(file)
            file_extension = os.path.splitext(file)[1]
            # Check if the file extension is ".txt" or ".word"
            if file_extension == ".txt":
                print("File extension is .txt")
                loader = UnstructuredFileLoader(file)
            elif file_extension == ".pdf":
                print("File extension is .pdf")
                loader = PyPDFLoader(file)
            else:
                print("File extension not supported")
                return

            pages = loader.load_and_split()
            for page in pages:
                page = page.page_content
                page = preprocess(page)
                text += page
    else:
        text = "input non riuscito"
    return text

def text_to_chunks(text, overlap_percentage = 0.05):
    chunks = []
    overlap_size = int(chunk_size * overlap_percentage)

    start = 0
    end = chunk_size

    while start < len(text):
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap_size
        end = start + chunk_size

    return chunks

class SemanticSearch:

    def __init__(self):
        self.fitted = False

    def encode(self, chunk):
        embeddings = model.encode(chunk, convert_to_tensor=True)
        return embeddings

    # applico il nearest neighbors sull'embedding del pdf
    def fit(self, data, batch=1000, n_neighbors=5):
        self.data = data  # salvo i chunks del pdf in data
        self.corpus_embeddings = self.get_text_embedding(data, batch=batch)  # qui creo gli embedding
        self.fitted = True

    def fit2(self, embeddings_file):
        stored = np.load(embeddings_file)
        self.corpus_embeddings = stored["embeddings"]  # qui creo gli embedding
        self.dim_corpus = stored["dim_corpus"]
        self.fitted = True

    # restituisco i top n chunks più simili alla domanda
    def __call__(self, text): # text è la domanda input dell'utente
        top_k = min(n_chunks, self.dim_corpus)
        domanda_embeddings = self.encode([text]) # embedding applicato alla domanda
        #scores = util.cos_sim(domanda_embeddings, self.corpus_embeddings)[0]
        scores = util.dot_score(domanda_embeddings, self.corpus_embeddings)[0]
        top_results = torch.topk(scores, k=top_k)
        return [self.data[i] for i in top_results[1]]

    # questa è la classe che fa l'embedding sul contenuto del pdf
    def get_text_embedding(self, texts, batch):
        embeddings = []
        print("Numero di chunks nel documento: ")
        print(len(texts))
        self.dim_corpus = len(texts)

        for i in range(0, len(texts), batch):
            text_batch = texts[i:(i + batch)]
            emb_batch = self.encode(text_batch)  # chiamo il modello
            embeddings.append(emb_batch)
        embeddings = np.vstack(embeddings)  #a function provided by NumPy that takes a sequence of arrays and stacks them vertically to form a new array.
        return embeddings

def generate_text(openAI_key, prompt, engine="text-davinci-003"):
    openai.api_key = openAI_key
    completions = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        max_tokens=500,     #incide sulla lunghezza della risposta output, quindi sul prezzo della chiamata
        n=1,
        stop=None,
        temperature=0.7,    #più è basso meno si spende, forse
    )
    message = completions.choices[0].text
    return message

def generate_answer(question, openAI_key):
    # genero i chunks
    topn_chunks = recommender(question) # metodo __call__ : confronto l'embedding della domanda all'embedding del pdf ed ottengo gli n snippet di testo più vicini
    prompt = ""
    prompt += 'search results:\n\n'
    for c in topn_chunks:
        prompt += c + '\n\n'

    prompt += "Instructions: Compose a comprehensive reply to the query using the search results given. " \
              "If the search results mention multiple subjects with the same name, create separate answers for each. " \
              "Only include information found in the results and don't add any additional information. " \
              "Make sure the answer is correct and don't output false content. " \
              "If the text does not relate to the query, simply state 'Non è stata trovata una risposta alla tua domanda nel testo'." \
              "Ignore outlier search results which has nothing to do with the question. Only answer what is asked. " \
              "The answer should be short and concise. Answer step-by-step. \n\nQuery: {question}\nAnswer: "

    prompt += f"Query: {question}\nAnswer:"

    """
    file_object = open('docs\domande.txt', 'a')
    file_object.write('\n\n\nprompt:\n')
    file_object.write(prompt)
    # Close the file
    file_object.close()
    """

    answer = generate_text(openAI_key, prompt, "text-davinci-003")
    #answer = prompt
    print(prompt)
    return answer

def cleanup():

    if os.path.isfile(embeddings_file):
        os.remove(embeddings_file)

def run():
    while not stop_event.is_set():
        progress_bar.pack()
        progress_bar.start()
        query = query_entry.get()
        api_key = key_entry.get()
        url = url_entry.get()

        if(url):
            print("url true")
            text = to_text(url, True)
            chunks = text_to_chunks(text)
            recommender.fit(chunks)
        else:
            print("url false")
            global file_paths
            if os.path.isfile(embeddings_file):
                recommender.fit2(embeddings_file)
                print("Embeddings loaded from file")
            else:
                text = to_text(file_paths, False)
                chunks = text_to_chunks(text)
                recommender.fit(chunks)
                np.savez(embeddings_file, embeddings=recommender.corpus_embeddings, dim_corpus=recommender.dim_corpus)
        progress_bar.stop()
        progress_bar.pack_forget()
        answer = generate_answer(query, api_key)
        text_area.pack()
        text_area.delete(1.0, tk.END)
        text_area.insert(tk.END, answer)
        stop_event.set()  # Set the stop event to stop the thread
        break

def start_thread():
    global thread, stop_event
    if thread and thread.is_alive():
        print("già in esecuzione")
        return  # Do nothing if a thread is already running
    stop_event.clear()  # Reset the stop event

    # Start the new thread
    stop_event = threading.Event()
    thread = threading.Thread(target=run)
    thread.start()

def input_file():
    cleanup()
    global file_paths
    filetypes = [("PDF Files", "*.pdf"), ("Text Files", "*.txt"), ("Word Files", "*.docx")]
    file_paths = filedialog.askopenfilenames()

def update_sizes(event=None):
    window.update_idletasks()  # Update the window to get the current size
    text_area.configure(width=(window.winfo_width() // 10), height=(window.winfo_height() // 25))

def fine_tune():
    global file_paths
    text = to_text(file_paths, False)
    chunks = text_to_chunks(text)
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    import torch
    tokenizer = T5Tokenizer.from_pretrained('BeIR/query-gen-msmarco-t5-large-v1')
    model = T5ForConditionalGeneration.from_pretrained('BeIR/query-gen-msmarco-t5-large-v1')
    # Select the device
    device = 'cuda'
    model.to(device)

    # Parameters for generation
    batch_size = 16  # Batch size
    num_queries = 5  # Number of queries to generate for every paragraph
    max_length_paragraph = 512  # Max length for paragraph
    max_length_query = 64  # Max length for output query

    def _removeNonAscii(s):
        return "".join(i for i in s if ord(i) < 128)

    with open('generated_queries_all.tsv', 'w') as fOut:
        for start_idx in tqdm(range(0, len(chunks), batch_size)):
            sub_paragraphs = chunks[start_idx:start_idx + batch_size]
            inputs = tokenizer.prepare_seq2seq_batch(sub_paragraphs, max_length=max_length_paragraph, truncation=True,
                                                     return_tensors='pt').to(device)
            outputs = model.generate(
                **inputs,
                max_length=max_length_query,
                do_sample=True,
                top_p=0.95,
                num_return_sequences=num_queries)

            for idx, out in enumerate(outputs):
                query = tokenizer.decode(out, skip_special_tokens=True)
                query = _removeNonAscii(query)
                para = sub_paragraphs[int(idx / num_queries)]
                para = _removeNonAscii(para)
                fOut.write("{}\t{}\n".format(query.replace("\t", " ").strip(), para.replace("\t", " ").strip()))

    # Now we create a SentenceTransformer model from scratch
    word_emb = models.Transformer('all-MiniLM-L6-v2')
    pooling = models.Pooling(word_emb.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_emb, pooling])

    train_examples = []
    with open('generated_queries_all (1).tsv') as fIn:
        for line in fIn:
            try:
                query, paragraph = line.strip().split('\t', maxsplit=1)
                train_examples.append(InputExample(texts=[query, paragraph]))
            except:
                pass

    # For the MultipleNegativesRankingLoss, it is important
    # that the batch does not contain duplicate entries, i.e.
    # no two equal queries and no two equal paragraphs.
    # To ensure this, we use a special data loader
    train_dataloader = datasets.NoDuplicatesDataLoader(train_examples, batch_size=8)
    # MultipleNegativesRankingLoss requires input pairs (query, relevant_passage)
    # and trains the model so that is is suitable for semantic search
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Tune the model
    num_epochs = 3
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=num_epochs, warmup_steps=warmup_steps,
              show_progress_bar=True)

    os.makedirs('search', exist_ok=True)
    model.save('search/search-model')


recommender = SemanticSearch()
# Create the main window
window = tk.Tk()
window.title("Chat Insights")
window.geometry("600x400")

file_paths = None
select_button = tk.Button(window, text="Select Files", command=input_file)
select_button.pack()
url_label = tk.Label(window, text="oppure inserisci un URL:")
url_label.pack()
url_entry = tk.Entry(window, width=60)
url_entry.pack()
key_label = tk.Label(window, text="API Key:")
key_label.pack()
key_entry = tk.Entry(window, width=60)
key_entry.pack()
query_label = tk.Label(window, text="Fai la tua domanda:")
query_label.pack()
query_entry = tk.Entry(window, width=80)
query_entry.pack()
select_button = tk.Button(window, text="Fine Tuning del modello", command=fine_tune)
select_button.pack()
# Create button to retrieve the query and API key
submit_button = tk.Button(window, text="Submit", command=start_thread)
submit_button.pack()
stop_event = threading.Event()
thread = None
progress_bar = ttk.Progressbar(window, mode='indeterminate')
progress_bar.pack_forget()

# Create a text area to display the extracted text
text_area = tk.Text(window)
progress_bar.pack_forget()

# Make the input boxes and text area adjust dynamically
window.bind('<Configure>', update_sizes)
query_entry.pack_propagate(False)
text_area.pack_propagate(False)

# Start the main event loop
window.mainloop()
