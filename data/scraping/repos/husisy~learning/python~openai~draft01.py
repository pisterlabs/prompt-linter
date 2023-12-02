# https://platform.openai.com/docs/tutorials/web-qa-embeddings
import os
import re
import json
import pickle
import collections
from bs4 import BeautifulSoup
import html.parser
import urllib
import numpy as np
from tqdm import tqdm
import tiktoken
import openai
import openai.embeddings_utils
import dotenv

dotenv.load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

if not os.path.exists('data'):
    os.makedirs('data')

class NaiveChatGPT:
    def __init__(self) -> None:
        self.message_list = [{"role": "system", "content": "You are a helpful assistant."},]
        self.response = None #for debug only

    def reset(self):
        self.message_list = self.message_list[:1]

    def chat(self, message='', tag_print=True, tag_return=False):
        message = str(message)
        if message: #skip if empty
            self.message_list.append({"role": "user", "content": str(message)})
            self.response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=self.message_list)
            tmp0 = self.response.choices[0].message.content
            self.message_list.append({"role": "assistant", "content": tmp0})
            if tag_print:
                print(tmp0)
            if tag_return:
                return tmp0

# TODO maybe not necessary
class ContextChatGPT:
    def __init__(self) -> None:
        self.message_list = [{"role": "system", "content": "You are a helpful assistant. Answer the question based on the context below, "
                              "and if the question can't be answered based on the context, say \"I don't know\""},]
        self.response = None

    def reset(self):
        self.message_list = self.message_list[:1]

    def set_context(self, context, use_gpt_reply=False):
        tmp0 = '\nAbove is some context, no need to reply or just acknowledge it with "..."'
        self.message_list.append({'role':'user', 'content': context+tmp0})
        if use_gpt_reply:
            self.response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=self.message_list)
            tmp0 = self.response.choices[0].message.content
            self.message_list.append({"role": "assistant", "content": tmp0})
        else:
            raise NotImplementedError

    def chat(self, message, tag_print=True, tag_return=False):
        message = str(message)
        if message: #skip if empty
            self.message_list.append({"role": "user", "content": str(message)})
            self.response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=self.message_list)
            tmp0 = self.response.choices[0].message.content
            self.message_list.append({"role": "assistant", "content": tmp0})
            if tag_print:
                print(tmp0)
            if tag_return:
                return tmp0

# chatgpt = ContextChatGPT()

chatgpt = NaiveChatGPT()

class HyperlinkParser(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.hyperlinks = []

    # Override the HTMLParser's handle_starttag method to get the hyperlinks
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)

        # If the tag is an anchor tag and it has an href attribute, add the href attribute to the list of hyperlinks
        if tag == "a" and "href" in attrs:
            self.hyperlinks.append(attrs["href"])

def get_domain_hyperlinks(domain, url):
    link_list = url
    ret = []
    for link in set(link_list):
        clean_link = None
        if re.search('^http[s]*://.+', link): #HTTP_URL_PATTERN
            url_obj = urllib.parse.urlparse(link)
            if url_obj.netloc == domain: #same domain
                clean_link = link
        else:
            if link.startswith("/"): #relative link
                link = link[1:]
            elif link.startswith("#") or link.startswith("mailto:"):
                continue
            clean_link = "https://" + domain + "/" + link
        if clean_link is not None:
            ret.append(clean_link[:-1] if clean_link.endswith("/") else clean_link)
    return list(set(ret))


def crawl_openai(filepath='data/openai_website.json'):
    if os.path.exists(filepath):
        with open(filepath, 'r') as fid:
            url_text_dict = json.load(fid)
    else:
        root_url = "https://openai.com/"
        domain = urllib.parse.urlparse(root_url).netloc
        url_queue = collections.deque([root_url])
        visited_url = set([root_url])

        url_text_list = []
        while url_queue:
            url = url_queue.pop()
            print(url) # for debugging and to see the progress
            try:
                with urllib.request.urlopen(url) as response:
                    if not response.info().get('Content-Type').startswith("text/html"):
                        continue
                    raw_text = response.read().decode('utf-8')
            except Exception as e:
                print(e)
                continue
            text = BeautifulSoup(raw_text, "html.parser").get_text() #remove tag
            if "You need to enable JavaScript to run this app." in text:
                print("Unable to parse page " + url + " due to JavaScript being required")

            url_text_list.append((url, text))

            parser = HyperlinkParser()
            parser.feed(raw_text)
            for link in get_domain_hyperlinks(domain, parser.hyperlinks):
                if link not in visited_url:
                    url_queue.append(link)
                    visited_url.add(link)
        url_text_dict = {x:y for x,y in url_text_list}
        with open(filepath, 'w') as fid:
            json.dump(url_text_dict, fid)
    ret = dict()
    for key,value in url_text_dict.items():
        # replace -, _, and #update with spaces
        tmp0 = key[19:].replace('-',' ').replace('_', ' ').replace('#update','')
        ret[tmp0] = re.sub(r'\s+', ' ', f'{tmp0}. {value}')
    ret = list(ret.items())
    return ret


def split_english_sentence_into_chunks(text, max_token, tokenizer):
    sentence_list = text.split('. ')
    num_token_list = [len(tokenizer.encode(" " + x)) for x in sentence_list]
    ret = []
    num_token_current = 0
    chunk = []
    for sentence, num_token in zip(sentence_list, num_token_list):
        if num_token_current + num_token > max_token:
            ret.append(". ".join(chunk) + ".")
            chunk = []
            num_token_current = 0
        if num_token > max_token:
            continue
        chunk.append(sentence)
        num_token_current = num_token_current + num_token + 1
    return ret


def get_chunked_text_embedding(text_list=None, max_tokens=None, tokenizer=None, filepath='data/openai_website_embedding.pkl'):
    # (ret0) chunked_text_list (list,tuple(text:str, num_token:int))
    # (ret1) embedding_np (np.float64, (N,1536))
    if os.path.exists(filepath):
        with open(filepath, 'rb') as fid:
            tmp0 = pickle.load(fid)
            chunked_text_list = tmp0['chunked_text_list']
            embedding_np = tmp0['embedding_np']
    else:
        assert text_list is not None
        chunked_text_list = [(y,len(tokenizer.encode(y))) for x in text_list for y in split_english_sentence_into_chunks(x, max_tokens, tokenizer)]
        embbeding_list = []
        # about 0.24USD
        for x,_ in tqdm(chunked_text_list):
            tmp0 = np.array(openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'], dtype=np.float64)
            embbeding_list.append(tmp0)
        embedding_np = np.stack(embbeding_list, axis=0)
        with open(filepath, 'wb') as fid:
            tmp0 = {'chunked_text_list':chunked_text_list, 'embedding_np': embedding_np}
            pickle.dump(tmp0, fid)
    return chunked_text_list, embedding_np


_openai_qa_template = ("Answer the question based on the context below, and if the question can't be answered based on the context, "
            "say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:")

# TODO replace with chatgpt (which is much cheaper)
def answer_question(question, text_list, embedding_np, model="gpt-3.5-turbo", max_context_len=1800, tag_print_context=False, response_max_tokens=150):
    """
    Answer a question based on the most similar context from the dataframe texts
    text_list(list, tuple(text:str, num_token:int))
    """
    assert model in {'text-davinci-003', 'gpt-3.5-turbo'}
    assert all(len(x)==2 for x in text_list)
    assert len(text_list) == embedding_np.shape[0]
    text_len_list = np.array([x[1] for x in text_list])
    text_str_list = [x[0] for x in text_list]

    q_embedding = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']
    distance = np.array(openai.embeddings_utils.distances_from_embeddings(q_embedding, embedding_np, distance_metric='cosine')) # 0: cloest
    ind0 = np.argsort(distance)
    tmp0 = np.nonzero((text_len_list[ind0] + 4).cumsum() > max_context_len)[0].min()
    context_text_list = [text_str_list[x] for x in ind0[:tmp0]]
    context_text = "\n\n###\n\n".join(context_text_list)
    if tag_print_context:
        print(f"Context:\n{context_text}\n\n")
    prompt = _openai_qa_template.format(context=context_text, question=question)
    try:
        if model=='text-davinci-003':
            response = openai.Completion.create(prompt=prompt, temperature=0, max_tokens=response_max_tokens,
                top_p=1, frequency_penalty=0, presence_penalty=0, stop=None, model=model)
            ret = response["choices"][0]["text"].strip()
        else:
            chatgpt.reset()
            ret = chatgpt.chat(prompt, tag_print=False, tag_return=True)
    except Exception as e:
        print(e)
        ret = ""
    return ret


# Load the cl100k_base tokenizer which is designed to work with the ada-002 model
tokenizer = tiktoken.get_encoding("cl100k_base")

max_tokens = 500
url_text_list = crawl_openai()
tmp0 = [x[1] for x in url_text_list]
chunked_text_list, embedding_np = get_chunked_text_embedding(tmp0, max_tokens, tokenizer) #about 0.24USD


question = "Am I allowed to publish model outputs to Twitter, without a human review?"
#(davinci) 'No, you are not allowed to publish model outputs to Twitter without a human review. Manually review each generation before sharing or while streaming is required.'
#(chat) 'No, according to the sharing and publication policy mentioned in the context, it is required to manually review each generation before sharing to mitigate the possible risks of AI-generated content.'
question = "What day is it?"
#(davinci) "I don't know."
#(chat) "I don't know. The context does not provide enough information to answer the question."
question = "What is our newest embeddings model?"
#(davinci) 'The newest embeddings model is text-embedding-ada-002.'
#(chat) 'The newest embeddings model is text-embedding-ada-002.'
answer_question(question, chunked_text_list, embedding_np, tag_print_context=False)
