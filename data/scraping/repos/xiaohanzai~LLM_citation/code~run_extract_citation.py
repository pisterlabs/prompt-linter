import openai
import os
import glob
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.getenv('OPENAI_API_KEY')

# it's actually totally shit letting it output json format......
template_prompt = '''
Extract the references in the text below, and summarize in one sentence the reason why each reference is cited. Here is an example input and output:

Text:
Lorimer et al. (2013) forecast that widefield radio interferometers that are presently coming online could detect tens per day of these events. In addition, several schemes have been proposed for measuring DM towards time-steady astrophysical sources (Lieu & Duan 2013;Lovelace & Richards 2013).

Output:
Lorimer et al. (2013):
Referenced to support the forecast that widefield radio interferometers coming online could detect a significant number of these events, emphasizing the potential impact of future technology.

Lieu & Duan 2013; Lovelace & Richards 2013:
These papers are cited to highlight proposed schemes for measuring dispersion measure (DM) towards time-steady astrophysical sources, which is relevant to the discussion of DM measurements in the context of highly dispersed radio bursts.

Now extract the citations in the text below. If no citations appear in the text, print "N/A". Ignore references of the sections of the current paper. Expand the abbreviations used in the input text.

Text:
```{input_text}```
'''

# 1500 or 2000 can split most texts into paragraphs; 2000 seems slightly better
# that regex is meant to split sentences while ignoring et al.
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=300,
    separators=["\n\n", "\n", "(?<=(?<!et\sal\.\s)\.\s)", " ", ""]
)

# define model etc.
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
prompt = ChatPromptTemplate.from_template(template_prompt)
chain = LLMChain(llm=llm, prompt=prompt)

# get all intro texts
fnames = glob.glob('../data/2*.pickle')

# in reality this can't be run in one go because openai may get stuck at some point
# I ran it in batches of 20 in a jupyter notebook...
for i in range(len(fnames)):
    fname = fnames[i]

    with open(fname, 'rb') as f:
        data = pickle.load(f)

    # split text
    text = data['text']['text']
    docs = r_splitter.split_text(text)

    fname_out = '../data/' + fname[fname.rindex('/'):-6] + 'txt'
    with open(fname_out, 'w') as f:
        for j in range(len(docs)):
            # get response from gpt
            response = chain.run(docs[j])
            f.write(response + '\n')
