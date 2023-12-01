# Note: This version I designed for me! I will update a public version later
### 1.Import all neccesary libaries
from tkinter import*
from tkinter.scrolledtext import ScrolledText
from PyPDF2 import PdfReader,generic
from tkinter import filedialog
from builder1 import process_text
import networkx as nx
import re
import os
import openai
import socket
import nltk
from nltk.stem import  PorterStemmer
work_dir = os.getcwd()

# Set global variable for number of sentences and keyphases to extract when internet is not available
n, k = 5, 5
# Set global variable for number of sentences and keyphases to extract when internet is available
n_sentences, ks = 36, 5
nw = 100
### 2. Write all necessary functions
## 2.1 This fuction creates a window with desired title, color, and size
def create_window(title, color, w, h):
    # Creat a window
    wd = Tk()
    # Write a title of the window
    wd.title("Summary for Long Document and Fun Chat  ")
    # Set the minimum size of the window when window appears
    wd.minsize(width = w, height = h)
    # Set the background color for the window
    wd.configure(bg = color)
    return wd


## 2.2 This fuction creates a cavas
## parameters: window: what window the canvas will be put in, color, width, height,
## x: how many pixels from the left of the window,  y: how many pixels from the top of the window
def create_canvas(window, color, w, h, x, y):
    c = Canvas(window, bg = color, width = w, height = h)
    c.place(x = x, y = y)
    return c


## 2.3 This function creates a textbox with scroll bar
## parameters: width,height,x: how many pixels from the left of the window,  y: how many pixels from the top of the window
# wchar:number characters can be inserted for each row,hchar: number characters can be inserted for each column
def scroll_text(w, h, x, y, wchar, hchar):
    #Create a frame in the window
    frame= Frame(window, width = w, height = h)
    frame.place(x = x, y = y)
    text_box = ScrolledText(frame, width=wchar, height=hchar)
    text_box.pack(fill=BOTH, expand=1)
    return text_box


## 2.4 This function gets the text from a PDF file
def pdf2text(pdf_file):
    reader = PdfReader(pdf_file)
    n_pages = len(reader.pages)
    print(f" number of pages: {n_pages}")
    lp = extract_chapter_pdf(reader)
    text = extract_text(0, n_pages, reader)
    if lp == []: lp = get_chapters_text(text)
    if text =='':
        out_box.insert(END,"\nSystem: ",'tag2')
        out_box.insert(END, "The uploaded PDF file cannot be converted to text. Please upload another file!  ")

    return n_pages, lp, text


## 2.5 This function gets the text from a text file
def ftext2text(file):
    with open(file, 'r',encoding = "utf-8") as f:
        text = f.read()
    return text


# 4.5 This function uploads a file ( PDF or txt) and returns the absolute path of a file.
# Parameter: none
# Return: file path
def upload_file():
    try:
        fname = filedialog.askopenfilename(filetypes = [('PDF Files', '*.pdf'), ('Text Files', '*.txt')])
        return fname
    except:
        OSError
        pass

# 4.6 This function gets a user's question and feeds that to chat_API() function
# then gets the answer and insert it to out_box.
# Parameter: Click the Enter key on the  keyboard after finishing enter a question.
# Return: none
def enter(event):
    u_text = out_box.get("1.0", "end-1c")
    if is_key_here() and is_on():
        s_text = chat_API(u_text)
    else:
        s_text = " Sorry, currently chat function only works on API mode!"
    out_box.insert(END, "\nSystem: ", 'tag2')
    out_box.insert(END, s_text)
    out_box.insert(END, "\nTam: ", 'tag1')

def getkey(event):
    text = key_box.get('1.0', "end-1c")
    key = text[-51:]
    key = key.strip()
    key = key.replace(" ","")
    key = key.replace("\n", "")
    if testkey(key):
        if os.path.exists(work_dir + "\\key.txt"): os.remove(work_dir + "\\key.txt")
        text2file(key, work_dir + "\\key.txt")
        insert_keybox("The system got the API key! Please: 1.Type an OpenAI API model's name(ex: gpt-4,gpt-3.5-turbo,...) into this box 2. Click 'Return' key:")
    else:
        insert_keybox("Please enter a working OpenAI API key and click 'Right arrow' key!")

def set_api_model(event):
    text = key_box.get('1.0', "end-1c")
    model = text[135:]
    model = model.strip()
    model = model.replace(" ","")
    if testmodel(model):
        text2file(model, work_dir + "\\model.txt")
        insert_keybox(f"The system is working on {model}. System is in API mode! ")
        return model
    else:
        f"The  entered model has problem. The system will use GPT 4 model. System is in API mode!"

# 4.10 This function checks if an entered model  works or not
# Parameter: model
# Return: either True or False.
def testmodel(model):
    try:
        openai.api_key = ftext2text(work_dir + "\\key.txt")
        ans = openai.ChatCompletion.create(
            model = f"{model}",
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "How are you?"}
            ]
        )
        if ans.choices[0].message['content'] != "": return True
    except:
        OSError
    pass
    return False

def testkey(key):
    try:
        openai.api_key = key
        ans = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Who is president of US?"}
            ]
        )
        # print(f" reponse: {ans.choices[0].message['content']}")
        if  ans.choices[0].message['content'] != "": return True
    except: OSError
    pass
    return False


def text2file(text,file):
    with open(file, 'w',encoding = "utf-8") as f:
         f.write(text)


## 2.8 This function gets the input text from the file
def get_textFfile(out_box):
    #Clean the output textbox
    if count_words(out_box) > 1000: out_box.delete("1.0", END)
    # Get the absolute path of the uploading file
    fname = upload_file()
    title = os.path.basename(fname)[:-4]
    if fname == "": pass
    else:
        # If the uploading file is pdf, convert to a string
        # title = get_info(fname)
        if is_pdf(fname):
            # convert pdf file to text
            n_pages, lp, text = pdf2text(fname)
            out_text = ""
            # If the file can be extracted by list of chapters(sections) and number of pages great than 100
            if n_pages > 100 and lp != [] :
                #API mode
                if is_on() and is_key_here():
                    out_box.insert(END, "\nSystem: ", 'tag2')
                    out_box.insert(END, f"{user_know()} ", 'tag3')
                    out_box.insert(END, f"Summary and keywords for each chapter of the uploaded {n_pages}-page document ", 'tag4')
                    out_box.insert(END, f"'{title}':", 'tag5')
                    out_box.see(END)
                    # For each chapter
                    for e in lp:
                        # Call the function clean_text() to clean each chapter's text
                        a, c, chap = clean_text(e[1])
                            # Feeds each chapter's clean text to TextStar and get more  than 30 sentences
                        gM, gk = get_n_sents(chap, n_sentences, get_kwords_box(''))
                        # Pass the summary result of TextStar to API
                        # and set the output's length (100 tokens)
                        r_chap, kchap= API_sum_keywords(gM, get_sents_box(''),get_kwords_box(''))
                        # sents = r_chap.split('.')
                        # s_chap = ".".join(sents[:-1]).replace('\n',' ')
                        e0= e[0].replace('\n',' ')
                        out_box.insert(END, f"\n{e0}: ", 'tag4')
                        out_box.insert(END,r_chap)
                        out_box.insert(END, f"\nKeywords of this chapter: ", 'tag5')
                        out_box.insert(END,kchap)
                        out_box.see(END)
                        window.update()
                        # Concatenate all chapters' summary from API
                        out_text += r_chap
                    sumbook, kbook = API_sum_keywords(out_text, get_sents_box(''), get_kwords_box(''))
                    out_box.insert(END, "\nSystem: ", 'tag2')
                    out_box.insert(END, f"{user_know()} ", 'tag3')
                    out_box.insert(END, f"\nSummary for the whole uploaded {n_pages}-page document ", 'tag4')
                    out_box.insert(END, f"'{title}':", 'tag5')
                    out_box.insert(END, f"\n{sumbook}")
                    out_box.insert(END, f"\nKeywords for whole document: ", 'tag5')
                    out_box.insert(END, f"{kbook}")
                    out_box.insert(END, "\nTam: ", 'tag1')
                    out_box.see(END)
                    window.update()
                # Local mode
                else:
                    sumbook, gk = get_n_sents(text, get_sents_box(''), get_kwords_box(''))
                    # print(f"kw {gk}")
                    sumbook = sumbook.replace('\n',' ')
                    kwb = order_keywords(gk)
                    out_box.insert(END, "\nSystem: ", 'tag2')
                    out_box.insert(END, f"{user_know()} ", 'tag3')
                    out_box.insert(END, f"\n\nSummary for the whole uploaded {n_pages}-page document ", 'tag4')
                    out_box.insert(END, f"'{title}':", 'tag5')
                    out_box.insert(END, f"\n{sumbook}")
                    out_box.insert(END, f"\n\nKeywords for whole document: ", 'tag4')
                    out_box.insert(END, f"\n{kwb}")
                    out_box.insert(END, f"\n\nSummary and keywords for each chapter of the uploaded document: ", 'tag5')
                    out_box.see(END)
                    window.update()
                    # Feeds each chapter's clean text to graph algorithm and get small number sentences
                    for e in lp:
                        # Call the function clean_text() to clean each chapter's text
                        a, c, chap = clean_text(e[1])
                        # print(f" clean chapter{e[0]}: {chap}")
                        gM, gk = get_n_sents(chap, get_sents_box(''), get_kwords_box(''))
                        kwc = order_keywords(gk)
                        # print(gM)
                        # Concatenate all chapters' summary form TextStar
                        # out_text += '\n' + e[0] + gM
                        e0 = e[0].replace('\n', ' ')
                        # out_box.insert(END, f"\n\nSummary for chapter  ", 'tag4')
                        out_box.insert(END, f"\n\n{e0}: ",'tag4' )
                        out_box.insert(END, gM)
                        out_box.insert(END, f"\nKeywords: ", 'tag4')
                        out_box.insert(END,kwc)
                        out_box.see(END)
                        window.update()
                    out_box.insert(END, "\nTam: ", 'tag1')
                    out_box.see(END)
            else:
                # Call function paper2out(text) to process the text which was extracted from PDF file.
                nsa, a, c, summary, keywords = paper2out(text)
                # If the system is working in a "Local mode" and a is not an empty string
                # and the desired number of summary's sentences less than number sentences in a.
                if a != "" and user_know() == "Local mode!" and get_sents_box("") < int(nsa):
                    # Insert title and a to out_box.
                    insert_outbox_article(title, a, "by author(s)", n_pages,keywords)
                else:
                    # Insert title and summary to out_box.
                    insert_outbox_article(title, summary, "", n_pages,keywords)
        # If the uploading file is txt
        if is_txt(fname):
            #convert the file to a string
            text = ftext2text(fname)
            #Call function paper2out(text) to process the input string
            nsa, a, c, summary,keywords = paper2out(text)
            if is_on() and is_key_here():
                insert_outbox_article("", summary, "", "",keywords)
            # If  "Local mode" is working and a is not an empty string
            # and the desired number of summary's sentences less than number sentences in a.
            elif a != "" and user_know() == "Local mode!" and get_sents_box("") < nsa:
                insert_outbox_article("", a, " by author(s)", "",keywords)
                out_box.insert(END, "\nTam: ", 'tag1')
            else:
                insert_outbox_article("", summary, "", "",keywords)

def order_keywords(keywords):
    kwb = ''
    for w in keywords:
        kwb += w + '; '
    return kwb

# 4.13 This function inserts an article's summary  to the out_box.
# Parameter: title of an uploading document, summary of the document, a string "by author(s)", total pages of an uploading document.
# Return: none
def insert_outbox_article(title, summary, aut, n,keywords):
    out_box.insert(END, "\nSystem: ", 'tag2')
    out_box.insert(END, f"{user_know()} ", 'tag3')
    out_box.insert(END, f"\nSummary of the uploaded {n}-page document ", 'tag4')
    out_box.insert(END, f"'{title}'", 'tag5')
    out_box.insert(END,f" {aut}:\n", 'tag4')
    out_box.insert(END, summary)
    out_box.insert(END, f"\nKeywords of the uploaded document: \n", 'tag4')
    out_box.insert(END, keywords)
    out_box.insert(END, "\nTam: ", 'tag1')
    out_box.see(END)

# 4.14 This function inserts a book's summary  to the out_box.
# Parameter: the title of an uploading book, a summary for the whole book , each chapter's summary,total pages of an uploading document.
# Return: none
def insert_outbox_book(title, sumbook, out_text, p):
    out_box.insert(END, "\nSystem: ", 'tag2')
    out_box.insert(END, f"{user_know()} ", 'tag3')
    out_box.insert(END, f"Summary of the uploaded {p}-page document ", 'tag4')
    out_box.insert(END, f"'{title}':", 'tag5')
    out_box.insert(END, f" \n{sumbook}")
    out_box.insert(END, f"\n\nSummary for each chapter of the uploaded {p}-page document ", 'tag4')
    out_box.insert(END, f"'{title}':", 'tag5')
    out_box.insert(END, f" \n{out_text}")
    out_box.insert(END, "\nTam: ", 'tag1')
    out_box.see(END)



# 4.15 This function checks which mode the system is currently working.
# Parameter: none
# Return: either "API mode!" or "Local mode!"
def user_know():
    if is_on() and is_key_here():
        return "API mode!"
    else:
        return "Local mode!"
## 2.9 Function get n sentences from a long document by TextStar (a graph based algorithm)
## parameters: text:input text, n: the number semtences of an output summary, k: the number key words of an output
def get_n_sents(text, n, k):
    sents, kwds = process_text(text=text, ranker=nx.degree_centrality, sumsize=n,
                               kwsize=k, trim=80)
    summary = ""
    # A sent is a list of tuples, each tuple is a pair of sentence id and sentence
    for sent in sents:
        # Extract only the sentence and convert tuple to string
        s = str(sent[1])
        summary += " " + s[0].upper() + s[1:]
    porter = PorterStemmer()
    l = []
    li = []
    for i in range(len(kwds)):
        for k in range(len(kwds)):
            if i != k and (
                    porter.stem(kwds[i]) == porter.stem(kwds[k]) or porter.stem(kwds[i])[:-1] == porter.stem(kwds[k])):
                if (i, k) not in li and (k, i) not in li:
                    li.append((i, k))
                    l.append((kwds[i], kwds[k]))

    print(f"l is {l}")
    # print(f" li is {li}")

    for t in l:

        if t[0] in summary and t[1] in kwds:
            kwds.remove(t[1])
        elif t[1] in summary and t[0] in kwds:
            kwds.remove(t[0])
        else:
            kwds.remove(t[0])
    kz = kwds[:get_kwords_box('')]


    return summary, kz


 # 4.16 This function cleans text.
# Parameter: a string.
# Return: either an abstract or an empty string,either a conclusion or an empty string,
#        a string without an abstract and a conclusion.
def clean_text(text):
    # Replace strange symbols which are not in [a-zA-Z0-9.!?:,{}()@$\n ] between character f and t by -.
    text = re.sub('f[^\w.!?:,{}%\[\]()@$/\n ]t', 'f-t', text)
    # Remove all strange synbols which are not in [a-zA-Z0-9.!?:,{}()@$\n -].
    text = re.sub('[^\w.!?:,{}%()\[\]@$=/~\n -]', ' ', text)  # this line code does not  result some missing words
    # Remove References and all text appears after that by calling remove_references(text) function.
    # text = re.sub('(\s%\s)+', ' ', text)
    text = remove_references(text)
    # Call  function get_Abstract(text) to get an abstract and a text without abstract.
    ab, text = get_Abstract(text)
    # Remove Algorithm format if they exist.
    text = re.sub('\nAlcgorithm.*?end(\n\d.*?end)+', ' ', text, flags=re.DOTALL)
    # Remove Figure if they exist.
    text = re.sub('\n[Ff]igure.*?\n', '\n', text, flags=re.DOTALL)
    # Remove Table if they exist.
    text = re.sub('\n[Tt]able.*?\.', ' ', text, flags=re.DOTALL)
    # Remove ROUGE if they exist.
    text = re.sub('ROUGE-[\dL]', ' ', text, flags=re.DOTALL)
    # Remove F1@5 if they exist.
    text = re.sub('F1@[\d]*', ' ', text, flags=re.DOTALL)
    # Remove text in between 2 round brackets if they exist.
    text = re.sub('\([^)]*\)', ' ', text, flags=re.DOTALL)
    # Remove text in between 2 square brackets if they exist.
    text = re.sub('\[.*\]', ' ', text, )
    # Remove https if they exist.
    text = re.sub('\d?http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # Remove string after forward slash (/) which usually is converted from a table from pdf if they exist.
    text = re.sub('\n?[/].+\n', ' ', text)
    # Remove 1 or more white space before a comma.
    text = re.sub('[\s]+,', ',', text)
    # Remove a string of number which usually is converted from a table from pdf.
    text = re.sub('\n^\d+\.?\d*$\n', '', text)
    # Remove the lines with a lot numbers which often come from tableges.
    text = re.sub('\n?[\w\s-]*(\s*?[\d]+[.][\d]+-?)+', ' ', text)
    # Replace '..' by '.'.
    text = re.sub('\.\.', '.', text)

    # Call a function get_Conclusion(text) to get a conclusion.
    c = get_Conclusion(text)
    # Replace all newlines by ' '.
    text = re.sub('\n', ' ', text)
    return ab, c, text
# 4.17  This function checks if the internet is connected or not.
# Parameter: none.
# Return: either True or False.
def is_internet():
    try:
        # Try to connect to "www.google.com" at port 443
        s = socket.create_connection(("www.google.com", 443))
        if s != None: return True
    except OSError:
        pass
    insert_keybox("Hello Tam! I am currently in the local mode! ")
    insert_sents_box("Number of summary's sentences:")
    return False


def API_sum_keywords(text, m, n):
    if os.path.exists(work_dir+ "\\model.txt"): run_model = ftext2text(work_dir+ "\\model.txt")
    else: run_model = "gpt-4"
    openai.api_key = ftext2text(work_dir + "\\key.txt")
    s_answer = openai.ChatCompletion.create(
        # model = "gpt-3.5-turbo-16k",
        model = "gpt-4",
        # max_tokens = m,
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Summary this document {text} to {m} words and get {n} keywords "}
        ]
       )
    r = s_answer.choices[0].message['content']
    print(r)
    s = r.split('Keywords', 1)[0]
    s = s.replace('Summary:','')
    s = s.replace('\n','')
    k = r.split('Keywords', 1)[1]
    k = k[1:]
    k = k.replace('\n', '')
    return s, k

# 4.19 This function connects to OpenAI API model.
# Parameter: a prompt.
# Return: either an answer for a prompt or an error message.
def chat_API(uq):
    if os.path.exists(work_dir+ "\\model.txt"): run_model = ftext2text(work_dir+ "\\model.txt")
    else: run_model = "gpt-4"
    openai.api_key = ftext2text(work_dir + "\\key.txt")
    answer = openai.ChatCompletion.create(
        model = run_model,
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{uq}"}
        ]
    )
    return answer.choices[0].message['content']


def get_info(pdf_file):
    r = PdfReader(pdf_file)
    t = '"' + get_title(r) + '"'
    return t

def get_title(r):
    # text = ""
        i = 0
        text = ""
        while text == "":
            try:
                p = r.pages[i]
                text = p.extract_text()
                i += 1
            # print(f"page 0: {text} ")
                lns = text.split('\n')
                # print(lns[0])
                return lns[0]
            except OSError:
                pass
                return ''


def get_author():
    pass
def get_year():
    pass


## 2.13 This function gets an Abstract and remove author's information
## Parameter: text
## Return: an Abstract and text without an Abstract if an Abstract exists
# #       or [] and text
def get_Abstract(text):
    # Find the word 'Abstract'or ABSTRACT'
    # The first letter must be uppercase.
    # If use flags=re.IGNORECASE here the result will return 'abstract'
    abs = re.findall('\nAbstract|ABSTRACT', text)
    # print(f" Result re.findall(Abstract) :\n {abs}")
    # Find the word 'Introduction' or 'INTRODUCTION'
    intr = re.findall('Introduction|INTRODUCTION', text)
    # print(f"intr:n\{intr} ")
    # If the text contains the word 'Abstract',no matter lowercase or uppercase
    if abs != []:
        # Split the text into 2 parts by the key word 'Abstract'or 'ABSTRACT',
        #The word 'abstract' can be somewhere in body text or reference, so it causes a wrong spliting position
        # Get the second part for text which does not have author's information
        text_no_author = text.split(str(abs[0]), 1)[1]
        # Split the text into 2 parts by the key word 'Introduction' or 'INTRODUCTION',
        # Get the first part which is abstract
        # if len(intr) > 0:
        abstract1 = text_no_author.split(str(intr[0]), 1)[0]
        # print(f"abstract1 : \n {abstract1}")
        abstract1 = re.sub('\n1\.?', '', abstract1)
        abstract1 = re.sub('\n',' ',abstract1)
        # if abstract1[-1] == '1': abstract1 = abstract1[:-1]
        # print(f"abstract1[-1]: {abstract1[-1]}")
        # print(f"abstract1 : \n {abstract1}")
        # Repeat len(intr) times to split the text and get the text after the word "Introduction".
        for i in range(len(intr)):text_no_author = text_no_author.split(str(intr[-1]), 1)[1]
        # Find the word 'KEYWORDS' or 'keywords'
        k = re.findall('KEYWORDS|Keywords', abstract1)
        #If the abstract1 contains the word 'KEYWORDS' or 'keywords'
        if k != []:
            #Split the text into 2 parts by the key word 'KEYWORDS' or 'Keywords',
            # Get the first part for abstract1 which does not have key words part
            abstract2 = abstract1.split(str(k[-1]),1)[0]
            abstract2 = re.sub('\.\s?[1]($|\n)','.',abstract2)
            # if not abstract2.endswith('.'): abstract2 = abstract2 + '.'
            # print(f"abstract2: {abstract2}")
            return abstract2, text_no_author
        # If the abstract1 does not contains the word 'KEYWORDS' or 'Keywords'
        else: return abstract1, text_no_author
    else: return '', text


## 2.14 This function removes references
## Parameters: text
## Return: text without References
def remove_references(text):
    # Find the word 'References or REFERENCES'
    r = re.findall('References|REFERENCES', text)
    # Find the word 'Acknowledgements' or 'ACKNOWLEDGEMENTS'
    a = re.findall('Acknowledgements|ACKNOWLEDGEMENTS', text)
    # If the word 'References or REFERENCES' exists
    if r != []:
        # If the word 'Acknowledgements' or 'ACKNOWLEDGEMENTS' exists
        if a != []: text = text.split(str(a[-1]),1)[0]
        # If the word 'Acknowledgements' or 'ACKNOWLEDGEMENTS' does not exists
        else:text = text.split(r[-1],1)[0]
    return text


##2.15 This function gets the Conclusion
## Parameters: text
## Return: a Conclusion or ""
def get_Conclusion(text):
    # Find the word 'Conclusion' or 'CONCLUSION'
    c = re.findall('Conclusion|CONCLUSION', text)
    # print(f"\nc:{c}")
    # If the word 'Conclusion' or 'CONCLUSION'  exists
    if c != []:
        # Split the text into 2 parts by key word 'Conclusion' or 'CONCLUSION'
        # Get the second part
        conc = text.split(str(c[-1]), 1)[1]
        # print(f" conc after split by the word conclusion: {conc}")
        return conc
    else:return ""


## 2.16 This function gets page number
## Parameters: n1 (id number), n2(generation number),r(PdfReader)
## Return: a page number
def get_page_num(n1, n2, r):
    return r._get_page_number_by_indirect(generic.IndirectObject(n1, n2, r))


## 2.17 This function extracts text from PDF file from page pm to page pn
## Parameters: pm(the page which begin to extract text), pn( stop to extract text after this page), r(PdfReader)
## Return: text which get from PdfReader
def extract_text(pm, pn, r):
    # Define a variable text with an empty string
    text = ""
    # Loop from  page pm to  page pn
    for p in range(pm, pn):
        # Call the function pages from PdfReader class
        page = r.pages[p]
        # Concatenate extracting text from each page to a variable text
        text += page.extract_text()
    return text


## 2.18 This function gets chapters'/sections' text directly from PDF file.
## Parameter: r(PdfReader)
## Return: Either an empty list or a  list of pairs of chapters'(sections') number and their correlating text.
def extract_chapter_pdf(r):
    # Create an empty list l which will store none list type elements of outline.
    l = []
    #  get outline  from PDF format
    outline = r._get_outline()
    # print(f" outline from pdf: {outline}")
    # print(f"evaluation_output type of outline: {type(outline)}")
    # If the extracting outline  is empty, then return an empty list
    if outline == []: return []
    #If the extracting outline  is not empty
    else:
        # Loop through all elements in outline
        for o in outline:
            # If type of the element is not list, then put that element's title and page  to the list l
            if type(o) != list:
                l.append((o.title,o.page))
                # print(f"o,tittle {o.title} o.page {o.page}")
        # print(f" o in outline and o not a list : {l}")
        # l = [(o.title, o.page) for o in outline if type(o) != list]
        # Check if the word "Chapter"  is in the list l or not
        # find = re.findall("Chapter|CHAPTER", str(l))
        # print(find)
        # If the word "Chapter|CHAPTER" is in the list l
        if (re.findall("Chapter|CHAPTER", str(l))!= []):
            # Create an empty list lc which will contain tuples of chapter and the starting page number of the cahpter.
            lc = []
            # Create an empty list li which will contain indexes  of l where store elements which include either  the word "Chapter" or "CHAPTER".
            li = []
            # Loop through all elements in the list l.
            for i in l:
                # If the word "Chapter"  is in an element i:
                if "Chapter" in i[0] or "CHAPTER" in i[0]:
                    # Append a pair of chapter and a correlating the starting page number of the chapter to the list lc.
                    lc.append((i[0],get_page_num(i[1].idnum,i[1].generation,r)))
                    # print(f"get_page_num(i[1].idnum,i[1].generation,r):{get_page_num(i[1].idnum,i[1].generation,r)}")
                    # Put the index of element i to the list li.
                    li.append(l.index(i))
            # print(f"li: {li}")
            # Append the element which is located next to the last chapter in a list l and its starting page number to the list lc
            lc.append((l[li[-1]+1][0],get_page_num(l[li[-1]+1][1].idnum, l[li[-1]+1][1].generation, r)))
            # Create an empty list lp which will contain pairs of chapter number and text in correlated chapter.
            lp = []
            # Put  tuples of chapter number and correlating text to the list lp.
            for e in range(len(lc)-1):lp.append((lc[e][0],extract_text(lc[e][1], lc[e+1][1], r)))
            # print(f"lp: {lp}")
            return lp
        # if the word "Conclusion|CONCLUSION" in l
        elif(re.findall("Conclusion|CONCLUSION", str(l)) != []):
                # Create an empty list l_sec, which will contain pairs of section's number and  correlated the starting page number.
            l_sec = []
            # Loop through the list l
            for i in l:
                # If an element contains a word "Conclusion" or "CONCLUSION"
                if "Conclusion" in i[0] or "CONCLUSION" in i[0]:
                    # Get the index of the list l where stores the element that includes either "Conclusion" or "CONCLUSION"
                    con_i = l.index(i)
                    # print(f" con_i: {con_i}")
            # Create a list l_s such that l_s contains elements in l
            # from index 0 to the index next to the index of element which has the word "Conclusion".
            l_s = l[:con_i+2]
            # print(f"l_s: {l_s}")
            # Loop through l_s.
            for s in range(len(l_s)):
                # Append a pair (section,a starting page number of correlated section) into the list l_sec
                l_sec.append((l_s[s][0], get_page_num(l_s[s][1].idnum, l_s[s][1].generation, r )))
            # print(f"l_sec: {l_sec}")
            # Create an empty list, which will contain pairs of section's number and correlated text.
            l_sec_text= []
            # Put  tuples of section's number and correlating text to the list l_sec_text
            for s1 in range(len(l_sec)-1): l_sec_text.append((l_sec[s1][0],extract_text(l_sec[s1][1], l_sec[s1+1][1], r)))
            # print(f"l_sec_text: {l_sec_text}")
            # print(f"len of l_sec_text: {len(l_sec_text)}")
            return l_sec_text
        # if either "Bibliography" or "BIBLIOGRAPHY" in the list l. Example NLP book
        elif(re.findall("Bibliography|BIBLIOGRAPHY", str(l)) != []):


            # Create an empty list,which will contain pairs of section and the correlated starting page number.
            l2 = []
            # Loop through the list l
            for i in l:
                # If an element contains a word "Bibliography|BIBLIOGRAPHY",
                if "Bibliography" in i[0] or "BIBLIOGRAPHY" in i[0]:
                    # Get the index of the list l where stores that element.
                    bib_i = l.index(i)
                    # print(f" con_i: {con_i}")
            # Create a list l_b such that l_b contains elements in l
            # from index 0 to the index next to the index of element which has the word "Bibliography|BIBLIOGRAPHY"
            l_b = l[:bib_i + 1]
            # print(f"l_b: {l_b}")
            # Loop through l_b
            for s in range(len(l_b)):
                # Append a pair (section's number,the correlated starting page number) into the list l_sec
                l2.append((l_b[s][0], get_page_num(l_b[s][1].idnum, l_b[s][1].generation, r)))
            # print(f"l_sec: {l_sec}")
            # Create an empty list, which will contain pairs of section's number and correlated text.
            l_text = []
            # Put  tuples of section's  and correlating text to the list l_sec_text.
            for s1 in range(len(l2) - 1): l_text.append(
                (l2[s1][0], extract_text(l2[s1][1], l2[s1 + 1][1], r)))
            # print(f"l_sec_text bio case: {l_text[5]}")
            # print(f"len of l_sec_text bio case: {len(l_text)}")
            return l_text
        else:return []


## This function gets chapters'/sections' text from a given input text.
## Parameter: text
## Return: Either an  empty list or a list of pairs, where each pair contains of a chapters'/sections' number and its correlated text.
def get_chapters_text(text):
    # Find a line which begins with a word "Chapter" follow by digit then end of line
    l = re.findall("\n[C][Hh][Aa][Pp][Tt][Ee][Rr]\s*[\d]+\n", text)
    # print(f" l chapters: {l}")
    # Find a line which begins with a word "Appendix"
    a = re.findall("\n[A][Pp][Pp][Ee][Nn][Dd][Ii][Xx]\s*[A-Za-z0-9]\n", text)
    # A line which begins with a word "Chapter" exists
    if l != []:
        lc = []
        # Generate index of l from 0 to second last index
        for i in range(len(l) - 1):
            # Append tuple( chapter, correlating text) to a list lc
            lc.append((l[i], text[(text.index(l[i]) + len(l[i])):text.index(l[i + 1])]))
        # print(f" index :{text.index(l[0])}")
        # print(f" index text: {text[(text.index(l[0]) + len(l[0])):text.index(l[ 1])]}")
        # Append the element which is located next to the last chapter in a list l and its page number to the list lc
        lc.append((l[-1], text[(text.index(l[-1]) + len(l[0])):text.index(a[0])]))
        # print(f"len lc: {len(lc)}")
        # print(f"lc: {lc[-1]}")
        return lc
    else:
        # Find the word "Content" or "CONTENT"
        c = re.finditer("C[Oo][Nn][Tt][Ee][Nn][Tt][Ss]", text)
        # If either the word "Content" or "CONTENT" exist:
        if c != []:
            # Create an empty list
            lc = []
            for i in c: lc.append((i.start(), i.end(), i.group(0)))
            ap = re.finditer('A[pP][Pp][eE][Nn][Dd][Ii][Xx]\s?[AI1]\s?.+\n', text)
            lap = []
            for i in ap: lap.append((i.start(), i.end(), i.group(0)))
            while lc != [] and len(lap)> 1 and lc[0][0] > lap[0][0]:
                lap.pop(0)
            if len(lc) > 0 and len(lc[0]) > 0 and len(lap) > 0 and len(lap[0]) > 0:
                contents = text[lc[0][0]:lap[0][0]]
                text_no_content = text[lap[0][0]:]
                contents = re.sub('\n[\.\s]+', '\n', contents)
                chs = re.findall("F", contents)
                li = []
                for n in range(len(chs)):
                    if "Contents" in chs[n] or "CONTENTS" in chs[n][2]: li.append(n)
                for m in li: chs.pop(m)
                lch_pos = []
                for ch in range(len(chs)):
                    if re.findall(f"{chs[ch]}", text_no_content, flags=re.IGNORECASE) != []:
                        for t1 in re.finditer(f"{chs[ch]}", text_no_content, flags=re.IGNORECASE):
                            lch_pos.append((t1.start(), t1.end(), t1.group(0)))
                    else:
                        text1 = chs[ch]
                        text1 = re.sub('\s+', '', text1)
                        ft = re.finditer(f"{text1}", text_no_content, flags=re.IGNORECASE)
                        for i, t2 in enumerate(ft):
                            if i == 0: lch_pos.append((t2.start(), t2.end(), t2.group(0)))
                lp_text = []
                for p in range(len(lch_pos) - 1):
                    lp_text.append((lch_pos[p][2], text_no_content[lch_pos[p][1]: lch_pos[p + 1][1]]))
                ap2 = re.finditer('Appendix\s?[AI1]\s?.+\n', text_no_content)
                lap2 = []
                for a in ap2: lap2.append((a.start(), a.end(), a.group()))
                while len(lch_pos) > 0 and len(lch_pos[-1]) > 1 and len(lap2) > 0 and len(lap2[0]) > 0 and lch_pos[-1][1] > lap2[0][0]:
                    lap2.pop(0)

                lp_text.append((lch_pos[-1][2], text_no_content[lch_pos[-1][1]: lap2[0][0]]))
                return lp_text
        else:
            return []
    return []


def paper2out(text):
    # Call the clean_text(text) function to clean the text.
    a, c, body_text = clean_text(text)
    # Use nltk sent_tokenize to get number sentences of a.
    sents = nltk.tokenize.sent_tokenize(a)
    nsa = len(sents)
    a = ""
    osummary = ""
    for s in sents:
        if ("University" or "Author" or "@") not in s: a += s
    if is_key_here() and is_on():
        # Get M sentences by Textstar
        gM, gk = get_n_sents(body_text, n_sentences, ks)
        # Concatenate  the abstract, the conclusion and the M sentences
        t1 = a + gM + c
        # and pass result to openAI API model and get the summary.
        st, kwords = API_sum_keywords(t1, get_sents_box(""),get_kwords_box(''))
        sents = st.split('.')
        summary = ".".join(sents[:-1])
        osummary += "\n" + summary + '.'
    # If internet is not available
    else:
        sum, gk = get_n_sents(body_text, get_sents_box(''), get_kwords_box(''))
        osummary += sum
        kwords =  order_keywords(gk)
    return nsa, a, c, osummary, kwords



## 2.20 This function checks a file is PDF or not
def is_pdf(fname):
    if fname[-3:] == 'pdf': return True
    else: return False


## 2.21 This function checks a file is txt file or not
def is_txt(fname):
    if fname[-3:] == 'txt': return True
    else: return False


# 4.32 This function connects to the left button.
# Parameter: none.
# Return : none.
def on_off():
    # If the left button shows "Local mode":
    if buttonMode.config('text')[-1] == "Local mode":
        # When a user clicks the left button, the left button will change to "API mode".
        buttonMode.config(text = "API mode")
        insert_sents_box("Number of summary's words:")
        if not is_internet():
            insert_outbox_message("Hi Tam! The internet is not connected. Please connect your device to the internet if you want to use the API mode!")
        if is_internet() and not is_key_here():
            insert_outbox_message("Hi Tam! Your device is connected to the internet! The left button is ‘ API mode’! Please click the bottom middle button \nto provide your OpenAI key text file, or enter your key into the top middle box after the colon  then click 'Right arrow' key \nif you want to use the API mode!")
            insert_keybox("Tam! Please click the middle button to provide your OpenAI key file, or enter your key here \nthen click 'Right arrow' key:")
        if is_internet() and is_key_here():
            insert_outbox_message("Hi Tam! I am currently in the API mode! I can  start chat with you or  I can "
                                  " summarize and extract keywords from a \ndocument for you. Please following "
                                  "these steps. 1/ Please enter a desired number of summary's words into the top "
                                  "right corner \nbox, then hit 'Return' key (You can skip this step if you do "
                                  "not want a specific number of  words in the summary). 2/ Please enter a desired "
                                  "number of keywords into the top left corner box then hit 'Return' key (You can"
                                  " skip this step if you do not \nwant a specific number of keywords). 3/ Please "
                                  "click on 'Upload a file' button at the bottom right corner; a small window \nappears;"
                                  " move to a location of a desired uploading file; if a desired uploading file is PDF"
                                  " file, it will display on the small window; click on the desired uploading file, "
                                  "then click on the 'Open' button on the small window;if a desired uploading file is "
                                  "TXT file, it will not display on the small window; click on the box which has the"
                                  " words 'PDF files', then the words 'Text \nfiles' shows up; click on the words "
                                  "'Text files', then a desired uploading text file will display;click on the desired "
                                  "\nuploading file, then click on the 'Open' button on the small window. I will show "
                                  "you my summary and keywords on the screen \n from few seconds to minutes depend on the length of the document :)")
    # If the left button shows "API mode":
    else:
        # When a user clicks the left button, the left button will change to "Local mode".
        buttonMode.config(text = "Local mode")
        # Insert the text below to sents_box.
        insert_keybox("Hello Tam! I am currently in the local mode! ")
        # Insert the text below to sents_box.
        insert_sents_box("Number of summary's sentences:")



def is_on():
    if buttonMode.config('text')[-1] == "API mode": return True
    else:return False


def count_words(box):
    return len(box.get("1.0",END).split(" "))


# 4.35 This function connects to the middle button "Upload a key".
# Parameter: none.
# Return : none.
def upload_key():
    file = upload_file()
    if is_txt(file):
        key = ftext2text(file)
        if testkey(key):
            text2file(key, work_dir + "\\key.txt")
            if is_on():
                insert_outbox_message("Hi Tam! I am currently in the API mode! I can  start chat with you or  I can  summarize and extract keywords from a \ndocument for you. Please following these steps. 1/ Please enter a desired number of summary's words into the top right corner box, then click 'Return' key (You can skip this step if you do not want a specific number of summary's words). 2/ Please enter a desired number of keywords into the top left corner box then click 'Return' key (You can skip this step if you do not want a specific number of keywords). 3/ Please click on 'Upload a file' button at the bottom right corner; a small window appears; move to a location of a desired uploading file; if a desired uploading file is PDF file, it will display on the small window; click on the desired uploading file, then click on the 'Open' button on the small window;if a desired uploading file is TXT file, it will not display on the small window; click on the box which has the words 'PDF files', then the words 'Text files' shows up; click on the words 'Text files', then a desired uploading text file will display;click on the desired uploading file, then click on the 'Open' button on the small window. I will show you my summary and keywords on the screen in few seconds :)")
                insert_keybox("Hi Tam! I am currently in the API mode! ")
            else:insert_keybox("The system got a working API key!")
        else:
            insert_outbox_message(" Please provide your OpenAI key again by clicking on  the middle button to provide your OpenAI key file, or entering \nyour key into the top  left box then click 'Right arrow' key!")
            insert_keybox("API mode needs:1/The internet is connected 2/The left button is ‘ API mode’ 3/Click the middle button to \nprovide your OpenAI key file, or enter your key here then click 'Right arrow' key:")
    else:
        insert_outbox_message(" Please provide your OpenAI key again by clicking on  the middle button to provide your OpenAI key file, or entering \nyour key into the top  left box then click 'Right arrow' key!")
        insert_keybox("API mode needs:1/The internet is connected 2/The left button is ‘ API mode’ 3/Click the middle button to provide your OpenAI key file, or enter your key here then click 'Right arrow' key:")

def insert_outbox_message(m):
    out_box.insert(END, "\nSystem:", 'tag2')
    out_box.insert(END, f"{m}")
    out_box.insert(END, "\nTam:", 'tag1')
    out_box.see(END)

# 4.36 This function checks if the provided key is valid or not.
# Parameter: none.
# Return : True or False.
def is_key_here():
    if not os.path.exists(work_dir + "\\key.txt"): return False
    else:
        key = ftext2text(work_dir + "\\key.txt")
        if testkey(key):
            insert_keybox("System is in API mode!")
            return True
        else:
            insert_keybox("Hello Tam! I am currently in the local mode! ")
            return False


def insert_keybox(message):
    key_box.delete('1.0', END)
    key_box.insert(END, f"{message}")
    key_box.see(END)

def insert_sents_box(message):
    sents_box.delete('1.0', END)
    sents_box.insert(END, f"{message}")
    sents_box.see(END)


def num_sents(event):
    t = sents_box.get("1.0","end-1c")
    return t[29:].strip()


# 4.39 This function gets an entered number of (words)sentences from the sents_box if a user enter it.
# Parameter: hit 'Return' key.
# Return : none
wsgobal = 0
numglobal = 0
def get_sents_box(event):
    global numglobal, wsgobal
    t = sents_box.get("1.0", "end-1c")
    if not is_on():
        r = re.findall("[^0-9]*", t[30:])
        if r == [''] or r[0] != '':
            numglobal = n
            return numglobal
        else:
            numglobal = num2int(t[30:])
            return numglobal
    else:
    # number of words
        r = re.findall("[^0-9]*", t[26:])
        if r ==[''] or r[0] != '':
            wsgobal = nw
            return wsgobal
        else:
            wsgobal = num2int(t[26:])
            return wsgobal


# 4.40 This function cleans space, newline, and converts an input string number to an integer.
# Parameter: a string.
# Return : an integer
def num2int(num):
    num = num.strip()
    num = num.replace(" ", "")
    num = num.replace("\n", "")
    num = int(num)
    return num

def get_kwords_box(event):
    global kw
    t = kwords_box.get("1.0", "end-1c")
    r = re.findall("[^0-9]*", t[28:])
    if r == [''] or r[0] != '':
        kw  = k
        return kw
    else:
        kw = num2int(t[28:])
        return kw


def clear():
    out_box.delete("1.0", END)
    out_box.insert(END, "System: Please enter a desired number summary sentences and click enter!", 'tag2')
    out_box.insert(END, "\nTam:", 'tag1')

def save2file(out_box):
    try:
        file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                 filetypes=[("Text files", "*.txt")])
        text = out_box.get("1.0", "end-1c")
        with open(file_path, 'w') as file:
            file.write(text)
    except:
        OSError
        pass



### 3.Call the function to create a withow with the specific title, color, and size
window = create_window("Fun Chat",'green4', 1086, 800)

key_box = Text(window, width = 94, height = 2, fg = 'forest green')
key_box.place(x = 164, y = 8)
key_box.insert(END, "Hello Tam! I am currently in the local mode! ")
key_box.bind('<Return>', getkey)

sents_box = Text(window, width = 19, height = 2, fg = 'forest green')
sents_box.place(x = 926, y = 8)
sents_box.insert(END,"Number of summary's sentences:")
sents_box.bind('<Return>', get_sents_box)


kwords_box = Text(window, width = 19, height = 2, fg = 'forest green')
kwords_box.place(x = 3, y = 8)
kwords_box.insert(END,"Number of desired \nkeywords:")
kwords_box.bind('<Return>', get_kwords_box)


### 5.Create a textbox  which contains the output text
#width = 780, height = 208,x=60, y=80,wchar=97, hchar=8
out_box = scroll_text(1006, 188, 26, 50, 127, 28)
out_box.insert(END, "System: " ,'tag2')
out_box.insert(END,"How are you doing  today Tam? I am in local mode! I will help you save time, energy, and  money"
                   " by  summarizing \nand extracting keywords from a document without fee :) Please following these steps. 1/ Please"
                   " enter a desired number of \nsentences into the top right corner box, then hit 'Return' key (You can"
                   " skip this step if you do not want a specific number \nof sentences). 2/ Please enter a desired"
                   " number of keywords into the top left corner box then hit 'Return' key (You can \nskip this step if you do "
                   "not want a specific number of keywords). 3/ Please click on 'Upload a file' button at the bottom \nright corner;"
                   " a small window appears; move to a location of a desired uploading file; if a desired uploading file is PDF file,"
                   " it will display on the small window; click on the desired uploading file, then click on the 'Open' button on the small window;"
                   "\nif a desired uploading file is TXT file, it will not display on the small window; click on the box "
                   "which has the words 'PDF \nfiles', then the words 'Text files' shows up; click on the words 'Text files',"
                   " then a desired uploading text file will display;click on the desired uploading file, then click on the 'Open' button on the small window."
                   " I will show you my summary and \nkeywords on the screen in few seconds :)  ")

out_box.insert(END,"\nTam: ",'tag1')
out_box.tag_config('tag1', foreground='red',font=('Arial', 10,"bold"))
out_box.tag_config('tag2', foreground='green',font=('Arial', 10,"bold"))
out_box.tag_config('tag3', foreground='purple4',font=('Arial', 10,"italic"))
out_box.tag_config('tag4', foreground='forest green',font=('Arial', 10,"italic"))
out_box.tag_config('tag5', foreground='brown4', font=('Arial', 10, "italic"))
out_box.bind('<Return>', enter)
text = out_box.get("1.0",END)
# print(f"text outbox: {text}")

### 8. Create a button which a user clicks to upload a file

buttonSave = Button(window, bg = "green", text= "S\nA\nV\nE",font=('Arial', 12,"bold"),
                 width = 1, height = 10, anchor=CENTER, highlightthickness = 1,
                 command = lambda: save2file(out_box))
# Place a button in a correct position
buttonSave.place(x = 1064, y = 200)
buttonClear = Button(window, bg = "green", text= "C\nL\nE\nA\nR",font=('Arial', 12,"bold"),
                 width = 1, height = 10, anchor=CENTER, highlightthickness = 1,
                 command = lambda: clear())
# Place a button in a correct position
buttonClear.place(x =3, y = 200)
buttonMode = Button(window, bg = "green", text= "Local mode",font=('Arial', 12,"bold"),
                 width = 30, height = 1, anchor=CENTER, highlightthickness = 1,
                 command = lambda : on_off())
buttonMode.place(x = 26, y = 504)

buttonKey = Button(window, bg = "green", text= "Provide a key",font=('Arial', 12,"bold"),
                 width = 30, height = 1, anchor=CENTER, highlightthickness = 1,
                 command = lambda : upload_key())
buttonKey.place(x = 386, y = 504)

buttonFile = Button(window, bg = "green", text= "Upload a file",font=('Arial', 12,"bold"),
                 width = 30, height = 1, anchor=CENTER, highlightthickness = 1,
                 command = lambda: get_textFfile(out_box))
# Place a button in a correct position
buttonFile.place(x = 746, y = 504)
window.mainloop()