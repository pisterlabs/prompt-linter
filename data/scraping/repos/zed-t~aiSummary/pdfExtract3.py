import os.path
import sys
import math
import openai
#import PyCryptodome
from PyPDF2 import PdfReader
import tkinter as tk
from tkinter import filedialog 
from tkinter import messagebox

def uniquify(name, goalPath):
    filename, extension = os.path.splitext(name)
    tempName = filename + extension
    path = os.path.join(goalPath, tempName)
    counter = 1

    while os.path.exists(path):
        tempName = filename + " (" + str(counter) + ")" + extension
        path = os.path.join(goalPath, tempName)
        counter += 1

    return path

def getTargetPDF():
    root = tk.Tk()
    root.withdraw()

    messagebox.showinfo("Prompt","Please select a file to summarize")
    file_path = filedialog.askopenfilename()
    return file_path


# Divide option and arguement inputs into two arrays
opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]

pdfName = getTargetPDF()

# Create an instance of the GPT-3 model
openai.api_key = "!!! Your GPT Key Here !!!"

# open the PDF file
pdfFile = open(pdfName, 'rb')

# create PDFFileReader object to read the file
pdfReader = PdfReader(pdfFile)
meta = pdfReader.metadata;

# set max length of input to GPT3
acceptableLengthGPT = 3297*2

# set path of output summary file
summaryPath = uniquify("summary of "+os.path.basename(pdfName).split(".")[0]+".txt", os.path.dirname(pdfName))

# set document metadata
documentData = "Document info:"
if(meta is not None):
    documentData = documentData +"\nAuthor: " + str(meta.author) +"\nTitle: " + str(meta.title)
documentData = documentData+"\n- - - - - - - - - - - - - - - - - - - -"+"\nNumber of Pages: " + str(len(pdfReader.pages))

# display and save document metadata
print(documentData)
with open(summaryPath, 'a') as f:
        # Write document metadata to the file
        f.writelines(documentData)

numOfPages = len(pdfReader.pages)
rawText = ""
pageEndIndices = []

# parse all text into single string
for i in range(0, numOfPages):
    nextText = pdfReader.pages[i].extract_text()
    rawText = rawText + nextText
    pageEndIndices.append(len(rawText))

end = 0
# send text to GPT3 in batches
for i in range(0, math.floor(len(rawText)/acceptableLengthGPT)):
    start = end

    # find end of batch
    if(end+acceptableLengthGPT < len(rawText)):
        end = end+acceptableLengthGPT
    else:
        end = len(rawText)

    # find index of last end of sentence in batch
    for j in range(0, acceptableLengthGPT):
        if(rawText[end] == "."):
            end += 1
            break
        else:
            end -= 1

    #print("Start and end indexed: "+rawText[start]+", "+rawText[end-1])

    # divide the text into lengths accepted by GPT3 (<3297 chars)
    targetText = rawText[start:end]

    # get which pages are currently being read from
    pages = []
    for j in range(0, numOfPages):
        if(((i == 0 and j == 0) 
            or (j > 0 and start <= pageEndIndices[j-1] + 1 < end)) 
            or start <= pageEndIndices[j] < end):
            pages.append(j+1)

    response = openai.Completion.create(
    engine="text-davinci-003",
    prompt="Summarize this text in the style of the New Yorker: " + targetText,
    max_tokens=800,
    n=1,
    temperature=0.3,
    presence_penalty=0.1,
)

    # Print the generated summary
    print("\n\nPage(s) "+str(pages)+":\n"+response["choices"][0]["text"])

    # Open a text file for writing
    with open(summaryPath, 'a', encoding="utf-8") as f:
        # Write the list of bullet point summaries to the file
        f.writelines("\n\nPage(s) "+str(pages)+":\n"+response["choices"][0]["text"])

# close the PDF file object
pdfFile.close()

# open summary file
os.startfile(summaryPath)

