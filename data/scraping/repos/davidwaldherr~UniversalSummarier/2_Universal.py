import openai
openai.api_key ='INSERT OPENAI KEY HERE'
from openai.embeddings_utils import get_embedding
import pandas as pd

# Before launching this program, you must create a bookSentences.csv file

# This program is a continuation of the 'hot' output from 1_bookToUniversal.py.
# It takes that boiling output (in the form of Book.txt) and converts it to
# and embedded BookSummary.csv and BookSummary.jsonl. It also reformats the
# contents of the Book.txt file to be more readable.

# The final step in the process is to create fine tunes from BookAuthorVoice.jsonl
# and BookSummary.jsonl, which will be utalized by the essay writing program.

# prompt the user to enter a file name
def getFileName():
    filename = input('Enter the name of the file: ')
    return filename

def embedDecision():
    summary = input('Would you like to embed the book summary? (y/n): ')
    return summary

def getSummary(filename):
    summary = []
    with open(filename, 'r') as f:
        for line in f:
            # make each line lowercase
            if line != '\n':
                # strip the line of whitespace
                line = line.strip()
                if filename in line:
                    summary.append(line)
                elif "Chapter: " in line:
                    summary.append(line)
                elif "Section: " in line:
                    summary.append(line)
                else:
                    summary.append("Paragraph: " + line)
    f.close()
    return summary

def reformatBookTxt(filename, summary):
    summaryFormat = []
    for line in summary:
            if filename in line:
                summaryFormat.append(line + '\n-----------------------------------\n\n')
            elif "Chapter: " in line:
                summaryFormat.append('\n\n\n' + line)
            elif "Section: " in line:
                summaryFormat.append('\n\n' + line)
            else:
                text = line.replace("Paragraph: ", "")
                summaryFormat.append('\n' + text)
    with open(filename, 'w') as f:
        for i in range(len(summaryFormat)):
            f.write(summaryFormat[i])
    f.close()

def createSummaryJSONL(summary):
    summaryJSONL = []
    for line in summary:
        if line.startswith("Paragraph: "):
            par = line.replace("Paragraph: ", "")
            par = par.replace('"', '')
            summaryJSONL.append('{"prompt":"", "completion":" ' + par + '"}')
    return summaryJSONL

# print the list to a jsonl file
def writeJSONL(filename, summaryJSONL):
    with open(filename, 'w') as f:
        for i in range(len(summaryJSONL)):
            f.write(summaryJSONL[i] + '\n')

# Create the list that contains the contents of the CSV file
def createCSV(mySummaryEmbed):
    summaryCSV = []
    chapter = ""
    section = ""
    paragraph = ""
    for i in range(len(mySummaryEmbed)):
        # remove all instances of " from the string
        mySummaryEmbed[i] = mySummaryEmbed[i].replace('"', '')
        if mySummaryEmbed[i].startswith('Chapter: '):
            # set the chapter variable to the line after "Chapter: "
            chapter = mySummaryEmbed[i].replace('Chapter: ', '')
            chapter = '"' + chapter + '"'
        elif mySummaryEmbed[i].startswith('Section: '):
            # set the section variable to the line after "Section: "
            section = mySummaryEmbed[i].replace('Section: ', '')
            section = '"' + section + '"'
        elif mySummaryEmbed[i].startswith('Paragraph: '):
            # set the paragraph variable to the line after "Paragraph: "
            paragraph = mySummaryEmbed[i].replace('Paragraph: ', '')
            paragraph = '"' + paragraph + '"'
            summaryCSV.append("{},{},{}".format(chapter, section, paragraph))
    return summaryCSV

# print the list to a csv file
def writeSummaryCSV(filename, summaryCSV):
    with open(filename, 'w') as f:
        f.write("chapter,section,paragraph\n")
        for line in summaryCSV:
            f.write(line + '\n')
        f.close()

def embedSummaryParagraphs(filename):
    df = pd.read_csv(filename, dtype=str)
    df = df[['chapter', 'section', 'paragraph']]
    # remove any duplicate rows
    df = df.drop_duplicates()
    # This will take ~multiple~ minutes
    df['babbage_search'] = df.paragraph.apply(lambda x: get_embedding(x, engine='text-search-babbage-doc-001'))
    df.to_csv(filename)

# Gather the necessary components
filename = getFileName()
embedDecision = embedDecision()
summary = getSummary(filename + '.txt')

# reformat the book.txt file
reformatBookTxt(filename + '.txt', summary)

# Create the BookSummary.jsonl file
summaryJSONL = createSummaryJSONL(summary)
writeJSONL(filename + 'Summary.jsonl', summaryJSONL) # create a JSONL file of the Summary

# Create the bookSummary.csv file
summaryCSV = createCSV(summary)
writeSummaryCSV(filename + 'Summary.csv', summaryCSV) # write to the csv file

# Embed the book summary
if embedDecision == 'y':
    embedSummaryParagraphs(filename + 'Summary.csv')