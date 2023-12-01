import openai
openai.api_key ='INSERT OPENAI KEY HERE'
from openai.embeddings_utils import get_embedding
import pandas as pd

# You need to create a virtual environment
# pip3 install virtualenv
# . venv/bin/activate

# This program accepts a text file as an input and then divides it into chapters,
# sections, and paragraphs, and sentences. It then uses that information to create a bookAuthorVoice.jsonl 
# file, BookSentence.csv file, and Book.txt summary.

# Before launching this program, you must create a bookSentences.csv file

# Surround the table of contents and the content itself with "Start" and "End" to automatically
# parse the text.

# x is deadspace
# 0 is empty line
# 1 is paragraph
# 2 is table of contents item
# 3 is chapter or part
# 4 is section

# prompt the user to enter a file name
def getFileName():
    filename = input('Enter the name of the file: ')
    return filename

# # prompt the user if they would like to embed all of the sentences in the book
def summarizeDecision():
    summary = input('Would you like to summarize the book? (y/n): ')
    return summary

def embedDecision():
    summary = input('Would you like to embed each sentence in book? (y/n): ')
    return summary

def sectionsDecision():
    sections = input('Would you like to create sections? (y/n): ')
    return sections

def embedSentences(filename):
    df = pd.read_csv(filename, dtype=str)
    df = df[['chapter', 'section', 'paragraph', 'sentence']] #    # remove any duplicate rows
    df = df.drop_duplicates()
    # # This will take ~many~ minutes
    df['babbage_search'] = df.sentence.apply(lambda x: get_embedding(x, engine='text-search-babbage-doc-001'))
    df.to_csv(filename)

# import the text file for a book, make it lowercase, and parse it into the text list
def readFile(filename):
    text = []
    with open(filename, 'r') as f:
        for line in f:
            # make each line lowercase
            if line != '':
                # strip the line of whitespace
                line = line.strip()
                line = line.lower()
            text.append(line)
            # print(line)
    return text

def startTable(text):
    table = []
    start = 0
    end = 0
    for i in range(len(text)):
        if text[i] == 'start':
            if start == 0:
                table.append('2')
                start = 1
            else:
                table.append('1')
        elif text[i] == 'end':
            if end == 0:
                table.append('2')
                end = 1
            else:
                table.append('1')
        elif text[i] == '':
            table.append('0')
        # elif text[i] == '• • •':
        #     table.append('0')
        else: 
            table.append('1')
    return table

# this function labels every line in the table of contents with 2.
# it also marks an x before the start of the table of contents.
def markContents(table):
    start = 999
    end = 999
    for i in range(len(table)):
        if table[i] == '2':
            if start == 999:
                start = i
            else:
                end = i
    # between the start and end, if the line = 1, change it to 2
    for i in range(start, end):
        if table[i] == '1':
            table[i] = '2'
    if start == 0:
        pass
    else:
        for i in range(0, start):
            table[i] = 'x'
    return table

# this function creates a list of the table of contents
def createContents(text, table):
    contents = []
    for i in range(len(table)):
        if table[i] == '2':
            contents.append(text[i])
    return contents

def lastElement(contents):
    last = contents[-1]
    return last

def markEndDeadspace(text, contents, table):
    # store the last element in contents in a variable
    last = lastElement(contents)
    # iterate through the text list and if the line is == last, save the index
    for i in range(len(text)):
        if text[i] == last:
            end = i
    # mark every line in table as an x after the end index
    for i in range(end, len(table)):
        table[i] = 'x'
    return table

def extendContents(contents):
    # remove one character at a time from each item in contents and append it to contents2
    contents2 = []
    for i in range(len(contents)):
        for j in range(len(contents[i])):
            contents2.append(contents[i][j:])
    # remove leading and trailing whitespace from each item in contents2
    for i in range(len(contents2)):
        contents2[i] = contents2[i].strip()
    contents3 = []
    for i in range(len(contents2)):
        if contents2[i] not in contents3:
            contents3.append(contents2[i])
    # remove any empty strings from contents2
    contents3 = [x for x in contents3 if x != '']
    return contents3

def findChapters(text, contents, table):
    for i in range(len(table)):
        if table[i] == '1':
            for j in range(len(contents)):
                if contents[j] == text[i]:
                    table[i] = '3'
                    break
    return table

def findSections(text, table):
    for i in range(len(table)):
        if table[i] == '1':
            text[i] = text[i].strip()
            # if text[i] does not end with a period, question mark, exclamation point, astrix, comma
            # it is under 15 words, and does not start with a '-', mark it in the table as a 4
            if text[i][-1] not in ['.', '?', '!', '”', '*', ',', '†', ';', ':', '‡', '’']:
                # if the line does not start with "-"
                if text[i][0] != '—':
                    if len(text[i].split()) < 8: # set how many words are allowed in a section
                        table[i] = '4'
    return table

# this function creates a JSONL file of all the paragraphs in the book
def createJSONL(text, table):
    jsonl = []
    for i in range(len(table)):
        if table[i] == '1':
            if len(text[i].split()) > 60:
                par = text[i].replace('"', '')
                jsonl.append('{"prompt":"", "completion":" ' + par + '"}')
    return jsonl

# print the list to a jsonl file
def writeJSONL(filename, list):
    with open(filename, 'w') as f:
        for i in range(len(list)):
            f.write(list[i] + '\n')

# this function creates a list that can be transformed into a csv file
# CSP - Chapter, Section, Paragraph
def getCSP(text, table):
    sentenceCSP = []
    for i in range(len(table)):
        if table[i] == '1':
            sentenceCSP.append("Paragraph: " + text[i])
        elif table[i] == '3':
            sentenceCSP.append("Chapter: " + text[i])
        elif table[i] == '4':
            sentenceCSP.append("Section: " + text[i])
    return sentenceCSP

# this function creates a CSPS format to be read for CSV and Summary
# CSPS - Chapter, Section, Paragraph, Sentence
def createEmbed(sentenceCSP):
    sentenceEmbed = []
    for line in sentenceCSP:
        if line.startswith('Paragraph: '):
            sentenceEmbed.append(line)
            line = line.replace('Paragraph: ', '')
            line = line.split('.')
            # remove outside whitespace
            line = [x.strip() for x in line]
            # remove empty strings from list
            line = [x for x in line if x]
            for sentence in line:
                sentenceEmbed.append("Sentence: " + sentence)
        else:
            sentenceEmbed.append(line)
    return sentenceEmbed

# this function creates a list that can be printed directly to the csv file
def createCSV(sentenceEmbed):
    csv = []
    chapter = ""
    section = ""
    paragraph = ""
    sentence = ""
    for i in range(len(sentenceEmbed)):
        # remove all instances of " from the string
        sentenceEmbed[i] = sentenceEmbed[i].replace('"', '')
        if sentenceEmbed[i].startswith('Chapter: '):
            # set the chapter variable to the line after "Chapter: "
            chapter = sentenceEmbed[i].replace('Chapter: ', '')
            chapter = '"' + chapter + '"'
        elif sentenceEmbed[i].startswith('Section: '):
            # set the section variable to the line after "Section: "
            section = sentenceEmbed[i].replace('Section: ', '')
            section = '"' + section + '"'
        elif sentenceEmbed[i].startswith('Paragraph: '):
            # set the paragraph variable to the line after "Paragraph: "
            paragraph = sentenceEmbed[i].replace('Paragraph: ', '')
            paragraph = '"' + paragraph + '"'
        elif sentenceEmbed[i].startswith('Sentence: '):
            # set the paragraph variable to the line after "Paragraph: "
            sentence = sentenceEmbed[i].replace('Sentence: ', '')
            sentence = '"' + sentence + '"'
            csv.append("{},{},{},{}".format(chapter, section, paragraph, sentence))
    return csv

# print the list to a csv file
def writeCSV(filename, csv):
    with open(filename, 'w') as f:
        f.write("chapter,section,paragraph,sentence\n")
        for line in csv:
            f.write(line + '\n')
        f.close()

def prepareToSummarize(sentenceCSP):
    # prepend and append any line that starts with "Chapter: ", or "Section: " with "==="
    for i in range(len(sentenceCSP)):
        if sentenceCSP[i].startswith('Chapter: ') or sentenceCSP[i].startswith('Section: '):
            sentenceCSP[i] = '===' + sentenceCSP[i]
        if sentenceCSP[i].startswith('==='):
            sentenceCSP[i] = sentenceCSP[i] + '==='
    # if there are over 5 lines in a row starting with "Paragraph: ", prepend and append every fourth line with "==="
    for i in range(len(sentenceCSP)):
        if sentenceCSP[i].startswith('Paragraph: '):
            if i % 4 == 0:
                sentenceCSP[i] = '===' + sentenceCSP[i]
            if i % 4 == 3:
                sentenceCSP[i] = sentenceCSP[i] + '==='
    # combine the list of lines into a single string
    sentenceCSP = ' '.join(sentenceCSP)
    # split the string by '==='
    sentenceCSP = sentenceCSP.split('===')
    sentenceCSP = [line.strip() for line in sentenceCSP]
    sentenceCSP = [x for x in sentenceCSP if x != '']
    # for each line that starts with "Paragraph: ", remove every instance of it after the first one
    for i in range(len(sentenceCSP)):
        if sentenceCSP[i].startswith('Paragraph: '):
            sentenceCSP[i] = sentenceCSP[i].replace('Paragraph: ', '')
            sentenceCSP[i] = "Paragraph: " + sentenceCSP[i]
    # remove any empty strings from the list
    sentenceCSP = [x for x in sentenceCSP if x != '']
    return sentenceCSP

def summarize(text):
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt="Summarize this for a second-grade student:\n\n" + text + "\n\n",
        temperature=0.69,
        max_tokens=2048,
    )
    return response.choices[0].text

def summarizeTwo(text):
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt="Summarize the following:\n\n" + text + "\n\n",
        temperature=0.69,
        max_tokens=2048,
    )
    return response.choices[0].text

def get_completions(preparedToSummarize):
    func = []
    for paragraph in preparedToSummarize:
        # if the line contains "Chapter", or "Section" skip it
        if paragraph.startswith("Chapter") or paragraph.startswith("Section"):
            func.append(paragraph)
        elif paragraph.startswith("Paragraph: "):
            completeMe = paragraph.replace("Paragraph: ", "")
            completion = summarize(completeMe)
            func.append(completion)
        else: 
            completion = summarizeTwo(paragraph)
            func.append(completion)
    # remove any empty strings
    func = [x for x in func if x]
    return func

def processSummary(extendedSummary):
    # remove any empty strings
    for line in extendedSummary:
        if line.startswith("Summary of "):
            line = line + '\n'
        elif line.startswith("Chapter: "):
            line = '\n\n' + line
        elif line.startswith("Section: "):
            line = '\n' + line
    return extendedSummary

# print the list to a csv file
def writeSummary(filename, extendedSummary):
    with open(filename, 'w') as f:
        f.write("Summary of " + filename + "\n\n")
        for line in extendedSummary:
            f.write(line + '\n')
        f.close()

# Get the file name and summarize/embed decisions from the user
myFile = getFileName()
summarizeDecision = summarizeDecision()
embedDecision = embedDecision()
sectionDecision = sectionsDecision()

# convert the file into a table that marks the significance of each line
text = readFile(myFile + '.txt')
table = startTable(text)
table = markContents(table)
contents = createContents(text, table)
table = markEndDeadspace(text, contents, table)
contents = extendContents(contents)
table = findChapters(text, contents, table)
if sectionDecision == "y":
    table = findSections(text, table)

# Create the CSV file for the contents of the book
sentenceCSP = getCSP(text, table) # use this for Summary
sentenceEmbed = createEmbed(sentenceCSP) # use this for CSV
csv = createCSV(sentenceEmbed)
writeCSV(myFile + 'Sentences.csv', csv) # write to the csv file

# Summarize the book
preparedToSummarize = prepareToSummarize(sentenceCSP) # there are no token issues with this
if summarizeDecision == 'y':
    firstSummary = get_completions(preparedToSummarize)
    extendedSummary = get_completions(firstSummary)
    writeSummary(myFile + '.txt', extendedSummary)
    
# Create BookAuthorVoice.jsonl
jsonl = createJSONL(text, table)
writeJSONL(myFile + 'AuthorVoice.jsonl', jsonl) # create a JSONL file of the Paragraphs

# Embed each sentence in the book
if embedDecision == 'y':
    embedSentences(myFile + 'Sentences.csv')


##
# Testing functions and everything below not needed for program
##

# print an edited version of the book to a text file
def writeBook(filename, text, table):
    with open(filename, 'w') as f:
        section = []
        paragraphs = []
        for i in range(len(text)):
            if table[i] == '4':
                section.append(text[i])
            elif table[i] == '1':
                paragraphs.append(text[i])
        for i in range(len(section)):
            f.write('SECTIONS\n')
            f.write(section[i])
            f.write('\n')
        for i in range(len(paragraphs)):
            f.write('PARAGRAPHS\n')
            f.write(paragraphs[i])
            f.write('\n')

# print the list to a text file
def writeTable(filename, table):
    with open(filename, 'w') as f:
        for i in range(len(table)):
            f.write(table[i])

# print the list to a text file
def writeContents(filename, contents):
    with open(filename, 'w') as f:
        for i in range(len(contents)):
            f.write(contents[i] + '\n')

writeBook('bookSections_Paragraphs.txt', text, table)
writeTable('bookTable.txt', table)
writeContents('bookContents.txt', contents)
print(len(text))
print(len(table))