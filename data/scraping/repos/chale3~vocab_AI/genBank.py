import openai

# load and set our key
def w25(message):
    openai.api_key = open("keys/key2.txt", "r").read().strip("\n")
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": message}]
    )
    reply_content = completion.choices[0].message.content
    return(reply_content)


path = "/Users/chale/Dev2/chatGPT_Projects/word_lists/gregMatUnknownList.text"
f = open(path, "r")
lines = f.readlines()
f.close()

#read 25 lines at a time from lines and store in a list
#then pass that list to openai
fullList = ""
box = 1
for i in range(0, len(lines), 25):
    x = lines[i:i+25]
    #x to comma separated string
    x = "".join(x)
    format = "\nYour response will follow this format: word, categorization"
    message = "categorize each of the following words as either \"an act\" or a \"state of being\": " + x + format
    message2 = "categorize each of the following words by their connotation as \"positive\", \"negative\", or \"neutral\": " + x + format
    reply = w25(message2)
    fullList += reply
    fullList += "\n"
    print(box)
    box += 1


#write the fullList to a file
path = "word_lists/May_09_conno.txt"
f = open(path, "w")
f.write(fullList)
f.close()

