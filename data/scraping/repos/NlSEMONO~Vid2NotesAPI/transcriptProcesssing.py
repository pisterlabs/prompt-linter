import requests
import cohere
# from NOTES import NOTES
# import transcriptGenerator
'''
Input: textChunks from transcriptGenerator.py
Output: generatedSummary - String which contains the summary of the video fed into transcriptGenerator.py
alternatively, you can use outputList - contains the same content as generatedSummary but separated by Cohere Request # 

Note about Cohere API requests:
    Each key has 5 requests each per minute. 
    We cannot excede 4081 tokens, and each word is about 1-3 tokens
    Therefore we cannot have more than 1360 words per request
    Assuming a person speaks about 250 words per minute in the worst case, we can take 5-ish minutes of text for each request
    Therefore, we can handle a video of 125 minutes?

Sticking with Generate Function
Used Prompts: Note: Square Brackets [] mean optional/extra
    What are the main points of this [into bullet points]:
    Summarize this into bullet Points

Note: The Transcript Summary function on Cohere is very inaccurate, using Generate instead
'''

# initialize the Cohere Client with an API Key
keys = ["5tF9d0IkQRph19apERAfxoudDUzmEnfyxo6pimfB", 
        "yRYJDINsAmmxz7y00xUfeAHAaUqjAO1c7XXLXzhv",
        "ivsDVDocsH8LnNhQ7b21PZURol6yr5x8UCVRwGdh",
        "2UGd1Y0q61JFhlGF75XIahwbeFm32cihdc69gSRp",
            "fOGiQj8bpLYdAht5fXKpAdalEoeYUJZ7O1A50jCr",
            "vFop3G1hOGieKjsRie1EfpKkF3gpXp93qL8Hafk1",
            "ua2N4o4bsuk1ZqyUSr7q96QOscwmYgG5Ej9pwTxu"]

modelType = "command"
randomness = 0
PUNC = {'?', '.', ',', '!', ':', ';'}
BLACKLIST = {'what'}
BLACKLISTED_SYMBOLS = {'-','*', ' '}
TEXT_BREAKPOINT = 6 * 60

# promptList = ["What are the main points of this in bullet points:", 
#                 "Summarize this into bullet points:",
#                 f"Summary about ", " in bullet points"]

def generate_prompt(title, medium='video', type=1):
    if type == 1:
        str_so_far = ''
        for char in title:
            if char in PUNC:
                break
            str_so_far += char
        new_title = str_so_far.split()
        new_title = ' '.join([word for word in new_title if word.lower() not in BLACKLIST])[:-1]
        return f'Summarize the main points of this {medium} about: {new_title} in bullet points:'
        # return f'What are the main points of this part of a video on {title} in bullet points:'
    else:
        return f"Can you turn this into a fill in the blanks question:"

def try_process(promptText, index, title, media='video', type=1):
    cohereClient = cohere.Client(keys[index])
    try:
        response = cohereClient.generate(
                        model=modelType,
                        prompt= (generate_prompt(title, media, type) + "\n\"" + promptText + "\""),
                        max_tokens=4050,
                        temperature=randomness,
                        truncate="END")
        #base case, we found a key that acutally works
        #no error, we can get something
        outputList = list(response.generations[0].text.split("\n"))
        print('++++++++++++++++++++++++++++++++++++\n\n')
        print(f'{outputList}')
        if type == 1:
            while '' == outputList[-1]:
                outputList.remove(outputList[-1])
                if(len(outputList) != 0 and 'summary of the main points' in outputList[0] or '' == outputList[0]):
                    outputList.remove(outputList[0])
            print('===================================\n\n')
            print(outputList)
            
            plainOutput = ""
            for k in outputList:
                if k != "":
                    while k != '' and k[0] in BLACKLISTED_SYMBOLS:
                        k = k[1:]
                    plainOutput += k
                else:
                    plainOutput += k
                plainOutput += "\n"
            print(plainOutput)
            return plainOutput
        else:
            return '*()&*68234' if (len(outputList) < 2 or outputList[2] == '') else f'{outputList[2]}\n'
    except cohere.CohereAPIError as e:
        # we have an error
        if(index < len(keys) - 1):
            #recursive case, key doesnt work and there are still keys to check
            #print("checked:" + str(index+1) + " out of " + str(len(keys)))
            return try_process(promptText, index+1, title)
        else:            
            #base case, index is past the # of keys we actually have
            # print("ERROR CODE 2: NO KEYS AVAILABLE!")
            return ""


def process_transcriptV2(text_chunks, title, length=90000, type=1):
    if(text_chunks == ""):
        #print("ERROR CODE 1: NO TRANSCRIPT FOUND!")
        return "ERROR CODE 1: NO TRANSCRIPT FOUND!"
    print(type)
    summary = ""
    titleCheck = False
    count = 0
    for chunk in text_chunks:
        if(titleCheck != True):
            #title chunk
            # title = chunk
            titleCheck = True
        else:
            print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n\n')
            print("chunk # " + str(count))
            print(f'\n{chunk}')
            summary += try_process(chunk, 0, title, 'text' if length <= TEXT_BREAKPOINT else 'video', type=type)
            count += 1

    return summary
    

"""
Debug:
"""
# flashcards = process_transcriptV2(NOTES, 'test', type=2)
# print(flashcards)

# summary = process_transcriptV2(transcriptGenerator.generate_transcript(transcriptGenerator.test_URL))
# print('=========================================')
# print(summary)