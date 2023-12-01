import requests
import cohere
# import transcriptGenerator
'''
Input: textChunks from transcriptGenerator.py. of note, the first index of textChunks is the title of the video!
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
"""
    For each sub-string, we need a prompt
    co.generate request Responses: obtained via cohere.CohereAPIError().something, forgot which one
    200 = ok
    400 = bad request
    498 = Blocked Input or Output
    500 = Internal Server Errors
"""
# initialize the Cohere Client with an API Key
keys = ["5tF9d0IkQRph19apERAfxoudDUzmEnfyxo6pimfB", 
        "yRYJDINsAmmxz7y00xUfeAHAaUqjAO1c7XXLXzhv",
        "ivsDVDocsH8LnNhQ7b21PZURol6yr5x8UCVRwGdh",
        "2UGd1Y0q61JFhlGF75XIahwbeFm32cihdc69gSRp",
            "fOGiQj8bpLYdAht5fXKpAdalEoeYUJZ7O1A50jCr",
            "vFop3G1hOGieKjsRie1EfpKkF3gpXp93qL8Hafk1",
            "ua2N4o4bsuk1ZqyUSr7q96QOscwmYgG5Ej9pwTxu"]

modelType = "command"
randomness = 0.9

# promptList = ["What are the main points of this in bullet points:", 
#                 "Summarize this into bullet points:",
#                 f"Summary about ", " in bullet points"]

def generate_prompt(title):
    return f'What are the main points of this video about: {title} in bullet points:'

def try_process(promptText, index, title):
    cohereClient = cohere.Client(keys[index])
    try:
        response = cohereClient.generate(
                        model=modelType,
                        prompt= (generate_prompt(title) + "\n\"" + promptText + "\""),
                        max_tokens=4050,
                        temperature=randomness,
                        truncate="END")
        
        #base case, we found a key that acutally works
        #no error, we can get something
        outputList = list(response.generations[0].text.split("\n"))
        outputList.remove(outputList[-1])
        if(len(outputList) != 0):
            outputList.remove(outputList[0])
        print()
        print(outputList)
        
        plainOutput = ""
        for k in outputList:
            plainOutput += k
            plainOutput += "\n"

        print(plainOutput)
        return plainOutput
    
    except cohere.CohereAPIError as e:
        # we have an error
        if(index < len(keys) - 1):
            #recursive case, key doesnt work and there are still keys to check
            #print("checked:" + str(index+1) + " out of " + str(len(keys)))
            return try_process(promptText, index+1, title)
        else:            
            #base case, index is past the # of keys we actually have
            print("ERROR CODE 2: NO KEYS AVAILABLE!")
            return ""


def process_transcriptV2(text_chunks, title):
    if(text_chunks == ""):
        #print("ERROR CODE 1: NO TRANSCRIPT FOUND!")
        return "ERROR CODE 1: NO TRANSCRIPT FOUND!"

    summary = ""
    titleCheck = False
    count = 0
    for chunk in text_chunks:
        if(titleCheck != True):
            #title chunk
            # title = chunk
            titleCheck = True
        else:
            print("chunk # " + str(count))
            summary += try_process(chunk, 0, title)
            count += 1

    return summary
    

"""
Debug:
"""

# summary = process_transcriptV2(transcriptGenerator.generate_transcript(transcriptGenerator.test_URL))
# print('=========================================')
# print(summary)