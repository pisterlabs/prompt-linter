# from PineconeEngine import PineconeSearch
import openai
import re
openai.api_key = "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"

def CollectFeedback():
    feedbackPrompt = """
    Now you are supposed to select the specific sentences that you found yourself interested to. After copy and pasting them, remember to give a short explaination to why you are interested to those key points.
    """
    
    print("\n==========================")
    print(feedbackPrompt)
    feedbackList = []
    while True:
        sentence = input('\n>>> Enter interesting sentences below: (type "done" to stop) \n')
        if sentence == "done":
            break

        reason = input(">>> Explain why you found this topic interesting: \n")
        data = {
            "topic": sentence,
            "reason": reason
        }
        feedbackList.append(data)
        
    return feedbackList

def RequestFeedback(feedback):
    global history

    feedbackTemplate = """
    User's interest: {topic}
    Reason: {reason}
    """
    feedbackInitiation = "Please give your response"

    for item in feedback:
        data = {
            "role": "user",
            "content": feedbackTemplate.format(
                topic=item["topic"], 
                reason=item["reason"]
            )
        }    
        history.append(data)
    history.append({"role": "user","content":feedbackInitiation})
    assistant, content = OpenaiHandshake(history)

    history.append(assistant)
    print("\n\n=====================\n", content)
    
    return assistant, content

def OpenaiHandshake(chain):
    response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=chain,
            temperature=0,
            presence_penalty=1
        )
    response_m = response.choices[0].message
    return response_m, response_m["content"]

# def AccessDatabase(query):
#     results = PineconeSearch(query, 5)
#     abstractList = []
#     for item in results["matches"]:
#         data = item["metadata"]["abstract"]
#         abstractList.append(data)
    
#     print("\n\n===================")
#     for item in abstractList:
#         print(item, end="\n\n")
#     return abstractList

def accessNewTopic(input_text):
    # Define the regular expression pattern to find text between square brackets
    pattern = r"New Topic: (.+?)\n"
    # Use the re.search() function to find the first occurrence of the pattern
    match = re.search(pattern, input_text)

    if match:
        return match.group(1)
    else:
        return None

systemPrompt = """
You are a marketing designer specializing in understanding people's intentions and interpreting their needs based on user feedback. Your objective is to use the user's input on a specific topic to guide them in exploring the field further.

The user will share their chosen topic for exploration.
Your role is to ask relevant questions that provide direction and clarity for the user's exploration. The new topic you suggested should try to stick to the original proposal. And you are supposed to add additional details to the original idea. You should not change the research direction.
After receiving the user's feedback, your task is to propose an improved version of the new topic for them to explore. The new topic should be concise, creative, and closely aligned with the user's feedback.
Additionally, you need to formulate a research question to help the user gain deeper insights into the idea development process.

Your Response Format:
> New Topic: [The new topic based on user feedback]
> Research Question: [A question to help the user explore the idea further]
> Questions to Consider:
1. [Question 1]
2. [Question 2]
3. [Question 3]
"""

IniQuery = input("Whats your initial query: \n")
history = [
    {
        "role": "system",
        "content": systemPrompt
    },
    {
        "role": "user",
        "content": IniQuery
    }
]

assistant, content = RequestFeedback([])
while True:
    query = accessNewTopic(content)
    # databaseResults = AccessDatabase(query)

    choice = input(">>> Do you wish to continue?")
    if choice != "y":
        break
    feedbackList = CollectFeedback()
    assistant, content = RequestFeedback(feedbackList)
