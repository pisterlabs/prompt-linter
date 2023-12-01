from PineconeEngine import PineconeSearch
import openai
import json
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
    global idealog

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
        idealog.append(data)
    idealog.append({"role": "user","content":feedbackInitiation})
    assistant, content = OpenaiHandshake(idealog)

    idealog.append(assistant)
    print("\n\n=====================\n", content)
    content = json.loads(content)
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

def Search_Query_Selection(content, select_index):
    questions_select = [content["Questions_to_Consider"][int(i)-1] for i in select_index]
    input_topic = [content["New_Topic"]] + questions_select
    query = " ".join(input_topic)
    return (query, questions_select)


def AccessDatabase(query, depth):
    input_article = []
    results = PineconeSearch(query, depth)
    processed_results_dict = [{"doi":i["metadata"]["doi"], "abstract":i["metadata"]["abstract"]} for i in results["matches"]]
    return processed_results_dict

def article_read(topic, directions, articles, basic_log):
    topic_prompt = "The user conducts research of topic {}, and wants to focus on {} as his research directions. Here is the article {}"
    kick_off_prompt = "Please response"
    for i in range(len(articles)):
        basic_log.append({"role": "user", "content": topic_prompt.format(topic, directions, articles[i]["abstract"])})
        basic_log.append({"role": "user", "content": kick_off_prompt})
        assistant, content = OpenaiHandshake(basic_log)
        articles[i]["comment"] = content
        basic_log.pop()
        basic_log.pop()
    print(articles)
    return articles

#Summary Writing(Not Done)
def generate_summary(articles):
    
    summary = 0
    return summary

systemSummaryPrompt = '''
You are a scientific writer. 
'''



systemGenePrompt = """
You are a scientific reader in the field of natural science. There is a researcher now conducting a scientif investigation and looking for your help.
The user will later tell you his research topic and some research directions under this topic and give you an article he find may helpful. Your job is to extract 
information that is related to the user's research topic and direction from the article. 

For example, a user is studying the topic about why salt can conduct electricity when it desolves into water. He then 
has some directions to work: ["Does water conduct electricity?", "Does the destruction of ionic bond can produce electical conductivity?", "Is water conductive liquid?"].
He then gives you an article: "We commonly think about salt as consisting of sodium chloride (NaCl) which is common, or table, salt.  But chemists define 
“salts” as any compound that, when dissolved in water, releases a positively charged cation and a negatively charged anion.  
These ions are what can conduct the electricity." 
Then, based on the topic and directions, you should output answer in below dictionary format(if the article doesn't mention this information, just write "")
{"Does water conduct electricity?": "", "Does the destruction of ionic bond can produce electriccal conductivity": "yes, the article supports it by the existence of free cations", "Is water conductive liquid?": ""}
"""


systemPromptforUser = """
You are a marketing designer specializing in understanding people's intentions and interpreting their needs based on user feedback. Your objective is to use the user's input on a specific topic to guide them in exploring the field further.

The user will share their chosen topic for exploration.
Your role is to ask relevant questions that provide direction and clarity for the user's exploration. The new topic you suggested should try to stick to the original proposal. And you are supposed to add additional details to the original idea. You should not change the research direction.
After receiving the user's feedback, your task is to propose an improved version of the new topic for them to explore. The new topic should be concise, creative, and closely aligned with the user's feedback.
Additionally, you need to formulate a research question to help the user gain deeper insights into the idea development process.

Your Response Format (in below python dictionary format):
{"New_Topic": "The new topic based on user feedback", "Research_Question": "A question to help the user explore the idea further", "Questions_to_Consider":
["Question", "Question", "Question", "Question", "Question", "Question", "Question", "Question", "Question", "Question"]}
Take this as an example :{ "New_Topic": "Optimizing Protein Production for Sustainable Agriculture", "Research_Question": "How can we improve protein production methods to meet the growing demand for sustainable agriculture?", 
"Questions_to_Consider": ["What are the current methods used for protein production in agriculture?", "How efficient and sustainable are these methods?",
"Are there any limitations or challenges associated with current protein production methods?", "What are the environmental impacts of protein production in agriculture?",
"Are there any alternative protein sources that can be explored?", "How can technology and innovation be leveraged to optimize protein production?", 
"What role can genetic engineering play in improving protein production?", "How can we ensure the quality and safety of protein produced through new methods?",
"What are the economic implications of adopting new protein production techniques?", "How can we educate and raise awareness about sustainable protein production among consumers and stakeholders?"]}

"""

IniQuery = input("Whats your initial query: \n")
idealog = [
    {
        "role": "system",
        "content": systemPromptforUser
    },
    {
        "role": "user",
        "content": IniQuery
    }
]
gptlog = [
    {
        "role": "system",
        "content": systemGenePrompt
    }
]


assistant, content = RequestFeedback([])
while True:
    choice = input(">>> If you don't like this research direction, press y to continue")
    if choice != "y":
        select_index = []
        print("------------------------------------------------------------------------------")
        print("If you like this direction, select questions from Questions_to_Consider that you like. First give number of questions.")
        num_q = int(input("Number of sentences: "))
        for i in range(num_q):
            input_index = int(input("no. question: "))
            select_index.append(input_index)
        print(select_index)
        break
    feedbackList = CollectFeedback()
    assistant, content = RequestFeedback(feedbackList)

query, questions_consider = Search_Query_Selection(content, select_index)
results = AccessDatabase(query, 1)
topic = content["New_Topic"]
research_directions = questions_consider
articles = article_read(topic, research_directions, results, gptlog)


