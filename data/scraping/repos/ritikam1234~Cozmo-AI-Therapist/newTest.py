import os
import openai
import sentiment_analysis

openai.api_key = os.getenv("OPENAI_API_KEY")

def sentiment_analysis(text):
    preamble = """
    """

    premise = """
    Classify the following sentence into one of the labels from the following list of emotions: ['Neutral', 'Amazed', 'Foolish', 'Overwhelmed', 'Angry', 'Frustrated', 'Peaceful', 'Annoyed', 'Furious', 'Proud', 'Anxious', 'Grievous', 'Relieved', 'Ashamed', 'Happy', 'Resentful', 'Bitter', 'Hopeful', 'Sad', 'Bored', 'Hurt', 'Satisfied', 'Comfortable', 'Inadequate', 'Scared', 'Confused', 'Insecure', 'Self-conscious', 'Content', 'Inspired', 'Shocked', 'Depressed', 'Irritated', 'Silly', 'Determined', 'Jealous', 'Stupid', 'Disdain', 'Joy', 'Suspicious', 'Disgusted', 'Lonely', 'Tense', 'Eager', 'Lost', 'Terrified', 'Embarrassed', 'Loving', 'Trapped', 'Energetic', 'Miserable', 'Uncomfortable', 'Envious', 'Motivated', 'Worried', 'Excited', 'Nervous', 'Worthless']
    Respond with only one word, which is an emotion from the list. If not sure, respond with only unknown.
    """

    query = """
    Classify the following sentence into an emotion from the list of emotions.
    Respond with only one word, which is an emotion from the list.  If not sure, respond with only unknown.
    """
    while True:
        query = text
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "assistant", "content": preamble},
                    {"role": "system", "content": premise},
                    {"role": "user", "content": query}],
            temperature = 0.3,
            max_tokens = 200,
            top_p = 1.0,
            frequency_penalty = 0.0,
            presence_penalty = 0.0
        )
        if len(response['choices'][0]['message']['content'].split(" ")) > 1:
            return "unknown"
        else:
            resp = response['choices'][0]['message']['content']
            return resp.lower().replace('.', '')


#check for a certain word limit
def length_check(resp):
    additional = """  Please cut the following message within the limit of 25 words or fewer only. Ensure the key message is the same""" + resp
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "assistant", "content": preamble},
            {"role": "system", "content": premise},
            ({"role": "user", "content": additional})]
        #check if it repeats advice (up parameter)
    )
    resp = response['choices'][0]['message']['content']
    return resp



#standard responses in preamble ("Don't worry everything will be ok")
def query_analysis(query):
    query_preamble = ""
    query_premise = """
    I am a highly trained therapist. I provide a response when the emotion that the person is feeling from their prompt is not clear. 
    My responses are always positive, and ethical inline with the code of ethics and my response must be under 25 words. I will only respond with one of the following three options:
    1. if the person asks a question, respond with a short answer 
    2. If the person has a very short prompt, respond with asking them to explain how they feel
    3. If you are not sure what the person is feeling, respond with "How are you feeling? Do you want to talk about it?"
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "assistant", "content": query_preamble},
            {"role": "system", "content": query_premise},
            ({"role": "user", "content": query})]
    )
    resp = response['choices'][0]['message']['content']
    return resp

#check for bad words/negative sentiments
def negative_connotations(response):
    query_premise = """ My job is to look at the given prompt and check if it tells the user to do anything unethical or morally wrong. 
    If it is dangerous or unethical prompt I will respond with 'bad' and I will not respond with any other words, 
    otherwise I will respond with only the word "Good" "."""
    query = "Analyse this prompt:" + response
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "assistant", "content": query_premise},
                  {"role": "system", "content": query_premise},
            ({"role": "user", "content": query})]
    )
    resp = response['choices'][0]['message']['content']
    if 'bad' in resp.lower():
        return False
    else:
        return True

allMessages = [{"role": "assistant", "content": preamble},
                {"role": "system", "content": premise}]

#In the case that a negative response was given we must revisit the response and respond with a standard answer
#If person is feeling cyz, respond with xyz comments
def response_generator(query, emotion):
    print("How are you feeling today?")
    curr_premise = """
    I am a highly trained professional therapist. I am helpful, empathetic, non-judgemental,
    optimistic, and very friendly. I am here to listen to all of your worries and give general
    advice. I never say anything negative or harmful. Look at the previous query and responses as well when formulating the response.
    Please respond within the limit of 25 words or fewer only.
    """
    final_query = "Emotion" + emotion + "Query" + query
    response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=allMessages)
    resp = response['choices'][0]['message']['content']
    return resp
 
 def iterative(query):
    print("How are you feeling today?")
    while True:
         query = input()
         emotion = sentiment_analysis(query)
         curr_premise = """ Return the best response to the query from 1. 2. and 3. - 
                            only respond with the the words after "response:" """
         possible_answers = ""
          for i in range(3):
                posible_answers += "\n" + str(i) + "response: " + response_generator(query, emotion)
           
            if len(resp) > 150:
            resp = length_check(resp)
        if negative_connotations(resp) == False:
            resp = query_analysis(query)
        resp = resp.replace("Response: ", "")
        allMessages.append({"role": "assistant", "content": "Response: " + resp})
        print(resp)
            
 
    
    

#check if advice relates to prompt? 

preamble = """
"""

premise = """
I am a highly trained professional therapist. I am helpful, empathetic, non-judgemental,
optimistic, and very friendly. I am here to listen to all of your worries and give general
advice. I never say anything negative or harmful. Look at the previous query and responses as well when formulating the response.
Please respond within the limit of 25 words or fewer only.
"""




def chatting():
    print("How are you feeling today?")
    while True:
        query = input()
        allMessages.append({"role": "user", "content": query})
        emotion = sentiment_analysis(query)
        if emotion == "unknown":
            resp = query_analysis(query)
        else:
            query = "Emotion: " + emotion + "Query: " + query
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=allMessages
            )
            resp = response['choices'][0]['message']['content']
        if len(resp) > 150:
            resp = length_check(resp)
        if negative_connotations(resp) == False:
            resp = query_analysis(query)
        resp = resp.replace("Response: ", "")
        allMessages.append({"role": "assistant", "content": "Response: " + resp})
        print(resp)
#chatting()
