import json
import pandas as pd
import openai
import numpy as np
import json
import requests
# from openai.embeddings_utils import distances_from_embeddings, cosine_similarity

openai.api_key = ""

# Read in the dataframe with vector embeddings
df = pd.read_csv('processed/embeddings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

print(f'max token in df: {df.n_tokens.max()}')
print(f'text: {df.iloc[df.n_tokens.idxmax()].text}')

# def lambda_handler(event, context):
#     print("Event received:", event)  # Add print statement
#     if event['request']['type'] == "LaunchRequest":
#         return on_launch()
#     elif event['request']['type'] == "IntentRequest":
#         return on_intent(event['request']['intent'])


def lambda_handler(event, context):
    print("Event received:", event)
    if event['request']['type'] == "LaunchRequest":
        return on_launch()
    elif event['request']['type'] == "IntentRequest":
        intent = event['request']['intent']

        if 'name' in intent:  # Add this check
            intent_name = intent['name']

            if intent_name == "AskQuestionIntent":
                print("AskQuestionIntent")
                return on_intent(intent)
            elif intent_name == "AMAZON.StopIntent":
                print('StopIntent')
                return on_stop_intent()
            elif intent_name == "AMAZON.YesIntent":
                print('YesIntent')
                return on_yes_intent()
            elif intent_name == "AMAZON.NoIntent":
                print('NoIntent')
                return on_no_intent()
            elif intent_name == "AMAZON.CancelIntent":
                print('CancelIntent')
                return on_cancel_intent()
            elif intent_name == "AMAZON.HelpIntent":
                print('HelpIntent')
                return on_help_intent()
        else:
            return {
                "version": "1.0",
                "response": {
                    "outputSpeech": {
                        "type": "PlainText",
                        "text": "Please ask a question."
                    },
                    "shouldEndSession": False
                }
            }

def on_cancel_intent():
    return {
        "version": "1.0",
        "response": {
            "outputSpeech": {
                "type": "PlainText",
                "text": "Do you have a additional request for me?"
            },
            "shouldEndSession": False
        }
    }

def on_help_intent():
    return {
        "version": "1.0",
        "response": {
            "outputSpeech": {
                "type": "PlainText",
                "text": "Sure, You can simply ask me any question and I'll try to answer it based on my knowledge base. If you want to exit the skill, simply say 'STOP'. What would you like to know?"
            },
            "shouldEndSession": False
        }
    }


def on_yes_intent():
    return {
        "version": "1.0",
        "response": {
            "outputSpeech": {
                "type": "PlainText",
                "text": "Great! Please ask your next question."
            },
            "shouldEndSession": False
        }
    }

def on_no_intent():
    return {
        "version": "1.0",
        "response": {
            "outputSpeech": {
                "type": "PlainText",
                "text": "Thank you for using my chatbot! Goodbye!"
            },
            "shouldEndSession": True
        }
    }



def on_launch():
    return {
        "version": "1.0",
        "response": {
            "outputSpeech": {
                "type": "PlainText",
                "text": "Welcome to my chatbot! Please ask a question?"
            },
            "shouldEndSession": False
        }
    }

def on_intent(intent):
    print("Intent received:", intent)  # Add print statement
    user_input = intent['slots']['Question']['value']
    print("User input:", user_input)  # Add print statement
    response_text = answer_question(df, user_input)
    print("Response text:", response_text)  # Add print statement
    
    return {
        "version": "1.0",
        "response": {
            "outputSpeech": {
                "type": "PlainText",
                "text": response_text
            },
            "shouldEndSession": False
        }
    }


def on_stop_intent():
    return {
        "version": "1.0",
        "response": {
            "outputSpeech": {
                "type": "PlainText",
                "text": "Thank you for using my chatbot! Goodbye!"
            },
            "shouldEndSession": True
        }
    }



def cosine_distance(u, v):
    """
    Compute the cosine distance between two vectors.

    Parameters
    ----------
    u : array_like
        Input array.
    v : array_like
        Input array.

    Returns
    -------
    cosine_distance : float
        The cosine distance between the two vectors.
    """

    # Convert input arrays to numpy arrays
    u = np.array(u)
    v = np.array(v)

    # Compute dot product
    dot = np.dot(u, v)

    # Compute norms of u and v
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    # Compute cosine distance
    cosine_distance = 1 - dot / (norm_u * norm_v)

    return cosine_distance


# Define a function to create a context for a question
def create_context(question, df, max_len=2000, size="ada"):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    # df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')
    df['distances'] = [cosine_distance(q_embeddings, emb) for emb in df['embeddings'].values]
    print(df.distances)
    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        # Add the length of the text to the current length
        print(f'row: {row["text"]}')
        print(f'n_tokens: {row["n_tokens"]}')
        print(len(row['text']))
        cur_len += row['n_tokens'] + 4
        print(f'current_length: {cur_len}')
        # If the context is too long, break
        if cur_len > max_len:
            break
        
        # Else add it to the text that is being returned
        returns.append(row["text"])
        print(returns)
    # Return the context
    return "\n\n###\n\n".join(returns)

# Define a function to answer a question based on the context
def answer_question(df, question, model="text-davinci-003", max_len=3000, size="ada", max_tokens=350, stop_sequence=None):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(question, df, max_len=max_len, size=size)
    
    print("Context:", context)  # Add print statement
    try:
        # if model == 'Davinci-GPT-3':
        #     model = "text-davinci-003"
        # Create a completions using the question and context
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
            {"role": "system", "content": "You are a helpful assistant. If someone asks about your identify then just respond normally without the context provided below. End your response with asking 'what more do you want to know?'"},
            {"role": "user", "content": f"Answer the question (in your own words) mostly based on the context below (keep your response short and concise), and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer: "},
            ],
             temperature = 0.5
            )
        return response["choices"][0]['message']['content']


        # response = openai.Completion.create(
        #     prompt=f"You are helpful assistant. Answer the question (in your own words) based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
        #     temperature=0.1,
        #     max_tokens=max_tokens,
        #     top_p=1,
        #     frequency_penalty=0,
        #     presence_penalty=0,
        #     stop=stop_sequence,
        #     model='text-davinci-003',
        # )
        # print("API response:", response)  # Add print statement
        # return response["choices"][0]["text"].strip()
        
        # else:
        #     pass
            # response = openai.ChatCompletion.create(
            #     model="gpt-3.5-turbo",
            #     messages=[
            #     {"role": "system", "content": "You are a helpful assistant."},
            #     {"role": "user", "content": f"Answer the question (in your own words) mostly based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:"},
            #     ],
            #      temperature = 0.5
            #     )
            # return response["choices"][0]['message']['content']
    
    except Exception as e:
        print("Exception in answer_question:", e)  # Add print statement
        return ""

