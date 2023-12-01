# Third-Party Imports
import streamlit as st
import openai
import langchain
from index_functions import load_data
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
from llama_index import SimpleDirectoryReader

# Main function to generate responses from OpenAI's API, not considering indexed data
def generate_response(prompt, history, model_name, temperature):
    # Fetching the last message sent by the chatbot from the conversation history
    chatbot_message = history[-1]['content']

    # Fetching the first message that the user sent from the conversation history
    first_message = history[1]['content']

    # Fetching the last message that the user sent from the conversation history
    last_user_message = history[-1]['content']

    # Constructing a comprehensive prompt to feed to OpenAI for generating a response
    full_prompt = f"{prompt}\n\
    ### The original message: {first_message}. \n\
    ### Your latest message to me: {chatbot_message}. \n\
    ### Previous conversation history for context: {history}"

    # Making an API call to OpenAI to generate a chatbot response based on the constructed prompt
    api_response = openai.ChatCompletion.create(
        model=model_name,  # The specific OpenAI model to use for generating the response
        temperature=temperature,  # The 'creativity' setting for the response
        messages=[  # The list of message objects to provide conversation history context
            {"role": "system", "content": full_prompt},  # System message to provide instruction
            {"role": "user", "content": last_user_message}  # The last user message to generate a relevant reply
        ]
    )
    
    # Extracting the generated response content from the API response object
    full_response = api_response['choices'][0]['message']['content']

    # Yielding a response object containing the type and content of the generated message
    yield {"type": "response", "content": full_response}

# Similar to generate_response but also includes indexed data to provide more context-aware and data-driven responses
def generate_response_index(prompt, history, model_name, temperature, chat_engine):
    # Fetching the last message sent by the chatbot from the conversation history
    chatbot_message = history[-1]['content']

    # Fetching the first message that the user sent from the conversation history
    first_message = history[1]['content']

    # Fetching the last message that the user sent from the conversation history
    last_user_message = history[-1]['content']

    # Constructing a comprehensive prompt to feed to OpenAI for generating a response
    full_prompt = f"{prompt}\n\
    ### The original message: {first_message}. \n\
    ### Your latest message to me: {chatbot_message}. \n\
    ### Previous conversation history for context: {history}"
    
    # Initializing a variable to store indexed data relevant to the user's last message
    index_response = ""

    # Fetching relevant indexed data based on the last user message using the chat engine
    response = chat_engine.chat(last_user_message)
    
    # Storing the fetched indexed data in a variable
    index_response = response.response

    # Adding the indexed data to the prompt to make the chatbot response more context-aware and data-driven
    full_prompt += f"\n### Relevant data from documents: {index_response}"

    # Making an API call to OpenAI to generate a chatbot response based on the constructed prompt
    api_response = openai.ChatCompletion.create(
        model=model_name,  # The specific OpenAI model to use for generating the response
        temperature=temperature,  # The 'creativity' setting for the response
        messages=[  # The list of message objects to provide conversation history context
            {"role": "system", "content": full_prompt},  # System message to provide instruction
            {"role": "user", "content": last_user_message}  # The last user message to generate a relevant reply
        ]
    )
    
    # Extracting the generated response content from the API response object
    full_response = api_response['choices'][0]['message']['content']
    
    # Yielding a response object containing the type and content of the generated message
    yield {"type": "response", "content": full_response}

#################################################################################
# Additional, specific functions I had in the Innovation CoPilot for inspiration:

# Function returns a random thanks phrase to be used as part of the CoPilots reply
# Note: Requires a dictionary of 'thanks phrases' to work properly
def get_thanks_phrase():
    selected_phrase = random.choice(thanks_phrases)
    return selected_phrase

# Function to randomize initial message of CoPilot
# Note: Requires a dictionary of 'initial messages' to work properly
def get_initial_message():
    initial_message = random.choice(initial_message_phrases)
    return initial_message

# Function to generate the summary; used in part of the response
def generate_summary(model_name, temperature, summary_prompt):
    summary_response = openai.ChatCompletion.create(
        model=model_name,
        temperature=temperature,
        messages=[
            {"role": "system", "content": "You are an expert at summarizing information effectively and making others feel understood"},
            {"role": "user", "content": summary_prompt},
        ]
    )
    summary = summary_response['choices'][0]['message']['content']
    print(f"summary: {summary}, model name: {model_name}, temperature: {temperature})")
    return summary

# Function used to enable 'summary' mode in which the CoPilot only responds with bullet points rather than paragraphs
def transform_bullets(content):
    try:
        prompt = f"Summarize the following content in 3 brief bullet points while retaining the structure and conversational tone (using wording like 'you' and 'your idea'):\n{content}"
        response = openai.ChatCompletion.create(
            model="gpt-4",
            temperature=.2,
            messages=[
                {"role": "system", "content": prompt}
            ],
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(response)
        print("Error in transform_bullets:", e)
        return content  # Return the original content as a fallback

# Function to add relevant stage specific context into prompt
def get_stage_prompt(stage):
      #Implementation dependent on your chatbots context
      return

# Function to grade the response based on length, relevancy, and depth of response
def grade_response(user_input, assistant_message, idea):
      #Implementation dependent on your chatbots context
      return      

# Function used to generate a final 'report' at the end of the conversation, summarizing the convo and providing a final recomendation
def generate_final_report():
      #Implementation dependent on your chatbots context
      return
