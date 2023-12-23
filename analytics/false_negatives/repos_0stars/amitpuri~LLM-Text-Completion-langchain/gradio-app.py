import os
import gradio as gr
import openai

import google.generativeai as palm
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage, BaseOutputParser

#from dotenv import load_dotenv
#load_dotenv()


llm_api_options = ["OpenAI API","Azure OpenAI API","Google PaLM API", "Llama 2"]
TEST_MESSAGE = "Write an introductory paragraph to explain Generative AI to the reader of this content."
openai_models = ["gpt-4", "gpt-4-0613", "gpt-4-32k", "gpt-4-32k-0613", "gpt-3.5-turbo",
                      "gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-16k-0613", "text-davinci-003", 
                     "text-davinci-002", "text-curie-001", "text-babbage-001", "text-ada-001"]

google_palm_models = ["models/text-bison-001", "models/chat-bison-001","models/embedding-gecko-001"]

temperature = 0.7

def compose_prompt():
    template = ("You are a helpful assistant that answers this question.")
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    return chat_prompt 


def azure_openai_text_completion(prompt: str,
                   model: str,
                   api_key: str,
                   azure_endpoint: str = None,
                   deployment_name: str = None
                   ):
    try:        
        openai_api_base = f"https://{azure_endpoint}.openai.azure.com"
        chat_prompt = compose_prompt()
        chat = AzureChatOpenAI(openai_api_type = "azure",                           
                                openai_api_key = api_key,
                                openai_api_base = openai_api_base,
                                deployment_name = deployment_name,
                                model = model,
                                temperature = temperature, 
                                openai_api_version="2023-05-15")
        llm_response = chat(
                chat_prompt.format_prompt(
                    text=prompt
                ).to_messages()
            )
        return "", llm_response.content
    except Exception as exception:
        print(f"Exception Name: {type(exception).__name__}")
        print(exception)
        return f" openai_text_completion Error - {exception}", ""

def openai_text_completion(prompt: str,
                   model: str,
                   api_key: str
                   ):
    try:        
        chat = ChatOpenAI(openai_api_key=api_key,
                          model=model,
                          temperature=temperature)

        chat_prompt = compose_prompt()
        
        llm_response = chat(
            chat_prompt.format_prompt(
                text=prompt
            ).to_messages()
        )
        return "", llm_response.content
    except Exception as exception:
        print(f"Exception Name: {type(exception).__name__}")
        print(exception)
        return f" openai_text_completion Error - {exception}", ""



def palm_text_completion(google_palm_key: str, prompt: str, model: str):
    try:        
        candidate_count = 1
        top_k = 40
        top_p = 0.95
        max_output_tokens = 1024
        palm.configure(api_key=google_palm_key)
        defaults = {
                  'model': model,
                  'temperature': temperature,
                  'candidate_count': candidate_count,
                  'top_k': top_k,
                  'top_p': top_p,
                  'max_output_tokens': max_output_tokens,
                  'stop_sequences': [],
                  'safety_settings': [{"category":"HARM_CATEGORY_DEROGATORY","threshold":1},{"category":"HARM_CATEGORY_TOXICITY","threshold":1},{"category":"HARM_CATEGORY_VIOLENCE","threshold":2},{"category":"HARM_CATEGORY_SEXUAL","threshold":2},{"category":"HARM_CATEGORY_MEDICAL","threshold":2},{"category":"HARM_CATEGORY_DANGEROUS","threshold":2}],
                }
        
        response = palm.generate_text(
          **defaults,
          prompt=prompt
        )
        return "", response.result
    except Exception as exception:
        print(f"Exception Name: {type(exception).__name__}")
        print(exception)
        return f" palm_text_completion Error - {exception}", ""

def test_handler(optionSelection, 
                 openai_key, 
                 azure_openai_key, 
                 azure_openai_api_base, 
                 azure_openai_deployment_name, 
                 google_generative_api_key, 
                 prompt: str = TEST_MESSAGE, 
                 openai_model_name: str ="gpt-4", 
                 google_model_name: str ="models/text-bison-001"):
    match optionSelection:
        case  "OpenAI API":
            message, response = openai_text_completion(prompt,openai_model_name, openai_key)
            return message, response
        case  "Azure OpenAI API":
            message, response = azure_openai_text_completion(prompt,openai_model_name, azure_openai_key, azure_openai_api_base, azure_openai_deployment_name)
            return message, response
        case  "Google PaLM API":
            message, response = palm_text_completion(google_generative_api_key, prompt,google_model_name)
            return message, response
        case  "Llama 2":
            return f"{optionSelection} is not yet implemented!", ""
        case _:
            if optionSelection not in llm_api_options:
                return ValueError("Invalid choice!"), ""

        

with gr.Blocks() as LLMDemoTabbedScreen:
    with gr.Tab("Text-to-Text (Text Completion)"):
        llm_options = gr.Radio(llm_api_options, label="Select one", info="Which service do you want to use?", value="OpenAI API")
        with gr.Row():
            with gr.Column(): 
                test_string = gr.Textbox(label="Try String", value=TEST_MESSAGE, lines=5)
                test_string_response = gr.Textbox(label="Response", lines=5)
                test_string_output_info = gr.Label(value="Output Info", label="Info")
                test_button = gr.Button("Try it")
    with gr.Tab("API Settings"):
        with gr.Tab("Open AI"):
            openai_model = gr.Dropdown(openai_models, value="gpt-4", label="Model", info="Select one, for Natural language")
            openai_key = gr.Textbox(label="OpenAI API Key", type="password")
        with gr.Tab("Azure Open AI"):
            with gr.Row():
                with gr.Column():                    
                    azure_openai_key = gr.Textbox(label="Azure OpenAI API Key", type="password")
                    azure_openai_api_base = gr.Textbox(label="Azure OpenAI API Endpoint")
                    azure_openai_deployment_name = gr.Textbox(label="Azure OpenAI API Deployment Name")
        with gr.Tab("Google PaLM API"):
            with gr.Row():
                with gr.Column():
                    google_model_name = gr.Dropdown(google_palm_models, value="models/text-bison-001", label="Model", info="Select one, for Natural language") 
                    google_generative_api_key = gr.Textbox(label="Google Generative AI API Key", type="password")
        
    test_button.click(
            fn=test_handler,
            inputs=[llm_options, 
                    openai_key, 
                    azure_openai_key, 
                    azure_openai_api_base, 
                    azure_openai_deployment_name, 
                    google_generative_api_key, 
                    test_string, 
                    openai_model, 
                    google_model_name],
            outputs=[test_string_output_info, test_string_response]
    )
    
if __name__ == "__main__":
    LLMDemoTabbedScreen.launch()