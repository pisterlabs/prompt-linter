import os
import openai

def chat_ai(prompt: str, chat_content: str = "You are an AI assistant that helps people find information.",  temp: float = 0.7, stop_word: str = None, my_engine: str = "GPT"):
    """
    execute LLM interaction using a prompt and applying a role to the AI assistant.

    The prompt contains the following elements:
    - request: the request to the AI (e.g., "Tell me what is the meaning of life, based to the below considerations:")
    - text: the text to analyze (e.g., "Life is a wonderful not one-way journey. It is a journey that we can enjoy only if we are able to understand the meaning of life.")
    """    
    # set openai configuration
    openai.api_type = "azure"
    openai.api_base = "https://saopenai.openai.azure.com/"    
    openai.api_version = "2023-07-01-preview"
    openai.api_key = os.getenv("OPENAI_API_KEY")
    eng = "saGPT"
    mod = None
    
    if my_engine == "LLAMA":        
        # LLAMA
        openai.api_type = "open_ai"
        openai.api_base = "http://51.159.159.233:3001/v1" 
        openai.api_version = ""
        eng = None
        mod = "/models/llama-2-13b-chat.bin"
    
            
    #set api key from environment variable
    openai.api_key = os.getenv("OPENAI_API_KEY")   

    response = openai.ChatCompletion.create(
        engine = eng, 
        model  = mod,
        messages=[
            {
                "role": "system",
                "content": chat_content,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=temp,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=stop_word,
    )

    
    res = response["choices"][0]["message"]["content"]    

    return res



def run_chatbot(eng: str = "GPT"):
    #Define AI role
    chatRole = None #the prompt is enough

    # Define the text of the short story that you want to use as the source of the chatbot    
    text = """
    In un mondo immaginario di nome Aeropolis, quando si attraversano le enormi porte di vetro si entra in un mondo di bellezza e meraviglia.

    Umano: Ciao, chi sei?
    Chatbot: Sono un assistente AI abitante di Aeropolis. Come posso aiutarti oggi?
    Umano:

    """

    #define stop condition
    stop_word = "!STOP!"
    chat = True

    # Define the chat history variable
    chat_history = ""

    # Start a loop to interact with the chatbot until the stop word is used    
    while chat:        
        # Get the user input
        user_input = input("Umano: ")
        if stop_word in user_input:
            exit()
        
        # Append the user input to the chat history
        chat_history += f"Umano: {user_input}\n"

        # Generate the chatbot response using the openAI API
        prompt=f"{text}\n{chat_history}Chatbot:"

        res = chat_ai(prompt, temp = 1, my_engine = eng)

        # Append the response text to the chat history
        chat_history += f"Chatbot: {res}\n"

        # Print the response text
        print(f"Chatbot: {res}")        

    return


run_chatbot("GPT")
