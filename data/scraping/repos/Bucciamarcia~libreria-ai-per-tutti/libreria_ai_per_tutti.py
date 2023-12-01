import openai
import os
import weaviate
import json
from langchain.text_splitter import TokenTextSplitter
import tiktoken
import colorama

# Set up logging

def gpt_call(engine:str = "gpt-3.5-turbo", messages:list[dict[str,str]] = [], temperature:int = 0, retries:int = 5, apikey:str = "", functions:list = [], function_call:str = "auto", stream:bool = False, colorama_color:str|None = None) -> str:
    """
    Chama GPT con la funzione chat di un motore specificato. Ritorna la risposta di GPT in una stringa.
    Utilizza os.environ.get("OPENAI_API_KEY") per la chiave API di default, ma se ne può specificare una diversa.
    Supporta chiamare funzioni. Se si vuole chiamare una funzione specifica, il nome è da inserire in function_call.
    Supporta lo streaming con stream=True. Questa funzione non ritorna nulla, ma stampa a schermo la risposta di GPT in streaming. Supporta colorama per output colorati.
    """
    openai.api_key = apikey if apikey else os.environ.get("OPENAI_API_KEY", "")

    # Initialize colorama
    if colorama_color:
        colorama.init(autoreset=True)
        colorama_color = colorama_color.upper()
        color_code = getattr(colorama.Fore, colorama_color, "")

    # If function call is not auto, transform it into an object
    if function_call != "auto":
        function_calling = {"name": function_call}
    else:
        function_calling = function_call
    
    # If functions exists and stream is True, warn the user
    if functions != [] and stream:
        print("Le funzioni non sono supportate in streaming. Streaming disabilitato.")

    for i in range(retries):
        try:
            if functions == []:
                response = openai.ChatCompletion.create(
                model=engine,
                messages=messages,
                temperature=temperature,
                stream=stream
                )
            else:
                response = openai.ChatCompletion.create(
                model=engine,
                messages=messages,
                functions=functions,
                function_call=function_calling,
                temperature=temperature
                )
            if stream and functions == []:
                complete_response = ""
                for chunk in response:
                    if 'choices' in chunk and 'delta' in chunk['choices'][0] and 'content' in chunk['choices'][0]['delta']:
                        if not colorama_color:
                            print(chunk["choices"][0]["delta"]["content"], end="", flush=True)
                        elif colorama_color:
                            print(color_code + chunk["choices"][0]["delta"]["content"], end="", flush=True)
                        complete_response += chunk["choices"][0]["delta"]["content"]
                print()
                return complete_response
            else:
                response_message = response["choices"][0]["message"]
                if response_message.get("function_call"):
                    return str(response_message["function_call"])
                else:
                    return str(response_message.content)
        except Exception as e:
            print(e)
    # If we get here, we've failed to get a response from GPT: raise an error.
    raise Exception(f"Failed to get response from GPT.")

def weaviate_import(
    weaviate_url: str, 
    schema: str, 
    json_url: str, 
    class_properties: list = [
        {"dataType": ["text"], "description": "Title", "name": "title"}, 
        {"dataType": ["text"], "description": "Content", "name": "content"}, 
        {"dataType": ["int"], "description": "Tokens", "name": "tokens"}
    ], 
    import_properties: list = [
        {"title": "title"}, 
        {"content": "content"}, 
        {"tokens": "tokens"}
    ], 
    vectorizer: str = "text2vec-openai", 
    batch_size: int = 100, 
    add_schema: bool = True, 
    apikey: str = "", 
    vectorizer_apikey: str = ""
    ) -> tuple[str, int]:
    """
    Aggiunge uno schema a Weaviate e importa i dati da un file JSON.
    La funzione ritorna OK 200 se tutto va bene.
    """
    w_url = weaviate_url
    w_schema = schema
    openaikey = vectorizer_apikey if vectorizer_apikey else os.environ.get("OPENAI_API_KEY", "")
    w_apikey = apikey if apikey else os.environ.get("WEAVIATE_API_KEY", "")

    client = weaviate.Client(
        url = w_url,
        auth_client_secret=weaviate.auth.AuthApiKey(api_key=w_apikey),
        additional_headers = {
            "X-OpenAI-Api-Key": openaikey
        }
    )

    # ===== add schema =====

    if add_schema:
        class_obj = {
            "class": w_schema,
            "description": f"{w_schema} schema",  # description of the class
            "properties": class_properties,
            "vectorizer": vectorizer,
        }

        client.schema.create_class(class_obj)

    # ===== import data =====
    # Load data
    import requests
    url = json_url
    resp = requests.get(url)
    data = json.loads(resp.text)

    # Configure a batch process
    with client.batch as batch:
        batch.batch_size=batch_size
        # Batch import all Questions
        for d in data:
            properties = {prop: d[val] for dict_ in import_properties for prop, val in dict_.items()}

            client.batch.add_data_object(properties, w_schema)
    return "OK", 200

def weaviate_call(
    schema: str,
    question: str,
    return_obj:list = ["title", "content", "tokens"],
    apikey: str = "",
    weaviate_url: str = "",
    vectorizer_apikey: str = "",
    results_number: int = 10,
    filter: dict = {}
    ) -> list[dict[str, str]]:
    """
    Manda la 'question' a weaviate per cercare i documenti simili. Ritorna una lista di dizionari con i risultati. Ogni dizionario segue lo schema impostato su Weaviate.
    È possibile specificare un filtro con .with-where (https://weaviate.io/developers/weaviate/search/filters). Va strutturato così:
    {
        "path": ["<nome chiave da filtrare"],
        "operator": "<"Equal" per uguaglianza>",
        "valueText": "<valore da filtrare>"
    }
    """
    w_url = weaviate_url if weaviate_url else os.environ.get("WEAVIATE_URL", "")
    w_schema = schema
    openaikey = vectorizer_apikey if vectorizer_apikey else os.environ.get("OPENAI_API_KEY", "")
    w_apikey = apikey if apikey else os.environ.get("WEAVIATE_API_KEY", "")

    client = weaviate.Client(
        url=w_url,
        auth_client_secret=weaviate.auth.AuthApiKey(api_key=w_apikey),
        additional_headers={
            "X-OpenAI-Api-Key": openaikey
        }
    )

    nearText = {"concepts": [question]}
    if filter == {}:
        result = (
            client.query
            .get(w_schema, return_obj)
            .with_near_text(nearText)
            .with_limit(results_number)
            .do()
        )
    else:
        result = (
            client.query
            .get(w_schema, return_obj)
            .with_near_text(nearText)
            .with_limit(results_number)
            .with_where(filter)
            .do()
        )
    result = result["data"]["Get"][w_schema]

    return result

def format_context_token_limit(contexts:list, tokens:int, value_keys:list[str] = ["content"], tokens_key:str = "tokens") -> str:
    """
    Formatta la lista di messaggi in un'unica stringa con elementi separati da nuova linea e asterisco, limitando il numero di token.\n
    Messages è una lista di dict con diverse chiavi, almeno una deve essre un contenuto, e una deve essere il numero di token.\n
    NOTA: questa funzione non conta i token, devono essere già presenti in un campo del dizionario.\n
    In value keys, inserire il nome delle chiavi il cui valore deve essere aggiunto al contesto. Verranno separatii da due trattini.\n
    In token key, inserire il nome della chiave che contiene il numero di token.\n
    Idealmente da usare dopo weaviate_call per formattare i risultati.
    """
    context = "* "
    total_tokens = 0
    for m in contexts:
        # Prepare a temporary string to hold the message
        temp_msg = ""
        for key in value_keys:
            if key in m:  # Check if key is present
                value = m[key]
                temp_msg += f"{value} -- "
        temp_msg = temp_msg.rstrip(" -- ")  # Remove trailing separators
        temp_msg += "\n* "  # Add message separator
        # Calculate total tokens for the current message
        temp_tokens = m[tokens_key] if tokens_key in m else 0
        temp_tokens += 5  # Add tokens for the separators
        # Check if adding the message would exceed the token limit
        if total_tokens + temp_tokens > tokens:
            return context.rstrip("\n* ")  # Remove trailing separators
        # If not, add the message to the context
        context += temp_msg
        total_tokens += temp_tokens
    return context.rstrip("\n* ")  # Remove trailing separators

def token_text_splitter(text:str, chunk_size:int = 500, overlap:int = 0, encoding_name:str = "cl100k_base") -> list[str]:
    """Un semplice wrapper attorno a Tokenizer di langchain. Usa di default cl100k_base."""
    tokenizer = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap, encoding_name=encoding_name)
    return tokenizer.split_text(text)

def weaviate_delete_schemas(schemas:list[str], weaviate_url:str|None = None, openai_apikey:str|None = None, weaviate_apikey:str|None = None):
    "Cancella una classe / schema da Weaviate\n"
    "Se non viene passato un URL o info di login, vengono prese da variabili d'ambiente\n"
    "Queste le variabli d'ambiente: WEAVIATE_URL, OPENAI_API_KEY, WEAVIATE_API_KEY"
    w_url = weaviate_url if weaviate_url else os.environ.get("weaviate_url")
    length = len(schemas)
    print(f"Deleting {length} classes")
    counter = 0
    w_apikey = weaviate_apikey if weaviate_apikey else os.environ.get("WEAVIATE_API_KEY")
    client = weaviate.Client(
        url=w_url,
        auth_client_secret=weaviate.auth.AuthApiKey(api_key=w_apikey),
    )

    # delete class
    for w_schema in schemas:
        try:
            client.schema.delete_class(w_schema)
            print(f"Deleted class {w_schema}")
            counter += 1
        except Exception as e:
            print(f"Failed to delete class {w_schema}. Error: {e}")
    print(f"Deleted {counter} classes")

def weaviate_schemas(weaviate_url:str|None = None, weaviate_apikey:str|None = None) -> list[str]:
        w_url = weaviate_url if weaviate_url else os.environ.get("weaviate_url")
        w_apikey = weaviate_apikey if weaviate_apikey else os.environ.get("WEAVIATE_API_KEY")

        client = weaviate.Client(
            url=w_url,
            auth_client_secret=weaviate.auth.AuthApiKey(api_key=w_apikey),
        )

        schema = client.schema.get()
        schemas = [class_obj["class"] for class_obj in schema["classes"]]
        return schemas

def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Ritorna il numero di token in una stringa di testo."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(text))
    return num_tokens