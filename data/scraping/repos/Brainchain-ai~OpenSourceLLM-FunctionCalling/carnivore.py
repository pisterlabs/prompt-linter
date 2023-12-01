import json
import os
import requests
import tiktoken
from langchain.document_loaders import UnstructuredURLLoader
from icecream import ic

def gpt(
    user_prompt: str = None,
    system_prompt: str = "You are an autoregressive language model that has been fine-tuned with instruction-tuning and RLHF. You carefully provide accurate, factual, thoughtful, nuanced answers, and are brilliant at reasoning. If you think there might not be a correct answer, you say so. Since you are autoregressive, each token you produce is another opportunity to use computation, therefore you always spend a few sentences explaining background context, assumptions, and step-by-step thinking BEFORE you try to answer a guestion.",
    backend: str = "azure",
    model: str = "gpt-4-32k",
    temperature: float = 0.0,
    top_p: float = 0.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    history: list = [],
    n: int = 1,
    separator: str = "\n\n",
    latest_only: bool = False,
    choices_only: bool = False,
):
    if backend != "azure" and backend != "openai":
        raise Exception("backend = 'azure' or 'openai' only 😡")
    
    if backend == "azure":
        from promptlayer import openai
        import promptlayer
        promptlayer.api_key = os.environ.get("PROMPTLAYER_API_KEY")
        openai.api_type = "azure"
        openai.api_base = "https://brainchainuseast2.openai.azure.com/"
        openai.api_version = "2023-08-01-preview"
        openai.api_key = os.environ.get("AZURE_OPENAI_API_KEY")

        if type(history) == str:
            history = json.loads(history.replace("'",'"'))
        
        if len(history) == 0:
            history.append({"role": "system", "content": system_prompt})
        history.append({"role": "user", "content": user_prompt})
        
        response = openai.ChatCompletion.create(
            engine=f"{model}", 
            messages=history, 
            n=n,
            temperature=temperature,
            top_p=top_p, 
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty
        )

        if latest_only:
            response["choices"] = [response["choices"][-1]]
        
        if choices_only:
            return [msg["message"]["content"].replace("\"","'") for msg in response["choices"]]

        history.append(
            {
                "role": "assistant",
                "content": response["choices"][0]["message"]["content"]
            }
        )
        return history
    
    if backend == "openai":
        import promptlayer
        from promptlayer import openai
        promptlayer.api_key = os.environ.get("PROMPTLAYER_API_KEY")
        openai.api_key = os.environ.get("AZURE_OPENAI_API_KEY")

        response = openai.ChatCompletion.create(
            model=model, messages=history, n=n
        )
        history.append(
            {
                "role": "assistant",
                "content": [msg["message"]["content"].replace("\"","'") for msg in response["choices"]],
            }
        )
        return history

class CarnivoreClient:
    def __init__(self, env: str = "prod"):
        if env == "prod":
            self.service_url = "https://brainchain--carnivore.modal.run/"
        elif env == "dev":
            self.service_url = "https://brainchain--carnivore-dev.modal.run/"

    def scrape(self, link, tags, tag_delimiter=" "):
        payload = {"base_url": link, "tags": ["a", "p"], "use_google_web_cache": False, "exclude_tags": ["html", "script", "style"]}
        response = requests.post(
            self.service_url, json=payload, headers={"accept": "application/json"}
        )
        print(response.content)
        return response.content

    def content(self, link, tags=["a", "p"], return_string=True):
        extracted = []
        print("Processing: ", end="")
        for tag in tags:
            print(f"[ <{tag}> ... </{tag}> ], ", end="\n")
            content = self.scrape(link, tag)["content"]
            if tag in content:
                for item in content[tag]:
                    print(f"{item}")
                    extracted.append(item)
        if return_string:
            return ''.join(extracted)
        else:
            extracted

    def condense(self, page):
        from langchain.text_splitter import TokenTextSplitter
        content = '\n'.join(self.content(page, return_string=True))
        system_prompt = "Condense the following information taken from a page scrape. Do not omit any details. Just synthesize the information appropriately."
        agg = []
        text_splitter = TokenTextSplitter(
            chunk_size=16000, chunk_overlap=800
        )
        chunks = text_splitter.split_text(content, disallowed_special=())

        for text in chunks:
            print("TEXT! ", text)
            agg.append(gpt(system_prompt="Summarize this text, retaining all important information", user_prompt=text,history=[], model="gpt-3.5-turbo-16k", backend="azure")[-1]["content"])

        # print("Agg is... ", agg)
        yy = "\n".join(agg)
        # print("YY len: ", len(yy))
        # print("YY contents: ", yy)

        return gpt(system_prompt="Coalesce all this information into a nice summary that preserves all useful facts.", user_prompt=yy, model="gpt-3.5-turbo-16k", backend="azure", history=[])[-1]["content"]
    
    def content_analyzer(self, link):
        return self.analyze_content(link)

    def analyze_content(self, link, tags="*", tag_delimiter=" ", backend="azure", model="gpt-4-32k"):
        contents_metadata = {}
        enc = tiktoken.encoding_for_model(model)
        content_store = []
        running_length = 0
        extracted = []

        loader = UnstructuredURLLoader(urls=[link], mode="single", show_progress_bar=True)
        d = loader.load()
        content = "".join([item.page_content for item in d])
        print("Content is... \n", content)
        
        tokens_in_doc = len(enc.encode(content))
        # ic(content)


        if tokens_in_doc <= 16000:
                # Call the ChatCompletion API to fact-check
                chat = gpt(system_prompt="You are unstructured web soup content summarizer. Clean up and extract all relevant info from this unstructured mess.", user_prompt=content, model="gpt-4-32k", backend="azure")
                ic("Chat is... ", chat, "\n\n")
                return chat[-1]["content"]
        else:
            return "This text is too long. Please use the fts_document_qa_tool with specific questions to extract information from this text."
                
    def links(self, link):
        page = self.scrape(link, "a")
        links = page["external_links"] + page["links"]
        return list(filter(lambda x: x.startswith("http"), links))

    def slurp(self, link):
        return self.scrape(link, "*")


