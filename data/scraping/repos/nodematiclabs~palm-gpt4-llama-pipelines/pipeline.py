import kfp.dsl as dsl
from kfp import compiler
from kfp.dsl import Artifact, Input, Output

from typing import List

@dsl.component(
    base_image='python:3.11',
    packages_to_install=['bs4', 'lxml', 'requests']
)
def urls_from_sitemap(sitemap: str) -> List[str]:
    import requests
    from bs4 import BeautifulSoup

    # Send a GET request to the sitemap URL
    response = requests.get(sitemap)

    # If the GET request is successful, the status code will be 200
    if response.status_code == 200:
        # Get the content of the response
        sitemap_content = response.content

        # Create a BeautifulSoup object and specify the parser
        soup = BeautifulSoup(sitemap_content, 'xml')

        # Find all <loc> tags in the sitemap
        loc_tags = soup.find_all('loc')

        # Get the URLs from the <loc> tags and add them to a list
        urls = [tag.text for tag in loc_tags]

        return urls[0:2]

    else:
        print("Failed to retrieve sitemap: ", response.status_code)
        return None

@dsl.component(
    base_image='python:3.11',
    packages_to_install=['bs4', 'requests']
)
def pages_from_urls(urls: List[str], pages: Output[Artifact]):
    import json
    import requests
    from bs4 import BeautifulSoup

    def extract_text(url):
        # Send a GET request to the webpage
        response = requests.get(url)
        # If the GET request is successful, the status code will be 200
        if response.status_code == 200:
            # Get the content of the response
            webpage = response.text
            # Create a BeautifulSoup object and specify the parser
            soup = BeautifulSoup(webpage, "html.parser")
            # Get the text of the webpage
            text = soup.get_text()
            return text
        else:
            raise Exception("Failed to retrieve webpage.")

    texts = {}
    for url in urls:
        texts[url] = extract_text(url)

    with open(pages.path, 'w') as f:
        json.dump(texts, f)

@dsl.component(
    base_image='huggingface/transformers-pytorch-gpu:4.29.2',
    packages_to_install=[
        'git+https://github.com/huggingface/accelerate.git',
        'bs4',
        'huggingface-hub',
        'requests',
        'sentencepiece',
        'tokenizers>=0.13.3',
        'torch',
        'git+https://github.com/huggingface/transformers',
    ]
)
def llama_descriptions(pages: Input[Artifact], output: Output[Artifact]):
    import json
    import torch
    import transformers

    from huggingface_hub import login
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering

    login(token="")

    model = "meta-llama/Llama-2-13b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float32,
        device_map="auto",
    )

    with open(pages.path, 'r') as f:
        page_data = json.loads(f.read())

    descriptions = {}
    for url in page_data.keys():
        keywords = page_data[url].replace("!", " ").replace(".", " ").replace("(", " ").replace(")", " ").replace("-", " ").replace(",", " ").replace(" ", ", ").replace("\n", ", ")
        keywords = ", ".join([keyword for keyword in keywords.split(", ") if keyword and keyword != "|" and keyword != "-"])

        # Respect token limits
        while len(tokenizer.encode(keywords)) > 3072:
            keywords = ", ".join(keywords.split(", ")[1:])
        message = f"Question: What is a short, but good, one-sentence description for a webpage containing the keywords \"{keywords}\"?\nAnswer: "

        # Clear CUDA memory
        torch.cuda.empty_cache()
        sequences = pipeline(
            message,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=1024,
        )

        print(f"Result: {sequences[0]['generated_text']}")
        descriptions[url] = sequences[0]['generated_text']

    with open(output.path, 'w') as f:
        json.dump(descriptions, f)

@dsl.component(
    base_image="python:3.11",
    packages_to_install=["google-cloud-aiplatform", "requests"],
)
def palm_descriptions(pages: Input[Artifact], output: Output[Artifact]):
    import json
    import vertexai

    from vertexai.language_models import ChatModel

    PROJECT_ID = ""
    LOCATION = "us-central1"
    PALM_API_KEY = ""

    vertexai.init(project=PROJECT_ID, location=LOCATION)
    chat_model = ChatModel.from_pretrained("chat-bison@001")
    parameters = {"temperature": 0.2, "max_output_tokens": 1024, "top_p": 0.8, "top_k": 40}

    with open(pages.path, 'r') as f:
        page_data = json.loads(f.read())

    descriptions = {}
    for url in page_data.keys():
        keywords = page_data[url].replace("!", " ").replace(".", " ").replace("(", " ").replace(")", " ").replace("-", " ").replace(",", " ").replace(" ", ", ").replace("\n", ", ")
        keywords = ", ".join([keyword for keyword in keywords.split(", ") if keyword and keyword != "|" and keyword != "-"])

        # Respect token limits
        while len(tokenizer.encode(keywords)) > 3072:
            keywords = ", ".join(keywords.split(", ")[1:])
        message = f"Question: What is a short, but good, one-sentence description for a webpage containing the keywords \"{keywords}\"?\nAnswer: "

        chat = chat_model.start_chat(examples=[])
        response = chat.send_message(message, **parameters)
        descriptions[url] = response.text

    with open(output.path, 'w') as f:
        json.dump(descriptions, f)

@dsl.component(
    base_image="python:3.11",
    packages_to_install=["openai", "tiktoken"],
)
def gpt4_descriptions(pages: Input[Artifact], output: Output[Artifact]):
    import json
    import openai
    import tiktoken

    openai.api_key = ""
    encoding = tiktoken.encoding_for_model("gpt-4")

    with open(pages.path, 'r') as f:
        page_data = json.loads(f.read())

    descriptions = {}
    for url in page_data.keys():
        keywords = page_data[url].replace("!", " ").replace(".", " ").replace("(", " ").replace(")", " ").replace("-", " ").replace(",", " ").replace(" ", ", ").replace("\n", ", ")
        keywords = ", ".join([keyword for keyword in keywords.split(", ") if keyword and keyword != "|" and keyword != "-"])

        # Respect token limits
        while len(encoding.encode(keywords)) > 3072:
            keywords = ", ".join(keywords.split(", ")[1:])
        message = f"Question: What is a short, but good, one-sentence description for a webpage containing the keywords \"{keywords}\"?\nAnswer: "
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{
                "role": "user",
                "content": message
            }],
            max_tokens=1024
        )
        descriptions[url] = completion.choices[0].message

    with open(output.path, 'w') as f:
        json.dump(descriptions, f)

@dsl.component(
    base_image='python:3.11',
    packages_to_install=['json']
)
def log_comparison(
    url: str,
    llama: Input[Artifact],
    palm: Input[Artifact],
    gpt4: Input[Artifact],
):
    import json

    with open(llama.path, 'r') as f:
        llama_data = json.loads(f.read())

    with open(palm.path, 'r') as f:
        palm_data = json.loads(f.read())

    with open(llama.path, 'r') as f:
        gpt4_data = json.loads(f.read())

    print(f"""
        URL: {url}\n\n
        Llama: {llama_data[url]}\n\n
        Palm: {palm_data[url]}\n\n
        GPT-4: {gpt4_data[url]}
    """)

@dsl.pipeline(
    name="website-summarizer",
)
def website_summarizer(sitemap: str):
    urls_from_sitemap_task = urls_from_sitemap(
        sitemap=sitemap
    )
    pages_from_urls_task = pages_from_urls(
        urls=urls_from_sitemap_task.output
    )
    llama_descriptions_task = llama_descriptions(
        pages=pages_from_urls_task.output
    )
    llama_descriptions_task.set_cpu_request("12")
    llama_descriptions_task.set_cpu_limit("12")
    llama_descriptions_task.set_memory_request("64Gi")
    llama_descriptions_task.set_memory_limit("64Gi")
    llama_descriptions_task.set_accelerator_limit("1")
    llama_descriptions_task.set_accelerator_type("NVIDIA_TESLA_A100")
    palm_descriptions_task = palm_descriptions(
        pages=pages_from_urls_task.output
    )
    gpt4_descriptions_task = gpt4_descriptions(
        pages=pages_from_urls_task.output
    )
    with dsl.ParallelFor(
        name="urls",
        items=urls_from_sitemap_task.output,
    ) as url:
        log_comparison(
            url=url,
            llama=llama_descriptions_task.output,
            palm=palm_descriptions_task.output,
            gpt4=gpt4_descriptions_task.output,
        )

compiler.Compiler().compile(website_summarizer, 'pipeline.yaml')