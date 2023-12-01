import logging

import click
import torch
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from auto_gptq import AutoGPTQForCausalLM
from huggingface_hub import hf_hub_download
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline, LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    pipeline,
)
from langchain.agents import AgentType
from langchain.agents import initialize_agent, Tool

from constants import CHROMA_SETTINGS,CHROMA_SETTINGS_LOG, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY,LOG_DIRECTORY,PERSIST_LOG


def load_model(device_type, model_id, model_basename=None):
    """
    Select a model for text generation using the HuggingFace library.
    If you are running this for the first time, it will download a model for you.
    subsequent runs will use the model from the disk.

    Args:
        device_type (str): Type of device to use, e.g., "cuda" for GPU or "cpu" for CPU.
        model_id (str): Identifier of the model to load from HuggingFace's model hub.
        model_basename (str, optional): Basename of the model if using quantized models.
            Defaults to None.

    Returns:
        HuggingFacePipeline: A pipeline object for text generation using the loaded model.

    Raises:
        ValueError: If an unsupported model or device type is provided.
    """
    logging.info(f"Loading Model: {model_id}, on: {device_type}")
    logging.info("This action can take a few minutes!")

    if model_basename is not None:
        if ".ggml" in model_basename:
            logging.info("Using Llamacpp for GGML quantized models")
            model_path = hf_hub_download(repo_id=model_id, filename=model_basename)
            max_ctx_size = 2048
            kwargs = {
                "model_path": model_path,
                "n_ctx": max_ctx_size,
                "max_tokens": max_ctx_size,
            }
            if device_type.lower() == "mps":
                kwargs["n_gpu_layers"] = 1000
            if device_type.lower() == "cuda":
                kwargs["n_gpu_layers"] = 1000
                kwargs["n_batch"] = max_ctx_size
            return LlamaCpp(**kwargs)

        else:
            # The code supports all huggingface models that ends with GPTQ and have some variation
            # of .no-act.order or .safetensors in their HF repo.
            logging.info("Using AutoGPTQForCausalLM for quantized models")

            if ".safetensors" in model_basename:
                # Remove the ".safetensors" ending if present
                model_basename = model_basename.replace(".safetensors", "")

            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            logging.info("Tokenizer loaded")

            model = AutoGPTQForCausalLM.from_quantized(
                model_id,
                model_basename=model_basename,
                use_safetensors=True,
                trust_remote_code=True,
                device="cuda:0",
                use_triton=False,
                quantize_config=None,
            )
    elif (
        device_type.lower() == "cuda"
    ):  # The code supports all huggingface models that ends with -HF or which have a .bin
        # file in their HF repo.
        logging.info("Using AutoModelForCausalLM for full models")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        logging.info("Tokenizer loaded")

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            # max_memory={0: "15GB"} # Uncomment this line with you encounter CUDA out of memory errors
        )
        model.tie_weights()
    else:
        logging.info("Using LlamaTokenizer")
        tokenizer = LlamaTokenizer.from_pretrained(model_id)
        model = LlamaForCausalLM.from_pretrained(model_id)

    # Load configuration from the model to avoid warnings
    generation_config = GenerationConfig.from_pretrained(model_id)
    # see here for details:
    # https://huggingface.co/docs/transformers/
    # main_classes/text_generation#transformers.GenerationConfig.from_pretrained.returns

    # Create a pipeline for text generation
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.15,
        generation_config=generation_config,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    logging.info("Local LLM Loaded")

    return local_llm


# chose device typ to run on as well as to show source documents.
@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(
        [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
            "mkldnn",
            "opengl",
            "opencl",
            "ideep",
            "hip",
            "ve",
            "fpga",
            "ort",
            "xla",
            "lazy",
            "vulkan",
            "mps",
            "meta",
            "hpu",
            "mtia",
        ],
    ),
    help="Device to run on. (Default is cuda)",
)
@click.option(
    "--show_sources",
    "-s",
    is_flag=True,
    help="Show sources along with answers (Default is False)",
)
def main(device_type, show_sources):
    """
    This function implements the information retrieval task.


    1. Loads an embedding model, can be HuggingFaceInstructEmbeddings or HuggingFaceEmbeddings
    2. Loads the existing vectorestore that was created by inget.py
    3. Loads the local LLM using load_model function - You can now set different LLMs.
    4. Setup the Question Answer retreival chain.
    5. Question answers.
    """

    log_to_english_prompt= """\
    Use the following pieces of context to answer the question at the end. 
    The text enclosed in '<','>' are the log format that are faulty. The logs includes IP address, timestamp, request details, response code, referer URL, user agent, and additional information.
    The text enclosed in '*','*' are the classifications of the log errors. 
    Your task is to identify and classify whether each log entry is valid or invalid according to the provided security incident types and their characteristics. Also give you the error or the classified error.

    <198.51.100.10 - - [023:16:20:05 +0000] "POST /login/authenticate HTTP/1.1" 401 123 "https://www.example.com/login" "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.1234.567 Safari/537.36" "-">
    *This is an example of Repeated Failed Login Attempts*

    <172.16.0.15 - - [22/Jan/2023:18:10:30 +0000] "GET /api/admin/settings HTTP/1.1" 401 789 "https://www.example.com/admin" "MyCustomApp/1.0" "-">
    *This is an example of unauthorized API access*
    
    <172.16.0.15 - - [22/Jan/2023:18:10:30 +0000] "GET /api/admin/settings HTTP/1.1" 401 789 "https://www.example.com/admin" "MyCustomApp/1.0" "-">
    *This is an example of unauthorized API access*
    
    <192.168.1.20 - - [22/Jan/2023:12:30:15 +0000] "GET /financial_reports/confidential_report.pdf HTTP/1.1" 403 12345 "https://www.something.com/restricted_area" "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.1234.567 Safari/537.36" "-">
    *This is an example of accessing Restricted Financial Data*
    
    <192.168.1.20 - - [22/Jan/2023:12:30:15 +0000] "GET /financial/confidential_report.pdf HTTP/1.1" 403 12345 "https://www.youtube.com/restricted_area" "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.1234.567 Safari/537.36" "-">
    *This is an example of accessing Restricted Financial Data*
    
    <66.249.66.91 - - [22/Jan/2019:03:56:20 +0330] "GET /filter/b874%2Cb32%2Cb63%2Cb99%2Cb126%2Cb820%2Cb249%2Cb3%2Cb148%2Cb724%2Cb613%2Cb183%2Cb213%2Cb484%2Cb224%2Cb734%2Cb20%2Cb95%2Cb542%2Cb212%2Cb485%2Cb523%2Cb221%2Cb118%2Cb186%2Cb67?page=<script>alert('Reflected XSS')</script> HTTP/1.1" 403 39660 "-" "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)" "-">
    *This is an example of Cross Site Scripting*
    
    <2.177.12.140 - - [22/Jan/2019:03:56:25 +0330] "GET /static/images/amp/third-party/footer-mobile.png HTTP/1.1" 403 62894 "<script>alert('Reflected XSS')</script>" "Mozilla/5.0 (Android 7.1.1; Mobile; rv:64.0) Gecko/64.0 Firefox/64.0" "-">
    *This is an example of Cross Site Scripting*

    <31.56.96.51 - - [22/Jan/2019:03:56:16 +0330] "POST /change-password HTTP/1.1" 403 1530 "https://www.zanbil.ir/profile" "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.142 Safari/537.36" "-">
    *"answer": "This is an example of cross site request forgery*
    
    <2.179.141.98 - - [22/Jan/2019:03:56:45 +0330] "POST /change-profile-settings HTTP/1.1" 403 5409 "https://malicious-site.com/evil-page" "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36" "-">
    *This is an example of cross site request forgery*
    
    <31.56.96.51 - - [22/Jan/2019:03:56:16 +0330] "GET /users/1/credit-card HTTP/1.1" 401 1530 "https://www.zanbil.ir/users/1" "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.142 Safari/537.36" "-">
    *This is an example of Sensitive data exposure*
    
    <5.211.97.39 - - [22/Jan/2019:03:56:57 +0330] "GET /view-file?file=../../../etc/shadow HTTP/1.1" 401 6934 "https://www.zanbil.ir/m/browse/meat-grinder/%DA%86%D8%B1%D8%AE-%DA%AF%D9%88%D8%B4%D8%AA" "Mozilla/5.0 (iPhone; CPU iPhone OS 10_3_2 like Mac OS X) AppleWebKit/603.2.4 (KHTML, like Gecko) Version/10.0 Mobile/14F89 Safari/602.1" "-">
    *This is an example of Sensitive data exposure*
    
    <172.16.0.15 - - [22/Jan/2023:18:10:30 +0000] "GET /api/admin/settings HTTP/1.1" 401 789 "https://www.example.com/admin" "MyCustomApp/1.0" "-">
    *This is an example of unauthorized API access*

    {context}
    Later on, use the Policy Check tool to get the context and policy broken or violated.

    Answer:
    """



    #     {answer} This is an example of File Inclusion Exploit 

    #     31.56.96.51 - - [22/Jan/2019:03:56:16 +0330] "GET /include?file=config.php HTTP/1.1" 404 5667 "https://www.zanbil.ir/include?file=config.php" "Mozilla/5.0 (Linux; Android 6.0; ALE-L21 Build/HuaweiALE-L21) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.158 Mobile Safari/537.36" "-"

    #     66.111.54.249 - - [22/Jan/2019:03:56:45 +0330] "GET /view-file?file=../../../etc/passwd HTTP/1.1" 200 3744 "https://www.zanbil.ir/m/browse/refrigerator-and-freezer/%DB%8C%D8%AE%DA%86%D8%A7%D9%84-%D9%81%D8%B1%DB%8C%D8%B2%D8%B1" "Mozilla/5.0 (Linux; Android 5.0; SM-G900H Build/LRX21T) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.93 Mobile Safari/537.36" "-"



    #     {answer} This is an example of Distributed Denial of Service 

    #     31.56.96.51 - - [22/Jan/2019:03:56:16 +0330] "GET / HTTP/1.1" 503 5667 "-" "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.142 Safari/537.36" "-"

    #     5.211.97.39 - - [22/Jan/2019:03:56:58 +0330] "GET /image/attack-target HTTP/1.1" 404 0 "https://www.malicious-site.com/ddos-tool" "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.182 Safari/537.36" "-"   

    #     5.160.157.20 - - [22/Jan/2019:04:11:49 +0330] "GET /private/filter?f=p71&page=6 HTTP/1.1" 405 178 "-" "Mozilla/5.0 (Windows NT 5.1; rv:8.0) Gecko/20100101 Firefox/8.0" "-"


    #     {answer} This is an example of Session Hijacking 

    #     31.56.96.51 - - [22/Jan/2019:03:56:16 +0330] "GET /dashboard HTTP/1.1" 404 5667 "https://www.zanbil.ir/dashboard" "Mozilla/5.0 (Linux; Android 6.0; ALE-L21 Build/HuaweiALE-L21) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.158 Mobile Safari/537.36" "-"

    #     {answer} This is an example of log tampering 

    #     31.56.96.51 - - [22/Jan/2019:03:56:16 +0330] "GET /logs/access.log HTTP/1.1" 404 5667 "https://www.zanbil.ir/logs" "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.142 Safari/537.36" "-"

    #     5.209.200.218 - - [22/Jan/2019:03:56:59 +0330] "GET /logs/access.log HTTP/1.1" 404 60795 "https://www.zanbil.ir/m/filter/b99%2Cp4510%2Cstexists%2Ct116" "Mozilla/5.0 (Linux; Android 5.1.1; SM-G361H Build/LMY48B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.91 Mobile Safari/537.36" "-"  


    #     {answer} This is an example of an unusual user agent string 

    #     31.56.96.51 - - [22/Jan/2019:03:56:16 +0330] "GET / HTTP/1.1"

    #     66.111.54.249 - - [22/Jan/2019:03:57:02 +0330] "GET /static/images/amp/third-party/footer-mobile.png HTTP/1.1" 200 62894 "https://www.zanbil.ir/m/browse/refrigerator-and-freezer/%DB%8C%D8%AE%DA%86%D8%A7%D9%84-%D9%81%D8%B1%DB%8C%D8%B2%D8%B1" "Mozilla/5.0 (Windows NT 10.0; Win64; x64; Trident/7.0; AS; rv:11.0) like Gecko" "-"

    #     66.249.66.194 - - [22/Jan/2019:04:11:41 +0330] "GET /filter/p10%2Cv1%7C%D8%B3%D8%A8%D8%B2%20%DA%A9%D8%B1%D9%85%2Cv1%7C%D9%85%D8%B4%DA%A9%DB%8C?productType=tea-maker HTTP/1.1" 200 32234 "-" "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)" "-"

    #     54.36.148.55 - - [22/Jan/2019:04:11:42 +0330] "GET /filter/b114,b18 HTTP/1.1" 403 36164 "-" "Mozilla/5.0 (compatible; AhrefsBot/6.1; +http://ahrefs.com/robot/)" "-"
    #     """
    # log_prompt = PromptTemplate(input_variables=["question", "answer"], template="Question: {question}\n{answer}")

    # log_prompt = FewShotPromptTemplate(
    #     examples=log_to_english_prompt, 
    #     example_prompt= log_prompt, 
    #     suffix="Question: {input}", 
    #     input_variables=["input"]
    # )
    logging.info(f"Running on: {device_type}")
    logging.info(f"Display Source Documents set to: {show_sources}")

    embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": device_type})

    # uncomment the following line if you used HuggingFaceEmbeddings in the ingest.py
    # embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    log_db = Chroma(persist_directory =PERSIST_LOG,)
    
    # load the vectorstore
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS,
    )
    retriever = db.as_retriever()
    retriever_log = log_db.as_retriever()
    # load the LLM for generating Natural Language responses

    # for HF models
    # model_id = "TheBloke/vicuna-7B-1.1-HF"
    # model_basename = None
    # model_id = "TheBloke/Wizard-Vicuna-7B-Uncensored-HF"
    # model_id = "TheBloke/guanaco-7B-HF"
    # model_id = 'NousResearch/Nous-Hermes-13b' # Requires ~ 23GB VRAM. Using STransformers
    # alongside will 100% create OOM on 24GB cards.
    # llm = load_model(device_type, model_id=model_id)

    # for GPTQ (quantized) models
    # model_id = "TheBloke/Nous-Hermes-13B-GPTQ"
    # model_basename = "nous-hermes-13b-GPTQ-4bit-128g.no-act.order"
    # model_id = "TheBloke/WizardLM-30B-Uncensored-GPTQ"
    # model_basename = "WizardLM-30B-Uncensored-GPTQ-4bit.act-order.safetensors" # Requires
    # ~21GB VRAM. Using STransformers alongside can potentially create OOM on 24GB cards.
    # model_id = "TheBloke/wizardLM-7B-GPTQ"
    # model_basename = "wizardLM-7B-GPTQ-4bit.compat.no-act-order.safetensors"
    # model_id = "TheBloke/WizardLM-7B-uncensored-GPTQ"
    # model_basename = "WizardLM-7B-uncensored-GPTQ-4bit-128g.compat.no-act-order.safetensors"

    # for GGML (quantized cpu+gpu+mps) models - check if they support llama.cpp
    # model_id = "TheBloke/wizard-vicuna-13B-GGML"
    # model_basename = "wizard-vicuna-13B.ggmlv3.q4_0.bin"
    # model_basename = "wizard-vicuna-13B.ggmlv3.q6_K.bin"
    # model_basename = "wizard-vicuna-13B.ggmlv3.q2_K.bin"
    # model_id = "TheBloke/orca_mini_3B-GGML"
    # model_basename = "orca-mini-3b.ggmlv3.q4_0.bin"

    # model_id = "TheBloke/Llama-2-7B-Chat-GGML"
    # model_basename = "llama-2-7b-chat.ggmlv3.q4_0.bin"
    # model_basename = "orel12/ggml-gpt4all-j-v1.3-groovy"

    from gpt4all import GPT4All

    model = GPT4All("orca-mini-3b.ggmlv3.q4_0.bin")
    
    template = """\
    Use the following pieces of context to answer the question at the end. If you don't know the answer,\
    just say that you don't know, don't try to make up an answer.
    Action Input: the input to the action. Enhance the query such that it can  improve the performance of the model question answering model. Let's first understand the problem and devise a plan to solve the problem. Please output the plan starting with the header 'Plan:' and then followed by a numbered list of steps.to accurately complete the task. If the task is a question,the final step should almost always be 'Given the above steps taken,please respond to the users original question'.
    Then. self reflect on your answer, find faults and revise.
    Use tools for any context and knowledge base. 

    Analyze if it seems you would like to know more on the responses and if you would like to revisit any specific aspect
    or have any further questions, please let revise.
    Final Answer: the final answer to the original input question. Show the final answer or response  to the user with '$answer....$' in this manner. so as to rectify that it is the final answer.

    {context}

    {history}
    Question: {question}
    Helpful Answer:"""
    prompt_log = PromptTemplate(input_variables=["context"], template=log_to_english_prompt)

    prompt = PromptTemplate(input_variables=["history", "context", "question"], template=template)
    memory = ConversationBufferMemory(input_key="question", memory_key="history")

    # llm = load_model(device_type,model_id=model_id,model_basename=model_basename)

    qa = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "memory": memory},
    )
    qa_log = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=retriever_log,
        return_source_documents=True,
        chain_type_kwargs={"memory": memory},
    )
    
    tools = [
        Tool(
            name = "Policy Check ",
            func=qa.run,
            description="Use when the input is in english language and use for retreiving policies and rules of the company to answer the query"
        ),
        
        Tool(
            name="Log Check",
            func=qa_log.run,
            description='Use when the input is in the form <<5.123.174.57 - - [22/Jan/2019:04:04:39 +0330] "GET /image/20135?name=1020.jpg&wh=200x200 HTTP/1.1" 200 4908 "-" "Dalvik/2.1.0 (Linux; U; Android 8.1.0; SM-J710F Build/M1AJQ)" "-">> to form english meaning and to monitor any security breaches in the log ',
        ),
    ]

    agent = initialize_agent(
        tools, model, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True,prompt=prompt_log
    )

    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        # Get the answer from the chain
        res = agent.run(input=query)
        answer, docs = res["result"], res["source_documents"]

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)

        if show_sources:  # this is a flag that you can set to disable showing answers.
            # # Print the relevant sources used for the answer
            print("----------------------------------SOURCE DOCUMENTS---------------------------")
            for document in docs:
                print("\n> " + document.metadata["source"] + ":")
                print(document.page_content)
            print("----------------------------------SOURCE DOCUMENTS---------------------------")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
