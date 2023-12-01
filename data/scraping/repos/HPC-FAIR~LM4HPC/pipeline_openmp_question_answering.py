import torch
import json
import os
import openai
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM
)
from ._instruct_pipeline import InstructionTextGenerationPipeline
from transformers import pipeline
from ._utils_langchain import (get_pdf_text,
                               get_chunk_text,
                               get_vector_store,
                               get_retrievalQA)

# Load the configuration once when the module is imported
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')
with open(CONFIG_PATH, 'r') as f:
    CONFIG = json.load(f)


def llm_langchain(question, pdf_files, model, langchain_embedding):
    text = get_pdf_text(pdf_files)
    chunks = get_chunk_text(text)
    vectore_store = get_vector_store(chunks, langchain_embedding)
    qa_chain = get_retrievalQA(vector_store=vectore_store, model=model)
    answer = qa_chain(question)
    return answer


def llm_generate_dolly(model: str, question: str, pdf_files='', langchain_embedding='', **parameters) -> str:
    """
    Answer the question using the Dolly model.
    """
    if pdf_files == '':
        tokenizer_pretrained = AutoTokenizer.from_pretrained(
            model, padding_side="left")
        model_pretrained = AutoModelForCausalLM.from_pretrained(
            "databricks/dolly-v2-3b", device_map="auto", torch_dtype=torch.bfloat16)
        generate_text = InstructionTextGenerationPipeline(
            model=model_pretrained, tokenizer=tokenizer_pretrained, **parameters)
        return generate_text(question)[0]["generated_text"].split("\n")[-1]
    else:
        return llm_langchain(question, pdf_files, model, langchain_embedding)


def llm_generate_gpt(model: str, question: str, pdf_files='', langchain_embedding='', **parameters) -> str:
    """
    Answer the question using the GPT model.
    """
    if pdf_files == '':
        msg = [{"role": "system", "content": "You are an OpenMP export."}]
        msg.append({"role": "user", "content": question})
        response = openai.ChatCompletion.create(
            model=model,
            messages=msg,
            **parameters
        )
        return response['choices'][0]['message']['content']
    else:
        return llm_langchain(question, pdf_files, model, langchain_embedding)


def llm_generate_starchat(model: str, question: str, **parameters) -> str:
    """
    Answer the question using the StarChat model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(model,
                                                 load_in_8bit=True,
                                                 device_map='auto'
                                                 )
    system_prompt = "<|system|>\nBelow is a conversation between a human user and an OpenMP expert.<|end|>\n"
    user_prompt = f"<|user|>\n{question}<|end|>\n"
    assistant_prompt = "<|assistant|>"
    full_prompt = system_prompt + user_prompt + assistant_prompt
    inputs = tokenizer.encode(full_prompt, return_tensors="pt").to('cuda')
    outputs = model.generate(inputs,
                             eos_token_id=0,
                             pad_token_id=0,
                             max_length=256,
                             early_stopping=True)
    output = tokenizer.decode(outputs[0])
    output = output[len(full_prompt):]
    if "<|end|>" in output:
        cutoff = output.find("<|end|>")
        output = output[:cutoff]
    return output


def llm_generate_codellama(model_name: str, question: str, **parameters) -> str:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import transformers
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    response = pipeline(question)
    return response[0]["generated_text"]


def openmp_question_answering(model: str, question: str, pdf_files: str = '', langchain_embedding: str = '', **parameters) -> str:
    """
    Generates an answer to a question using the specified model and parameters.

    Parameters:
        model (str): The model to use for question answering. Options are 'databricks/dolly-v2-12b', 'gpt3', and 'starcoder'.
        question (str): The question to answer.
        **parameters: Additional keyword arguments to pass to the `pipeline` function.

    Returns:
        str: The generated answer.

    Raises:
        ValueError: If the model is not valid.
    """
    if model in CONFIG['openmp_question_answering']['models'] and model.startswith('databricks/dolly-v2'):
        response = llm_generate_dolly(
            model, question, pdf_files, langchain_embedding, **parameters)
        return response
    elif model in CONFIG['openmp_question_answering']['models'] and model.startswith('gpt-'):
        response = llm_generate_gpt(
            model, question, pdf_files, langchain_embedding, **parameters)
        return response
    elif model in CONFIG['openmp_question_answering']['models'] and model.startswith('HuggingFaceH4/starchat-'):
        response = llm_generate_starchat(model, question, **parameters)
        return response
    elif model in CONFIG['openmp_question_answering']['models'] and model.startswith('codellama/'):
        response = llm_generate_codellama(model, question, **parameters)
        return response
    else:
        raise ValueError('Unknown model: {}'.format(model))
