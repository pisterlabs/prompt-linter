import gradio as gr

from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferWindowMemory
from langchain.vectorstores import Chroma
from langchain import PromptTemplate, LLMChain

from optimum.bettertransformer import BetterTransformer
from optimum.intel import OVModelForCausalLM

from transformers import AutoTokenizer, pipeline

from typing import Dict, Any

import torch


# class AnswerConversationBufferMemory(ConversationBufferMemory):
class AnswerConversationBufferMemory(ConversationBufferWindowMemory):
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        return super(AnswerConversationBufferMemory, self).save_context(inputs,{'response': outputs['result']})


def clean_text(text):
    # Remove excessive whitespace
    cleaned_text = ' '.join(text.split())
    # Keep max one newline character
    cleaned_text = cleaned_text.replace('\n\n', '\n')

    return cleaned_text


def chatbot_llm_response(llm_response):
    text = clean_text(llm_response['result']) + '\nSources:\n'
    for source in llm_response["source_documents"]:
        text += source.metadata['source'] + '\n'

    return text

model_name = "databricks/dolly-v2-3b"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
model = OVModelForCausalLM.from_pretrained(model_name)

generate_text = pipeline(model=model,
                         torch_dtype=torch.bfloat16,
                         trust_remote_code=True,
                         device_map="auto",
                         accelerator="bettertransformer",
                         return_full_text=True,
                         max_new_tokens=256,
                         top_p=0.95,
                         top_k=50)

prompt = PromptTemplate(
    input_variables=["instruction"],
    template="{instruction}")

hf_pipeline = HuggingFacePipeline(pipeline=generate_text)
llm_chain = LLMChain(llm=hf_pipeline, prompt=prompt)

# top #2 when task = Retrieval June 2023 for under ~500 MB
model_name = "intfloat/e5-base-v2"
hf = HuggingFaceEmbeddings(model_name=model_name)

# Load up Vector Database
persist_directory = 'db'
vectordb = Chroma(persist_directory=persist_directory,
                   embedding_function=hf)
vectordb.get()
retriever = vectordb.as_retriever(search_kwargs={'k':3})

# Configure Conversation Chain
memory = AnswerConversationBufferMemory(k=3)
qa_chain_with_memory = RetrievalQA.from_chain_type(llm=hf_pipeline,
                                                   chain_type="stuff",
                                                   retriever=retriever,
                                                   return_source_documents=True,
                                                   memory=memory)
# try to set the tone
template = '''
You are the assistant to a tradesperson with knowledge of the Ontario Building Code. You provide specific details using the context given and the users question. 
If you don't know the answer, you truthfully say you don't know and don't try to make up an answer. 
----------------
{context}

Question: {question}
Helpful Answer:'''

qa_chain_with_memory.combine_documents_chain.llm_chain.prompt.template = template


examples = ["What are the requirements for plumbing venting and drainage systems?",
            "Summarize the electrical code regulation for wiring commercial buildings",
            "Tell me the maximum allowable span for floor joists in residential construction",
            "I'm looking for guidelines for fire assemblies and walls in tall buildings",
            "What are the insulation requirements in new residential constructions?"]


def process_example(args):
    for x in generate(args):
        pass
    return x


def generate(instruction):
    response = qa_chain_with_memory(instruction)
    processed_response = chatbot_llm_response(response)

    result = ""
    for word in processed_response.split(" "):
        result += word + " "
        yield result


with gr.Blocks(analytics_enabled=False) as demo:
    with gr.Column():
        gr.Markdown("""# Dolly-Expert-Lite         
                    Dolly-Expert-Lite is a bot for domain specific question 
                    answering. Currently powered by the new Dolly-v2-3b open 
                    source model. It's expert systems in the era of LLMs!

                    ## Building Code Expert 
                    In this example deployment, Dolly-Expert-Lite retrieves 
                    information via a vector database made using the 
                    [Ontario (Canada) Building Code](https://www.buildingcode.online) 
                    sitemap LangChain loader. For details on the original Dolly 
                    v2 model, please refer to the 
                    [model card](https://huggingface.co/databricks/dolly-v2-12b)

                    ### Type in the box below and click to ask the expert!

      """
                    )

        with gr.Row():
            with gr.Column(scale=3):
                instruction = gr.Textbox(placeholder="Enter your question here", label="Question", elem_id="q-input")

                with gr.Box():
                    gr.Markdown("**Answer**")
                    output = gr.Markdown(elem_id="q-output")
                submit = gr.Button("Generate", variant="primary")
                clear = gr.Button("Clear", variant="secondary")

                gr.Examples(
                    examples=examples,
                    inputs=[instruction],
                    cache_examples=False,
                    fn=process_example,
                    outputs=[output],
                )

    submit.click(generate, inputs=[instruction], outputs=[output])
    clear.click(lambda: None, [], [output])
    instruction.submit(generate, inputs=[instruction], outputs=[output])

demo.queue(concurrency_count=16).launch(debug=True)
demo.launch()