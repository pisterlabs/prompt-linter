from langchain.cache import InMemoryCache
import langchain
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.prompts import FewShotPromptTemplate
from langchain import OpenAI, PromptTemplate
from langchain.chat_models import ChatOpenAI
import sys
import logging
import json
import re
import os


def execute_editor_chain(input:str, instruction: str, operation:str):
   
    llm = OpenAI(temperature=.5)

    f = open('experiments/examples-generated.json')
    examples = json.load(f)

    example_prompt = PromptTemplate(
        input_variables=["document", 
                        "operation", 
                        "instruction",
                        "thought", 
                        "action", 
                        "edited_document", "output"],
        template="Document: {document}\nOperation: {operation}\nInstruction: {instruction}\nThought: {thought}\nAction: {action}\nEdited Document: {edited_document}\nOutput: {output}",
    )

    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        OpenAIEmbeddings(),
        FAISS,
        k=1
    )

    prompt_prefix = """
You are an AI writing assistant. You can help make additions and updates to a wide variety of documents.
Edit the document below to complete the task. If you can't complete the task, say "ERROR: I'm sorry, I can't help with this."

You should follow this format:

Document: this is the original document.
Operation: this is the operation the user wants you to perform.
Instruction: this is the instruction given by the user. Use this to guide the Operation.\
Thought: You should always think about what to do.
Action: this is the action you need to take to complete this task. Should be one of [insert, remove, update, expand, or condense].
Edited Document: The document after you have applied the action to the Action Target.
Output: Just the changed/new portion of the document (the difference between the Edited Document and the original Document). This is what you need to return.
"""

    similar_prompt = FewShotPromptTemplate(
        # We provide an ExampleSelector instead of examples.
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=prompt_prefix + "\nFor example:\n",
        suffix="###\n\nDocument: {input}\nOperation: {operation}\nInstruction: {instruction}\nThought:",
        input_variables=["input", "instruction", "operation"],
    )

    zero_shot_template = prompt_prefix + """
###
Document: {input}
Operation: {operation}
Instruction: {instruction}
Thought:
    """

    zero_shot_prompt = PromptTemplate(
        template=zero_shot_template,
        input_variables=["input", "instruction", "operation"],
    )

    chain = LLMChain(llm=llm, prompt=similar_prompt, verbose=True)
    # add a try except block to catch errors
    try:
        completion = chain.predict(
            input=input, instruction=instruction, operation=operation)
        return completion
    except:
        completion = "I'm sorry, I can't help with this."

    return completion