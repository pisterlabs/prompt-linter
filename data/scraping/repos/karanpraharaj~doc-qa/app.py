import gradio as gr
import random
import time
import anthropic
from anthropic import AsyncAnthropic
import asyncio

import anthropic_function

import json
import logging
import os
import datetime
from typing import List, Tuple
from tqdm import tqdm
from utils.file_utils import get_root_dir
from sentence_transformers import SentenceTransformer, CrossEncoder

from elasticsearch import Elasticsearch


ROOT_DIR = get_root_dir()
CACHE_DIR = os.path.join(ROOT_DIR, 'hf_cache')

model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2', cache_folder=CACHE_DIR)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", max_length=512)

#connect to a single node and check its health
es_client = Elasticsearch("http://10.0.74.130:9200")


ANTHROPIC_KEY = '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'

c = anthropic.AsyncAnthropic(api_key=ANTHROPIC_KEY)
c_sync = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

anthropic_func = anthropic_function.AnthropicFunction(
    api_key=ANTHROPIC_KEY,
    model="claude-2",
    temperature=0.2,
    max_tokens_to_sample=200
    )

conv_history = ""
interpret_conv_history = ""
retrievals = [""]*10
gen_retrievals = ""
retrieve_state = False
retrievals_current_display_list = [""]*10

current_interpretation = ""

turn = 0

def accordion_visibilty(retrieval_0):    
    print(retrieval_0)
    if retrieval_0 == "":
        return gr.Column.update(visible=False)
    else:
        return gr.Column.update(visible=True)

### Anthropic function calling ###
def get_coreferences(original_sentence):
    # Process a sentence to extract anaphoric references, named entities, their mappings, and the sentence with the anaphoric references replaced with their named entities.
    coreference_info = {
        "original_sentence": original_sentence,
        "coreferences_list": coreferences_list,
        "named_entities_list": named_entities_list,
        "coreference_entity_mapping": coreference_entity_mapping,
        "revised_sentence": revised_sentence
    }
    return json.dumps(coreference_info)

def apply_date_filter(original_sentence):
    # Get a date-time range to use for looking up documents between two dates
    date_range_info = {
        "original_sentence": original_sentence,
        "start_date": start_date,
        "end_date": end_date,
        "revised_sentence": revised_sentence
    }
    return json.dumps(date_range_info)

anthropic_func.add_function(
    "apply_date_filter", "When asked a question that needs looking up documents between two dates, get a date range in YYYY-MM-DD format (e.g. 2001-10-04)). We return the question that was asked, as if it has no date component. We also return the original question, as it was asked, with the date component.",
    ["original_sentence: string", "start_date: string", "end_date: string", "revised_sentence: string"]
)

anthropic_func.add_function(
    "get_coreferences", "When a question has a coreference such as a pronoun or an anaphoric reference, we need to resolve it to a named entity. This function takes a sentence and returns a list of coreferences, a list of named entities, a mapping of coreferences to named entities, and the sentence with the coreferences replaced with their named entities. For example if the first question is 'How long has Jeff Skilling worked at Enron?' and the second question is 'What is his title at the company?', the  coreferences are 'his' and 'the company'. The named entities are 'Jeff Skilling' and 'Enron'. The mapping is {'his': 'Jeff Skilling', 'the company': 'Enron'}. The sentence with the coreferences replaced with their named entities is 'What is Jeff Skilling's title at Enron?'",
    ["original_sentence: string", "coreferences_list: list", "named_entities_list: list", "coreference_entity_mapping: dict", "revised_sentence: string"]
)

### End of Anthropic function calling ###


with gr.Blocks(css="footer {visibility: hidden}") as demo:
    title = gr.Markdown("<h1 style='font-family: Arial Black; color: rgb(186, 47, 96); font-size: 30px;'> reveal </h1>")

    with gr.Tab("Chat"):
        # messages = ""
        with gr.Row(equal_height=False):
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Reveal Chat")
                msg = gr.Textbox(label="Enter message:")
                interpretation = gr.Textbox(label="Interpretation:")
                with gr.Row(equal_height=True):
                    with gr.Column():
                        submit_btn = gr.Button("Submit", variant="primary", size="lg", min_width=5)
                    with gr.Column():
                        retrieve_check = gr.Checkbox(label="Look up docs", info="Retrieve a new set of documents?", min_width=5)
                clear = gr.Button("Start new conversation", size="lg", min_width=1)
            with gr.Column(scale=1, visible=False) as col:        
                for i in range(10):
                    with gr.Accordion(f"Document {i+1}", open=False) as accordion:
                        retrievals[i] = gr.Markdown("")

    # with gr.Tab("Events"):
    #     # Create a dropdown for start date and end date
    #     date = gr.Dropdown([i for i in range(1, 32)], label="Date", min_width=5)
    #     month = gr.Dropdown([i for i in range(1, 13)], label="Month", min_width=5)
    #     year = gr.Dropdown([i for i in range(1980, 2010)], label="Year", min_width=5)


    ## Chatbot
    
    def user(user_message, history):
        print(user_message)
        return gr.update(value="", interactive=True), history + [[user_message, None]]

    def set_switch_state(retrieve_check):
        global retrieve_state
        retrieve_state = retrieve_check
        print("SWITCH STATE: ", retrieve_check)

    def claude_user(user_message, conv_history):
        global turn
        turn += 1
        print("TURN from user. Overall turn: ", turn)
        return gr.update(value="", interactive=True), conv_history + f"{anthropic.HUMAN_PROMPT} {user_message} {anthropic.AI_PROMPT}:"
    
    def query_elastic(user_query, history):
        global retrieve_state
        global gen_retrievals
        global retrievals_current_display_list

        print("QUERY ELASTIC: ", retrieve_state)

        conv_history = history + [[user_query, None]]
        # print("USERRRRRRRRRRRRRRR CONV HISTORY: ", conv_history)

        interpreted_user_query = claude_interpret_sync(conv_history)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~INTERPRETED USER QUERY: ", interpreted_user_query)

        unranked_text = []
        result_text_list = []

        gen_retrievals = ""
        
        if retrieve_state:
            print("USER QUERY: ", user_query)
            query_embedding = model.encode(user_query.strip())

            query_string = {
                        "field": "embedding",
                        "query_vector": query_embedding,
                        "k": 50,
                        "num_candidates": 100,
                    }
            
            results = es_client.search(
                        index="enron_r11_concept_search", knn=query_string, size=50)
            
            doc_id_list = []
            repeated_text_count = 0

            for i, hit in enumerate(results['hits']['hits']):
                if hit["_source"]["text"] not in unranked_text:
                    doc_id = hit['_source']['document_id']
                    doc_id_list.append(doc_id)

                    # results_str = f'<code style="font-size: larger;">{doc_id}</code> <br>' + hit["_source"]["text"] + '\n\n'
                    unranked_text.append(hit["_source"]["text"])
            print("Length of unranked text: ", len(unranked_text))
            
            if len(unranked_text) < 10:
                # print("Extra docs needed!!!!!!!!!!!!!")
                num_extra_docs = 10 - len(unranked_text)
                # print("Num extra docs: ", num_extra_docs)
                
                for i, hit in enumerate(results['hits']['hits'][10:]):
                    if hit["_source"]["text"] not in unranked_text:
                        doc_id = hit['_source']['document_id']
                        doc_id_list.append(doc_id)

                        # results_str = f'<code style="font-size: larger;">{doc_id}</code> <br>' + hit["_source"]["text"] + '\n\n'
                        unranked_text.append(hit["_source"]["text"])

                # print("Length of unranked text: ", len(unranked_text))

            else:
                print("No extra docs needed!!!!!!!!!!!!!")
                    
            unranked_text = unranked_text
            # doc_id_list = doc_id_list[:10]
            
            # Rerank the results with the cross-encoder
            scores = reranker.predict([(user_query, unranked_doc) for unranked_doc in unranked_text])
            
            reranked_doc_score_tuples = []
            for i in range(len(unranked_text)):
                reranked_doc_score_tuples.append((unranked_text[i], scores[i]))
            
            reranked_docs = sorted(reranked_doc_score_tuples, key=lambda x: x[1], reverse=True)

            # Save reranked_docs in a file.
            with open("reranked_docs.txt", "w") as f:
                for doc in reranked_docs:
                    f.write(doc[0] + "\n\n")

            reranked_doc_ids = [x for _, x in sorted(zip(scores, doc_id_list), key=lambda pair: pair[0], reverse=True)]

            reranked_docs_text = [doc[0] for doc in reranked_docs]

            for i, doc in enumerate(reranked_docs_text):
                results_str = f'<code style="font-size: larger">{reranked_doc_ids[i]}</code><br>' + doc + '\n\n'
                if i < 5:
                    gen_retrievals += f'Document {i+1}: {doc} \n\n'
                result_text_list.append(results_str)

            retrievals_current_display_list = result_text_list

        else:
            result_text_list = retrievals_current_display_list

        return result_text_list

    def reset_turns(*retrievals):
        global turn
        turn = 0
        global conv_history
        conv_history = ""
        global interpret_conv_history
        interpret_conv_history = ""
        retrievals = [""]*10

        # print(f"TURNS BACK TO {turn}")
        # print(f"CONV HISTORY BACK TO NONE: {conv_history}")
        # print(f"INTERPRET CONV HISTORY BACK TO NONE: {interpret_conv_history}")

        return retrievals

    async def claude_bot(history):
        global conv_history
        global interpret_conv_history
        global gen_retrievals
        global retrieve_state
        global turn
        global current_interpretation

        if turn == 0:
            print("FIRST TURN. Instructing model...")

            instruction = "You are a polite and well-spoken assistant to legal attorneys to help them in e-discovery. You will start the first answer in the conversation with 'Hi! My name is Claude and I am an e-discovery assistant created by Reveal.' Your job is to review relevant documents when they are provided and answer questions about them. \
                        If you are given a question, you must answer it accurately with sources (if any). If it is a casual conversation, do not provide sources. \
                        If you are given a question and a set of documents, you must answer the question based on the documents. \
                        Your answers will be based on the documents (when they are provided). They will have two sections separated by a line: Answer and Source. \
                        When you are not given a set of documents, you can answer the question based on previously provided documents. \n\n \
                        "
        else:
            instruction = ""
        
        if retrieve_state: 
            retrieval_prompt = f"Here are 5 documents. \n\n" + \
                            "Documents: \n" + \
                            f"{gen_retrievals} \n\n" + \
                            f"Based on these documents, can you answer the following question: \n\n"
        else:
            retrieval_prompt = "Answer the following question. You can use any of the documents that may have been provided to you to come up with an answer. \n\n"

        user_message = history[-1][0]
        interpreted_user_message = current_interpretation

        print("CLAUDE BOT USER MESSAGE: ", user_message)
        print("CLAUDE BOT INTERPRETED USER MESSAGE: ", interpreted_user_message)

        prompt = conv_history + f"{anthropic.HUMAN_PROMPT} {instruction} {retrieval_prompt} {interpreted_user_message} {anthropic.AI_PROMPT} Answer:"
        interpret_prompt = interpret_conv_history + f"{anthropic.HUMAN_PROMPT} {user_message} {anthropic.AI_PROMPT} Answer:"

        bot_message = await c.completions.create(
                            prompt= prompt,
                            stop_sequences=[anthropic.HUMAN_PROMPT],
                            max_tokens_to_sample=4000,
                            model="claude-2",
                            temperature=0.2,
                            stream=True,
                        )
        
        turn += 1

        history[-1][1] = ""
        current_completion = ""
        
        async for data in bot_message:
            # delta = data["completion"][len(current_completion) :] # Incremental text
            # current_completion = data["completion"]
            delta = data.completion
            history[-1][1] += delta
            yield history
        
        conv_history = f"{prompt} {history[-1][1]}"
        interpret_conv_history = f"{interpret_prompt} {history[-1][1]}"
        # print("================CONV HISTORY:=======\n ", conv_history)


    def claude_interpret_sync(history):
        global interpret_conv_history
        global current_interpretation
        global turn

        print("CLAUDE INTERPRET SYNC TURN: ", turn) 

        instruction = "Read the conversation and confidently provide the user with your interpretation of the question without asking for clarifications. Only provide an interpretation of the question in the form of a paraphrased question. Prioritize recent questions over older ones (if any) to interpret the current question. Do not try and answer it. Do not try and add additional words that are not a part of the question such as 'Based on the documents provided'. For example:\n\n\nQuestion: Who is XYZ?\n\nAssistant: Interpretation: Who is XYZ?\n\n\n\Question: Who is his brother?\n\nAssistant: Interpretation: Who is XYZ's brother?\n\n\nQuestion:Did he have concerns about valuation at his company?\n\nAssistant: Interpretation: Did XYZ have concerns about valuation at the company XYZ worked at?\n\n\n"                

        user_message = history[-1][0]
        interpret_prompt = interpret_conv_history + f"{anthropic.HUMAN_PROMPT} {instruction}Now show your interpretation for this question based on the conversation above:\n\n\nQuestion: {user_message}\n\n{anthropic.AI_PROMPT} Interpretation: "

        bot_interpret = c_sync.completions.create(
                            prompt= interpret_prompt,
                            stop_sequences=[anthropic.HUMAN_PROMPT],
                            max_tokens_to_sample=4000,
                            model="claude-2",
                            temperature=0.0,
                        )
        
        interpret_history = history
        interpret_history[-1][1] = bot_interpret.completion        
        
        interpret_conv_history = f"{interpret_prompt} {interpret_history[-1][1]}"
        print("================INTERPRET CONV HISTORY:=======\n ", interpret_conv_history)
        print('\n\n=========================== INTERPRETATION: ', bot_interpret.completion, '\n\n')
        
        current_interpretation = bot_interpret.completion

        return current_interpretation

    async def claude_interpret(history):
        global current_interpretation
        # global conv_history
        # global interpret_conv_history

        # instruction = "Read the conversation and confidently provide the user with your interpretation of the question without asking for clarifications. Only provide an interpretation of the question in the form of a paraphrased question. Prioritize recent questions over older ones (if any) to interpret the current question. Do not try and answer it. Do not try and add additional words that are not a part of the question such as 'Based on the documents provided'. For example:\n\n\nQuestion: Who is XYZ?\n\nAssistant: Interpretation: Who is XYZ?\n\n\n\Question: Who is his brother?\n\nAssistant: Interpretation: Who is XYZ's brother?\n\n\nQuestion:Did he have concerns about valuation at his company?\n\nAssistant: Interpretation: Did XYZ have concerns about valuation at the company XYZ worked at?\n\n\n"                

        # user_message = history[-1][0]
        # interpret_prompt = interpret_conv_history + f"{anthropic.HUMAN_PROMPT} {instruction}Now show your interpretation for this question based on the conversation above:\n\n\nQuestion: {user_message}\n\n{anthropic.AI_PROMPT} Interpretation: "

        # bot_interpret = await c.completions.create(
        #                     prompt= interpret_prompt,
        #                     stop_sequences=[anthropic.HUMAN_PROMPT],
        #                     max_tokens_to_sample=4000,
        #                     model="claude-2",
        #                     temperature=0.0,
        #                     stream=True,
        #                 )
        
        # interpret_history = history
        # interpret_history[-1][1] = ""
        
        # async for data in bot_interpret:
        #     # delta = data["completion"][len(current_completion) :] # Incremental text
        #     # current_completion = data["completion"]
        #     delta = data.completion
        #     interpret_history[-1][1] += delta
        #     yield interpret_history[-1][1]
        
        # interpret_conv_history = f"{interpret_prompt} {interpret_history[-1][1]}"
        # print("================INTERPRET CONV HISTORY:=======\n ", interpret_conv_history)

        return current_interpretation


    response = msg.submit(set_switch_state, retrieve_check, None).then(fn=query_elastic, inputs=[msg, chatbot], outputs=retrievals).then(
            claude_interpret, chatbot, interpretation).then(
            accordion_visibilty, retrievals[0], col, queue=False).then(
            user, [msg, chatbot], [msg, chatbot], queue=False).then(
            claude_bot, chatbot, chatbot)

    response.then(lambda: gr.update(interactive=True), None, [msg], queue=False)
    
    submit_btn.click(set_switch_state, retrieve_check, None).then(fn=query_elastic, inputs=[msg, chatbot], outputs=retrievals).then(
            claude_interpret, chatbot, interpretation).then(
            accordion_visibilty, retrievals[0], col, queue=False).then(
            user, [msg, chatbot], [msg, chatbot], queue=False).then(
            claude_bot, chatbot, chatbot).then(
            lambda: gr.update(interactive=True), None, [msg], queue=False)

    clear.click(lambda: None, None, chatbot, queue=False).then(reset_turns, retrievals, retrievals, queue=False).then(accordion_visibilty, retrievals[0], col, queue=False)

demo.queue()
demo.launch(share=True, debug=True)
