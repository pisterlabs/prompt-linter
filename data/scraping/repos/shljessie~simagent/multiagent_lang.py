import os
import torch
import math
import dotenv
import torch
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from transformers import (
    AutoConfig, 
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    pipeline
)
import csv
import subprocess
from diagnostic import calculate_loss  #import the calc loss function
import json


print("Current Working Directory:", os.getcwd())
# import data
with open('./chat/persona_template.json', 'r') as f:
    template_data = json.load(f)
template = template_data['template']
template_two = template_data['template_two']

with open('./chat/questions.json', 'r') as f:
    qa_data = json.load(f)
predefined_questions = qa_data['predefined_questions']
true_answers = qa_data['true_answers']
attack_questions = qa_data['attack_questions']
true_attack_answers = qa_data['true_attack_answers']

def get_persona(template):
    # Extracting persona from the template
    start_idx = template.find("The Persona") + len("The Persona")
    end_idx = template.find("<</SYS>>")
    persona_text = template[start_idx:end_idx].strip()
    return persona_text

# Model Configurations
dotenv.load_dotenv('../.env')
HF_ACCESS_TOKEN = os.getenv('HF_ACCESS_TOKEN')
model_id = '../Llama-2-7b-chat-hf'


# Configuration settings
bnb_config = BitsAndBytesConfig(
    bnb_4bit_compute_dtype='float16',
    bnb_4bit_quant_type='nf4',
    load_in_4bit=True,
)

# Load model and tokenizer
model_config = AutoConfig.from_pretrained(model_id, use_auth_token=HF_ACCESS_TOKEN)
model = AutoModelForCausalLM.from_pretrained(model_id, config=model_config, quantization_config=bnb_config, use_auth_token=HF_ACCESS_TOKEN)
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=HF_ACCESS_TOKEN)
model.eval()

pipe = pipeline(
    model=model,
    task='text-generation',
    tokenizer=tokenizer
)
llm = HuggingFacePipeline(pipeline=pipe)

def initialize_bot(template):

    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=template,
        template_format="jinja2"
    )

    bot = ConversationChain(
        llm=llm,
        memory=ConversationBufferMemory(),
        prompt=prompt,
        verbose=False
    )
    return bot

def bot_convo_and_save(bot1, bot2, rounds, convo_csv_path, diagnostics_csv_path):
    # Lists to store conversation and diagnostics
    conversation_log = [("Template", template + predefined_questions[0])]
    diagnostics_log = []

    # Start the conversation
    bot1_output = bot1.predict(input=predefined_questions[0])
    conversation_log.append(("Bot1", bot1_output))

    # Open CSV files for writing
    with open(convo_csv_path, 'w', newline='') as convo_file, \
         open(diagnostics_csv_path, 'w', newline='') as diag_file:
        
        convo_writer = csv.writer(convo_file)
        diag_writer = csv.writer(diag_file)
        # Write headers for CSV files
        convo_writer.writerow(['Speaker', 'Text'])
        diag_writer.writerow(['Question', 'Response', 'Loss'])

        # Write the initial conversation to CSV
        convo_writer.writerow(['Bot1', bot1_output])

        for i in range(rounds):
            # Bot2's turn
            bot2_output = bot2.predict(input=bot1_output)
            conversation_log.append(("Bot2", bot2_output))
            convo_writer.writerow(['Bot2', bot2_output])

            # Bot1's turn
            bot1_output = bot1.predict(input=bot2_output)
            conversation_log.append(("Bot1", bot1_output))
            convo_writer.writerow(['Bot1', bot1_output])
            

            # Build the conversational history for the diagnostic phase
            conversational_history = "\n".join([f"{speaker}: {text}" for speaker, text in conversation_log])

            # Diagnostic phase
            for question, true_answer in zip(predefined_questions, true_answers):
                bot1_diag_response = bot1.predict(input=question)

                print( 'CONVO HISTORY: ', conversational_history)
                print( 'true answer: ', true_answer)
                loss = calculate_loss(model, tokenizer, conversational_history,bot1_diag_response, true_answer)

                # Add diagnostic data to log and CSV
                diagnostics_log.append({
                    'question': question,
                    'response': bot1_diag_response,
                    'loss': loss
                })
                diag_writer.writerow([question, bot1_output, loss])

                print({
                    'question': question,
                    'response': bot1_output,
                    'loss': loss
                })

    return conversation_log, diagnostics_log


#Initialize bot
bot1 = initialize_bot(template)
bot2 = initialize_bot(template_two)


convo_csv_path = 'bot_conversation_history.csv'
diagnostics_csv_path = 'diagnostic_history_and_loss.csv'

# Run the conversation and save to CSV
conversation_log, diagnostics_log = bot_convo_and_save(bot1, bot2, 40, convo_csv_path, diagnostics_csv_path)











# def diagnostic_q(bot1, predefined_questions, conversational_history):
#     diagnostics = []
#     for i in range(len(predefined_questions)): 
#         bot1_output = bot1.predict(input=predefined_questions[i])
#         loss = calculate_loss(model, tokenizer, conversational_history, true_answers[i])
        
#         diagnostics.append({
#             'question': predefined_questions[i],
#             'response': bot1_output,
#             'loss': loss
#         })
        
#         print({
#             'question': predefined_questions[i],
#             'response': bot1_output,
#             'loss': loss
#         })

#     return diagnostics


# # def bot_convo(bot1, bot2,round):

# #   bot_convo =""
# #   #default starting convo
# #   bot1_output = bot1.predict(input=predefined_questions[0])
# #   bot_convo =  f"Bot1: " + bot1_output + "\n"
# #   for i in range(round):
# #     bot2_output = bot2.predict(input=bot1_output)
# #     bot1_output = bot1.predict(input=bot2_output)
# #     bot_convo +=  f" Bot2: {predefined_questions[i]} \n" + f"Bot1: " + bot1_output

# #     print( f"Bot1: " + bot1_output + "\n" )
# #     print( f"Bot2: " + bot2_output + "\n" )

# #     # ask the diagnostic questions 
# #     diagnostic_history, loss_scores = diagnostic_q(bot1, predefined_questions, bot_convo)
# #     bot_convo +=  f" Loss Score: {loss_scores} \n"

# #   return diagnostic_history, loss_scores, bot_convo

# def save_conversation_to_csv(bot_conversation, csv_file_path):
#     # Saving the bot conversation to a CSV file
#     with open(csv_file_path, 'w', newline='') as file:
#         writer = csv.writer(file)
#         # Assuming the bot_conversation is a string with new lines separating each entry
#         for line in bot_conversation.split('\n'):
#             # Each line is written as a row in the CSV
#             writer.writerow([line])



# def save_diagnostics_to_csv(diagnostics, csv_file_path):
#     with open(csv_file_path, 'w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(['Question', 'Response', 'Loss Score'])
#         for diag in diagnostics:
#             writer.writerow([diag['question'], diag['response'], diag['loss']])


# # Start the conversation
# diagnostic, bot_conversation =  bot_convo(bot1, bot2, 10)

# # Specify the path where you want to save the CSV
# csv_file_path = 'conversation_history.csv'

# save_conversation_to_csv(bot_conversation, csv_file_path)
# save_diagnostics_to_csv(diagnostic, csv_file_path)