"""
Functionality to get summaries in the form of bulletpoints and formal reports from
either transcribed audio recordings from workers or by handwritten-notes taken
from the workers in the form of bullet points.

Input (defined as arbitrary values in the current script):
input_text: str, the text that the NLP pipeline is supposed to process.
usecase: str, either A or B. A is for the case that we input transcriptions of audio data
        B is for the case that we provide the bulletpoints that were extracted from the
        handwriting of the worker.
date: str, the date to append to the automatically inferred title 
lambda_consistency: float, weighting term of the consistency error
lambda_faith:       float, weighting term of the faithfulness error 

Output:
bullet_list:    [str], list of strings with each bullet point as an element
summary_report: str, summary of the input text in formal language and in paragraph form
title: str, inferred report title based on input text with the current date prepended

total_error: custom error term to measure how consistent the model is and how faithful the outputs are to the
            original input.
"""
import numpy as np
import cohere
import sys

sys.path.append("..")
sys.path.append(".")
from env import *

co = cohere.Client(COHERE_KEY) # pls don't steal my API key

# Hyperparameters
lambda_consistency = 0.5
lambda_faith = 0.5

def process_case_A(input_text, date=None):

 # Hacky way of dealing with very short input (cohere expects 250 characters minimum)
 if len(input_text)<= 250:
    input_text = input_text + " " * (250 - len(input_text)) 

 # Get bulletpoints from input text
    summary_bullets = co.summarize(
        text= input_text,
        length="long",
        format="bullets",
        model="summarize-xlarge",
        extractiveness="low",
        temperature=0, # between 0 and 5, between 0 and 1 gives good results, high values cause more "creative" answers
        additional_command="Use formal and technical language, use passive tense and focus only on objective facts.",
    )

    # Processing the generated bulletpoints
    summary_str = summary_bullets.summary
    summary_str = summary_str.replace("-","") # remove bullet point bullets
    # bullet_list = summary_str.split("\n") # turn into list of strings for nice processing in the app
    bullet_list = summary_str
    summary_str = summary_str.replace("\n", "") # remove new lines

    # Get report format summary from input text
    summary_report = co.summarize(
        text=input_text,
        length="long",
        format="paragraph",
        model="summarize-xlarge",
        extractiveness="low",
        temperature=0, # between 0 and 5, between 0 and 1 gives good results, high values cause more "creative" answers
        additional_command="Use formal and technical language, use passive tense and focus only on objective facts."
    )

    summary_report = summary_report.summary

    # Get embeddings of the texts
    embed_init = co.embed(
        texts=[input_text], # API only accepts list of strings as input
        model='small',
        truncate="NONE"
    )    
    embed_bullet = co.embed(
        texts=[summary_str],
        model='small',
        truncate="NONE"
    )

    embed_summary = co.embed(
        texts=[summary_report],
        model='small',
        truncate="NONE"
    )

    # The returned things are objects, we want them as numpy arrays
    embed_init = np.array(embed_init.embeddings[0]) 
    embed_bullet = np.array(embed_bullet.embeddings[0]) 
    embed_summary = np.array(embed_summary.embeddings[0]) 

    # Compute final error
    err_consistency = np.linalg.norm(embed_bullet - embed_summary)
    err_faith = np.linalg.norm(embed_init - embed_summary)

    error = lambda_consistency * err_consistency + lambda_faith * err_faith

    print("===========")
    print(f"Original input: {input_text}")
    print("===========")
    print(f"Summarized text: {summary_report}")
    print("===========")
    print(f"Summarized bulletpoints:\n{summary_bullets.summary}")
    print("===========")
    print(f"Consistency error: {err_consistency}, Faithfulness error:  {err_faith}, Total error: {error}")


    out = {"error":error, "err_faith":err_faith, "err_consistency":err_consistency,
           "bullets":bullet_list, "summary_report":summary_report, "input_text": input_text}
    return out


def process_case_B(input_list, date=None):
    # Convert list of strings to a paragraph
    input_text = ""

    # Really hacky stuff
    if len(input_text)<= 250:
        input_text = input_text + " " * (250 - len(input_text)) 

    for bullet_idx in range(len(input_list)):
        input_text = input_text + input_list[bullet_idx] + ". " # Adding period just for punctuation, can remove it later TODO

    # Returning the appended bulletpoints as a long string
    bullet_list = ""
    for i in range(len(input_list)):
       bullet_list = bullet_list + input_list[i] + "\n " 

    summary_report = co.summarize(
        text=input_text,
        length="long",
        format="paragraph",
        model="summarize-xlarge",
        extractiveness="low",
        temperature=0, # between 0 and 5, between 0 and 1 gives good results, high values cause more "creative" answers
        additional_command="Use formal and technical language, use passive tense and focus only on objective facts."
    )

    summary_report = summary_report.summary
    

    embed_init = co.embed(
        texts=[input_text], # API only accepts list of strings as input
        model='small',
        truncate="NONE"
    )


    embed_summary = co.embed(
        texts=[summary_report],
        model='small',
        truncate="NONE"
    )

    # The returned things are objects, we want them as numpy arrays
    embed_init = np.array(embed_init.embeddings[0]) 
    embed_summary = np.array(embed_summary.embeddings[0]) 
    # Compute final error
    err_consistency = 0.0
    err_faith = np.linalg.norm(embed_init - embed_summary)

    error = lambda_consistency * err_consistency + lambda_faith * err_faith

    print("===========")
    print(f"Original input: {input_text}")
    print("===========")
    print(f"Summarized text: {summary_report}")
    print("===========")
    print(f"Consistency error: {err_consistency}, Faithfulness error:  {err_faith}, Total error: {error}")

    out = {"error":error, "err_faith":err_faith, "err_consistency":err_consistency,
           "bullets":bullet_list, "summary_report":summary_report, "input_text": input_text}
    return out
