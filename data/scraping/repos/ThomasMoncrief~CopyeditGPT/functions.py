import openai
import global_var
import time


def run_editor(submit_text, chunk_count):
    edited_text = open("text_files/edited.txt", "w", encoding='utf-8', errors="ignore")
    run_count = 0
    while submit_text:
        adj_count = 0 #adjustment counter
        submit_chunk = "" #This will be sent to OpenAI
        if len(submit_text) > 4000:
            while submit_text[3999 + adj_count] != " ": #Makes sure to end on a space
                adj_count += 1
        submit_chunk += submit_text[:4000 + adj_count]
        submit_text = submit_text[4000 + adj_count:] #Writes over the chunk that was sent

        #**EDITOR SWITCH**
        #Activate top line for testing purpose. Activate second line to run the editor.
        # edited_text.write(submit_chunk)
        # time.sleep(1)
        edited_text.write(openai_api(submit_chunk))

        #Prints progress to terminal. Need to get something working for client side.
        run_count += 1
        print("Finished {:.0%}".format(run_count / chunk_count))

    edited_text.close()


def openai_api(original_text):
    
    openai.api_key = global_var.key #filled in by upload()
        
    prompt = "A professional copy editor has taken the text below and fixed mistakes. He followed the Chicago Manual of Style for writing numbers, capitalization, headers, and punctuation. He corrected any well-known factual mistakes involving statistics. He did not edit the voice or style of the prose. He formatted quotes as ASCII directional quotes.\n\n"
    prompt += original_text + "\n\nRewritten by the editor:\n"

    chatgpt_response = openai.Completion.create(
        model="text-davinci-003", 
        prompt=prompt, 
        temperature= 0.1, 
        max_tokens=2000, top_p=1, 
        frequency_penalty=0, 
        presence_penalty=0)['choices'][0]['text']
    return chatgpt_response
#testChange