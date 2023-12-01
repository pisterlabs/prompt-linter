import openai 
import gradio as gr
from contexts import message_history
 
openai.api_key = open("key.txt", "r").read().strip("\n")

def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0, max_tokens=500):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, 
        max_tokens=max_tokens, 
    )
    
    reply_content = response.choices[0].message["content"]
    
    # Check output for potentially harmful content
    response = openai.Moderation.create(
        input=reply_content
    )
    moderation_output = response["results"][0]
    
    # If the output is flagged, return a default response
    if moderation_output["flagged"]:
        return "I'm sorry, but I'm unable to generate a joke on that subject."
        
    return reply_content


def predict(input):
    # tokenize the new input sentence
    message_history.append({"role": "user", "content": f"{input}"})

    # reply from the assistant
    reply_content = get_completion_from_messages(message_history, model="gpt-3.5-turbo", temperature=0, max_tokens=500)
    
    # add the reply to the message history
    message_history.append({"role": "assistant", "content": f"{reply_content}"}) 
    
    # get pairs of msg["content"] from message history, skipping the pre-prompt
    response = [(message_history[i]["content"], message_history[i+1]["content"]) for i in range(4, len(message_history)-1, 2)]  # convert to tuples of list
    
    return response



if __name__=='__main__':
    
    # creates a new interface instance and assigns it to the variable demo.
    with gr.Blocks() as demo: 

        # creates a new Chatbot instance and assigns it to the variable chatbot.
        chatbot = gr.Chatbot() 

        # sets the title of the interface to "Joke Bot".
        with gr.Row(): 
            # creates a new Textbox component, which is used to collect user input. The show_label parameter is set to False to hide the label, and the placeholder parameter is set
            txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter").style(container=False)

        # sets the submit action of the Textbox to the predict function, which takes the input from the Textbox, the chatbot instance, and the state instance as arguments. 
        # This function processes the input and generates a response from the chatbot, which is displayed in the output area.
        txt.submit(predict, txt, chatbot) # submit(function, input, output)
        # txt.submit(lambda :"", None, txt)  
        # Sets submit action to lambda function that returns empty string 

        # sets the submit action of the Textbox to a JavaScript function that returns an empty string. The _js parameter is used to pass a JavaScript function to the submit method.
        # No function, no input to that function, submit action to textbox is a js function that returns empty string, so it clears immediately.
        txt.submit(None, None, txt, _js="() => {''}") 
            
    demo.launch()