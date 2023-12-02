import openai
import ipywidgets as widgets
from IPython.display import display

openai.api_key = "Your API key"

def chatgpt_query(prompts):
    # example of generating a completion
    response = openai.Completion.create(
      engine="text-davinci-002",
      prompt=prompts,
      max_tokens=60
)
    return(response['choices'][0]['text'].strip())

input_widget = widgets.Textarea(placeholder="Enter your question or prompt here...")
output_widget = widgets.Label()
submit_button = widgets.Button(description="Submit")

def on_submit(_):
    prompt = input_widget.value
    response = chatgpt_query(prompt)
    output_widget.value = response

submit_button.on_click(on_submit)

display(input_widget)
display(submit_button)
display(output_widget)
