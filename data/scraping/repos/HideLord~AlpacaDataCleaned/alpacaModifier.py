import gradio as gr
import openai

OAI_PROMPT = "You are a helpful assistant. You answer in a concise and accurate manner. Your responses are short and to the point."

class AlpacaModifier:
	def __init__(self):
		self.input = ''
		self.instruction = ''
		self.old_output = ''
		self.modified_output = ''
		
	def next_callback(self, instruction='', input='', old_output='', modified_output=''):
		# returns the next instruction_text, input_text, old_output_text, modified_output_text.
		pass
		
	def save_callback(self, instruction='', input='', old_output='', modified_output=''):
		# When this is called, all the changes done until this moment will be saved.
		pass
	
	def reset_callback(self, instruction='', input='', old_output='', modified_output=''):
		# Reset to the begining of the file.
		pass

	def skip_ahead(self, steps, instruction='', input='', old_output='', modified_output=''):
		while steps > 1:
			steps -= 1
			instruction, input, old_output, modified_output = self.next_callback(instruction, input, old_output, old_output)
		if steps == 1:
			return self.next_callback(instruction, input, old_output, old_output)
		return instruction, input, old_output, modified_output
		
	def ask_gpt(self, instruction='', input='', old_output='', modified_output='', key=''):
		openai.api_key = key

		composite_content = f"{instruction}\n\n{input}" if input else instruction
		print(f'Sending:\n{composite_content}')

		completion = openai.ChatCompletion.create(
			model="gpt-3.5-turbo",
			messages=[
					{"role": "system", "content": OAI_PROMPT},
					{"role": "user", "content": composite_content}
				]
		)
		modified_output = completion["choices"][0]["message"]["content"]
		return instruction, input, old_output, modified_output

	def modify_output(self):
		# Automatically modify the output in some way or just return it as it is.
		pass

	def run(self):
		with gr.Blocks() as demo:
			with gr.Column():
				gr.Markdown("""
                ## 🦙 Alpaca Dataset Editor
                Cleaned Dataset: [Github](https://github.com/gururise/AlpacaDataCleaned) - [Hugging Face](https://huggingface.co/datasets/yahma/alpaca-cleaned)
                
                *To use GPT to generate answers, OpenAI API key is required*
                """)
				instruction_text = gr.Textbox(lines=2, label="Instruction", value=self.instruction, interactive=True)
				input_text = gr.Textbox(lines=1, label="Input", value=self.input, interactive=True)
				old_output_text = gr.Textbox(lines=2, label="Old Output", value=self.old_output, interactive=False)
				modified_output_text = gr.Textbox(lines=10, label="Modified Output", value=self.modified_output, interactive=True)
			
			with gr.Row():
				button_next = gr.Button(value="Next")
				button_next.click(self.next_callback, 
					inputs=[instruction_text, input_text, old_output_text, modified_output_text], 
					outputs=[instruction_text, input_text, old_output_text, modified_output_text])
				button_save = gr.Button(value="Save")
				button_save.click(self.save_callback,
					inputs=[instruction_text, input_text, old_output_text, modified_output_text])
				button_reset = gr.Button(value="Reset To Begining")
				button_reset.click(self.reset_callback,
					inputs=[instruction_text, input_text, old_output_text, modified_output_text], 
					outputs=[instruction_text, input_text, old_output_text, modified_output_text])

			with gr.Row():
				skip_ahead = gr.Number(label="Items to skip", value=0, interactive=True)
				button_skip = gr.Button(value="Skip Ahead")
				button_skip.click(self.skip_ahead,
					inputs=[skip_ahead, instruction_text, input_text, old_output_text, modified_output_text], 
					outputs=[instruction_text, input_text, old_output_text, modified_output_text])

			with gr.Row():
				gpt_api_key = gr.Textbox(label="API key", placeholder="Enter your OpenAI API Key (optional)")
				button_ask_gpt = gr.Button(value="Ask GPT")
				button_ask_gpt.click(self.ask_gpt,
					inputs=[instruction_text, input_text, old_output_text, modified_output_text, gpt_api_key], 
					outputs=[instruction_text, input_text, old_output_text, modified_output_text])

		demo.launch()