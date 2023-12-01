import gradio as gr
import openai
import os
import jsonlines
import time
import nltk
import argparse
import tiktoken
import json

nltk.download('punkt')

def count_tokens(text):
    """Count the number of tokens in a text string using tiktoken."""
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def generate_model_response(api_key, user_input, num_responses):
    """Generate the model response from the user input."""
    while True:
        try:
            return openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                messages=[
                    {"role": "user", "content": f"Generate {num_responses} training data pairs in JSONL format as your response. The 'input' should be a distinctive user query related to '{user_input}', and the 'output' should be your unique, brief response. Ensure your response strictly aligns with the viewpoint, style, and tone of the original author, as if the author is directly answering the user's query. Each 'input' and 'output' pair should be unique and directly address the query without any additional context or information."}
                ],
                max_tokens=1000,
                api_key=api_key,
            )
        except openai.error.RateLimitError:
            print("Rate limit hit, waiting and retrying...")
            time.sleep(10)

def process_text(api_key, user_input, total_generations):
    """Process the text and generate the JSONL file."""
    openai.api_key = api_key
    responses = []
    total_tokens = count_tokens(user_input)
    print(f"Estimated total tokens in input: {total_tokens}")
    with jsonlines.open('chat.jsonl', mode='w') as writer:
        while len(responses) < total_generations:
            try:
                model_response = generate_model_response(api_key, user_input, total_generations)
                response = model_response['choices'][0]['message']['content']
                response_pairs = response.split("\n")
                for pair in response_pairs:
                    # Clean up the pair
                    pair = pair.strip()
                    if len(responses) >= total_generations:
                        break
                    pair.replace('\\', "")
                    json_pair = json.loads(pair)
                    writer.write(json_pair)
                    responses.append(pair)
            except openai.error.RateLimitError:
                print("Rate limit hit, waiting and retrying...")
                time.sleep(10)
                continue
            except openai.error.ServiceUnavailableError:
                print("Service unavailable, waiting and retrying...")
                time.sleep(10)
                continue
            except Exception as e:
                print(f"An error occurred: {e}")
                continue
    return '\n'.join(responses)

def define_gradio_interface():
    """Define the Gradio interface for the chat."""
    return gr.Interface(
        fn=process_text,
        inputs=[
            gr.inputs.Textbox(label="OpenAI API Key", type="password"),
            gr.components.Textbox(label="User Input", lines=20),
            gr.inputs.Slider(minimum=1, maximum=1000, step=1, default=10, label="Total Generations"),
        ],
        outputs=gr.components.Textbox(label="Chat History"),
    )

def main():
    """Main function to launch the Gradio interface."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="generate a publicly shareable link")
    args = parser.parse_args()

    define_gradio_interface().launch(share=args.share)

if __name__ == "__main__":
    main()
