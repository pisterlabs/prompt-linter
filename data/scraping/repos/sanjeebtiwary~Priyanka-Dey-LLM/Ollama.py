import langchain
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import gradio as gr


# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="meta-llama/Llama-2-70b-hf")

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b-hf")

# Define a function to generate a one-word summary
def generate_one_word_summary(review):
    input_ids = tokenizer.encode(review, return_tensors="pt", max_length=1024, truncation=True)
    output = model.generate(input_ids, max_length=1, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    generated_summary = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_summary

# Create a Gradio interface
iface = gr.Interface(
    fn=generate_one_word_summary,
    inputs="text",
    outputs="text",
    title="Restaurant Review One-Word Summary",
    description="Generate a one-word summary for a restaurant review.",
)

# Launch the interface
iface.launch()


