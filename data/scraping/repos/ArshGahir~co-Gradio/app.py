import cohere as co
import gradio as gr
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("COHERE_API_KEY")
co = co.Client(api_key)

personality = f"""You are a helpful and intelligent personal assistant. \
                Your job is to follow instructions to the best of your abilities in order to answer an questions or \
                complete any tasks. Assume you have full autonomy, and solve any issues that arise in order to achieve \
                the end goal of a task. In this case, full autonomy means that you are able to modify given instructions \
                as you see fit, in order to achieve the end goal."""

def generate(prompt, model, truncate, return_likelihoods, num_generations, max_tokens,
             k, p, frequency_penalty, presence_penalty, temperature):
    response = co.generate(prompt = prompt, 
                           model = model, 
                           truncate = truncate, 
                           return_likelihoods = return_likelihoods, 
                           num_generations = num_generations, 
                           max_tokens= max_tokens,
                            k = k, 
                            p = p, 
                            frequency_penalty = frequency_penalty, 
                            presence_penalty = presence_penalty,
                            temperature = temperature)
    if return_likelihoods != 'NONE':
        return (response.generations)
    else:
        return(response.generations[0].text)

def summarize(text, length, format, model, extractiveness, temperature, additional_command):
    response = co.summarize(text = text, 
                            length = length,
                            format = format,
                            model = model,
                            extractiveness = extractiveness,
                            temperature = temperature,
                            additional_command = additional_command
                            )
    return(response.summary)

def embed(text, model):
    text = [item for sublist in text for item in sublist]
    response = co.embed(texts=text,
                        model=model)
    return(response.embeddings)

# Endpoint Parameters
## https://docs.cohere.com/reference/generate
## https://docs.cohere.com/reference/summarize-2


#co:mmon
model_choice = ["command", "command-nightly", "command-light", "command-light-nightly"]

#co.Generate
generate_temperature_choice = [x for x in range(0,5)]
truncate_choice = ['NONE', 'START', 'END']
return_likelihoods_choice = ['NONE', 'GENERATION', 'ALL']

#co.Summary
length_choice = ["short", "medium", "long", "auto"]
format_choice = ["paragraph", "bullets", "auto"]
extractiveness_choice = ["low", "medium", "high", "auto"]
summarize_temperature_choice = [0,1,2,3,4,5]
summary_examples = ["At the fall of Troy, Cassandra sought shelter in the temple of Athena. There she embraced the wooden statue of Athena in supplication for her protection, but was abducted and brutally raped by Ajax the Lesser. Cassandra clung so tightly to the statue of the goddess that Ajax knocked it from its stand as he dragged her away. The actions of Ajax were a sacrilege because Cassandra was a supplicant at the sanctuary, and under the protection of the goddess Athena and Ajax further defiled the temple by raping Cassandra. In Apollodorus chapter 6, section 6, Ajax's death comes at the hands of both Athena and Poseidon \"Athena threw a thunderbolt at the ship of Ajax; and when the ship went to pieces he made his way safe to a rock, and declared that he was saved in spite of the intention of Athena. But Poseidon smote the rock with his trident and split it, and Ajax fell into the sea and perished; and his body, being washed up, was buried by Thetis in Myconos\".",
            "Hi Cassandra I am a big fan of your blog. You share a lot of useful tips here. I especially like your post \“How to Eat Apples with Long Nails\”. It’s both well written and useful. I would like to contribute a unique post for your blog as well. I have read your guidelines and will follow them while writing the post. If you’re interested, I would love to work with you on the topics and formats that best meet your needs for the blog. Would you prefer sample topics, a draft outline, or a complete post?Thank you,Zuko"]

#co.Embed
embed_model_choice = ["embed-english-v2.0", "embed-english-light-v2.0", "embed-multilingual-v2.0"]

with gr.Blocks() as demo:
    gr.Markdown("Use any of Cohere's Endpoints with this interface")
    with gr.Tab("Generate"):
        gr.Interface(fn = generate, 
                        inputs = [gr.components.Textbox(label = "Prompt", lines = 5, placeholder = "Write/Paste Text Here"),
                                gr.components.Dropdown(label = "Model", choices = model_choice, value = 'command', info = "A large model may be more accurate, but generation would be slower. \nSmall Model Token Limit = 1024, Large Model Token Limit = 2048"),
                                gr.components.Dropdown(label = "Truncate", choices = truncate_choice, value = 'END', info = "Cuts off the input prompt as specified in case the input exceeds the maximum token length"),
                                #gr.components.Radio(label = "Stream", choices = ['true', 'false']),
                                #gr.components.Radio(label = "preset", choices = ['true', 'false']),
                                gr.components.Radio(label = "Return Likelihoods", choices = return_likelihoods_choice, value = 'NONE', info = "Displays the probability of each token that is generated"),
                                gr.Slider(label = "Number of Generations", minimum = 1, maximum = 5, value = 1, step = 1, info = "Generates an output N times, the outputs may be similar.\nLikelhoods of each generation can be compared, and the most appropriate one can be returne."),
                                gr.Slider(label = "Max Tokens", minimum = 0, maximum = 4096, value = 50, step = 0.1, info = "Generates N number of tokens. A token may not be a full word, Cohere uses byte-pair encoding for Tokenization.\nIf value is set too small, there is an increased probability that model will generate incomplete answers."),
                                #add enable/disable for Top K and Top P
                                gr.Slider(label = "Top K", minimum = 0, maximum = 500, value = 50, step = 1, info = "The response is generated by considering the top K most probable tokens.\nMay see a proportional increase in accuracy of the output, inverse to the speed."),
                                gr.Slider(label = "Top P", minimum = 0, maximum = 0.99, value = 0.50, step = 0.1, info = "More dynamic than Top K; the response is generated by considering only the top P most probable tokens whose probabilities add up to, or exceed, the set threshold."),
                                gr.Slider(label = "Frequency Penalty", minimum = 0, maximum = 1, value = 1, step = 1, info = "Penalizes tokens that have already appeared in the input prompt or the generated output. Tokens with a greater frequency are penalized more to reduce repition."),
                                gr.Slider(label = "Presence Penalty", minimum = 0, maximum = 1, value = 0.42, step = 0.1, info = "Penalizes all tokens that have appeared at least once in the input prompt or the generated output, regardless of frequency."),  
                                gr.Slider(label = "Temperature", minimum = generate_temperature_choice[0], maximum = generate_temperature_choice[-1],
                                            value = generate_temperature_choice[round(len(generate_temperature_choice)/2)], step = 1, info = "Impacts the \"creativity\" of the model, a temperature of 0.5 - 1 would generate a tame and measured response. 0 = Deterministic to 5 = Unhinged. \nExperiment Observations: A higher temperature may lead to more hallucinations in RAG-optimized applications")], 
                        outputs = [gr.Textbox(label = "Generated Text", lines = 3)],
                        title = "Text Generation with Cohere co.generate()",
                        description = "Prompt the model to generate new text or perform NLP tasks on provided text\n Type Instruction first and then Text")
    
    with gr.Tab("Summarize"):
        gr.Interface(fn = summarize, 
                        inputs = [gr.components.Textbox(label = "Text to Embed", lines = 5, placeholder = "Write/Paste Text Here"),
                                gr.components.Dropdown(label = "Summary Length", choices = length_choice),
                                gr.components.Dropdown(label = "Format", choices = format_choice),
                                gr.components.Dropdown(label = "Model", choices = model_choice),
                                gr.components.Dropdown(label = "Quoting", choices = extractiveness_choice),
                                gr.Slider(label = "Temperature", minimum = summarize_temperature_choice[0], maximum = summarize_temperature_choice[-1],
                                            value = summarize_temperature_choice[round(len(summarize_temperature_choice)/2)], step = 1)], 
                        outputs = [gr.Textbox(label = "Summary", lines = 3)],
                        title = "Text Summarization with Cohere co.summarize()",
                        description = "Enter any text (250 words minimum) and get a concise summary!"
                        )
        
    with gr.Tab("Embed"):
        gr.Interface(fn = embed, 
                     inputs = [gr.components.Dataframe(label = "Texts to Embed",headers = ["Input"], row_count=2, col_count=1, max_cols = 1, datatype = "str", type = "array"),
                               gr.components.Dropdown(label = "Model", choices = embed_model_choice)],
                     outputs = [gr.Textbox(label = "Embeddings", lines = 3)],
                     title = "Text Embeddings with Cohere co.embed()",
                     description = "Enter some text and generate the embeddings. Future Features: Visualize embeddings as clusters, and file upload"
                     )

demo.launch()