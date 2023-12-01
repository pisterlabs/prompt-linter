import re
import os
from tqdm import tqdm
import openai
import json

from loguru import logger

def remove_code_snippets(text):
    return re.sub(r'```.*?```', '', text, flags=re.DOTALL)


def remove_tables(text):
    def replace_table(match):
        header = re.findall(r'\|([^|\n]+)\|', match.group(0))
        values = re.findall(r'\|([^|\n]+)\|', match.group(0)[match.end(1):])
        return "\n".join(header + values)

    return re.sub(r'(\|.*\|\n\|:?-+:?\|.*\n)((\|.*\|(\n)?)+)', replace_table, text, flags=re.MULTILINE)


def remove_urls(text):
    return re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)


def parse_readme(readme):
    main_text = re.sub(r"---\n.+?\n---", "", readme, flags=re.DOTALL).strip()
    main_text = remove_code_snippets(main_text)
    main_text = remove_tables(main_text)
    main_text = remove_urls(main_text)
    return main_text

def read_readme_files(base_dir):
    readmes = {}
    for author_name in tqdm(os.listdir(base_dir)):
        for model_name in os.listdir(os.path.join(base_dir, author_name)):
            readme_path = os.path.join(base_dir, author_name, model_name, "README.md")
            # logger.debug(readme_path)
            if os.path.exists(readme_path):
                with open(readme_path, "r") as f:
                    readmes[model_name] = f.read()
            logger.info(f"author_name, model_name, readme_path are {author_name}, {model_name}, {readme_path}")
            break
        break
    return readmes


def extract_metadata_gpt35(prompt):
    MODEL = "gpt-3.5-turbo"
    response = openai.ChatCompletion.create(
        model = MODEL,
        messages=[
        {"role": "system", "content": "You are classifying the category and task of pre-trained model based on the given README files."},
        {"role": "user", "content": "Can you clasify the following readme file into one of the element in the dict?\n \
        Use this format, dont't generate any other texts 'task: category'" + prompt},
        ],
        temperature = 0.1,    
        max_tokens = 200,
    )
    logger.info(response)
    message = response["choices"][0]["message"]["content"].strip().lower().replace("\n", " ")

    return response.choices[0].text.strip()

def truncate_text(text, max_tokens=200):
    tokens = text.split()
    truncated_tokens = tokens[:max_tokens]
    truncated_text = ' '.join(truncated_tokens)
    return truncated_text

def extract_metadata(metadata, readme):
    readme = parse_readme(readme)
    truncated_documentation = truncate_text(readme)
    # prompt = f"Extract pre-trained deep learning model metadata for the following model documentation: \n{readme}"
    prompt = f"Here is a mapping of {{task: category}}: mapping = {{\n"
    for task in [
        'feature-extraction: multimodal',
        'text-to-image:multimodal',
        'image-to-text:multimodal',
        'text-to-video:multimodal',
        'visual-question-answering:multimodal',
        'graph-machine-learning:multimodal',
        'graph-ml:multimodal',
        'depth-estimation:computer-vision',
        'image-classification:computer-vision',
        'object-detection:computer-vision',
        'image-segmentation:computer-vision',
        'image-to-image:computer-vision',
        'unconditional-image-generation:computer-vision',
        'video-classification:computer-vision',
        'zero-shot-image-classification:computer-vision',
        'text-classification:natural-language-processing',
        'token-classification:natural-language-processing',
        'table-question-answering:natural-language-processing',
        'question-answering:natural-language-processing',
        'zero-shot-classification:natural-language-processing',
        'translation:natural-language-processing',
        'summarization:natural-language-processing',
        'conversational:natural-language-processing',
        'text-generation:natural-language-processing',
        'text2text-generation:natural-language-processing',
        'fill-mask:natural-language-processing',
        'sentence-similarity:natural-language-processing',
        'table-to-text:natural-language-processing', # dataset task
        'multiple-choice:natural-language-processing',
        'text-retrieval:natural-language-processing', # dataset task
        'document-question-answering:natural-language-processing', # model task
        'text-to-speech:audio',
        'automatic-speech-recognition:audio',
        'audio-to-audio:audio',
        'audio-classification:audio',
        'voice-activity-detection:audio',
        'tabular-classification:tabular',
        'tabular-regression:tabular',
        'tabular-to-text:tabular', # dataset task
        'time-series-forecasting:tabular', # dataset task
        'reinforcement-learning:reinforcement-learning', # model task
        'robotics:reinforcement-learning', # model task
        'null: None',
        'other: None',
        ]:
            prompt = f"{prompt} '{task}'\n"
    logger.debug(prompt)

    # prompt = f"{prompt}}}\n\n"
    prompt = f"{prompt}\n{truncated_documentation}"

    # logger.debug(prompt)
    # response = openai.Completion.create(
    #     engine="davinci",
    #     prompt=prompt,
    #     max_tokens=100,
    #     n=1,
    #     stop=None,
    #     temperature=0.5,
    # )

    prompt = f""
    metadata_text = extract_metadata_gpt35(prompt)
    logger.debug(metadata_text)
    metadata[model_name] = metadata_text

    with open("metadata.txt", "w") as file:
        for entry in metadata:
            for key, value in entry.items():
                file.write(f"{key}: {value}\n")
            file.write("\n")
    return metadata

if __name__ == "__main__":
    base_dir = "/scratch/gilbreth/jiang784/PTMTorrent/huggingface/PTM-Torrent/ptm_torrent/huggingface/data/huggingface/repos"
    metadata_dir = "/scratch/gilbreth/jiang784/PTMTorrent/huggingface/PTM-Torrent/ptm_torrent/huggingface/data/huggingface/metadata"
    metadata_file = metadata_dir + "/huggingface.json"

    readmes = read_readme_files(base_dir)
    # logger.debug(readmes.keys())

    # extract metadata from readme using GPT APIs
    openai.api_key = "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"
    
    metadata = {}
    for model_name, readme in readmes.items():
        extract_metadata(metadata, readme)
        logger.info(f"Extracted metadata for {model_name}")
        # Assuming 'metadata' variable contains the extracted metadata
        break
    logger.debug(metadata)

