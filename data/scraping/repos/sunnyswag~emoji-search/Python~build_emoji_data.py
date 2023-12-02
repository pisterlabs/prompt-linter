import gzip
import json
import os
from typing import Dict, List
from constants import *
import math

import emoji
import openai
import tqdm

openai.api_key = API_KEY
openai.api_base = OPENAI_URL

def get_embeddings(inps: List[str], batch: int=1000) -> List[List[float]]:
	i = 0
	outputs = []
	while i < len(inps):
		result = openai.Embedding.create(input=inps[i:i+batch], model=EMBEDDING_MODEL)
		outputs += [x["embedding"] for x in result['data']]
		i += batch
	assert len(outputs) == len(inps)
	return outputs

def write_to_json(filename: str, data: List[Dict]):
    with open(f"{filename}.gz", "wb") as fp:
        with gzip.GzipFile(fileobj=fp, mode="wb") as gz:
            for x in tqdm.tqdm(data):
                gz.write((json.dumps(x) + "\n").encode("utf-8"))
		
def getchunks(data: List[Dict], num_files: int = 5):
	chunk_size = math.ceil(len(data) / num_files)
	chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
	return chunks

def write_to_json_with_chunks(filename_base: str, chunks: List[List[Dict]]):
	for i, chunk in enumerate(chunks):
		filename = f"{filename_base}_{i}"
		write_to_json(filename, chunk)

def extract_emoji_messages() -> Dict[str, str]:
	with open(os.path.join(EMOJI_DATA_DIR, "emoji-data.txt"), encoding="utf-8") as file:
		data = file.readlines()
		data = [x.split("\t") for x in data if not x.startswith("#")]
		emojis_msg = {
			x[-1].split("(")[1].split(")")[0]: x[-1].split("(")[1].split(")")[1].lower().strip() 
			for x in data
		}

	# Combine two sources of emoji messages
	emojis_dict = emoji.EMOJI_DATA
	emojis_full_msg = {}
	for k in emojis_dict:
		msg = emojis_dict[k]['en'].strip(":").replace("_", " ").replace("-", " ").lower()
		other_msg = emojis_msg.get(k)
		if other_msg and len(other_msg) > len(msg):
			msg = other_msg
		emojis_full_msg[k] = msg

	# remove the duplicates values
	emojis_full_msg_dedupe = {v: k for k, v in emojis_full_msg.items()}
	emoji_dict_swap = {v: k for k, v in emojis_full_msg_dedupe.items()}
	return emoji_dict_swap

def main():
	# Query embeddings
	emoji_messages = extract_emoji_messages()
	descriptions = [f"The emoji {em} is about {msg}." for em, msg in emoji_messages.items()]
	print("first 5th descriptions of emoji: ", descriptions[:5], "\nlen descriptions: ", len(descriptions))

	embeddings = get_embeddings(descriptions)
	print("first embeddings of emoji: ", embeddings[0], "\nlen embeddings: ", len(embeddings))

	# Save embeddings
	info = [
		{"emoji": em, "message": msg, "embed": embed} 
		for (em, msg), embed in zip(emoji_messages.items(), embeddings)
	]
	print("first 2th info of emoji: ", info[:2], "\nlen info: ", len(info))

	json_output_filename = os.path.join(EMBEDDING_DATA_DIR, "emoji_embeddings_json")
	write_to_json(json_output_filename, info)
	write_to_json_with_chunks(json_output_filename, getchunks(info))

if __name__ == "__main__":
	main()