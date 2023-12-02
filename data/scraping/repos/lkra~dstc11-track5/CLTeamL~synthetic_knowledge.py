import argparse
import json
import os
import re

import openai
from dotenv import load_dotenv
from tqdm import tqdm

from approaches.prompting import chatgpt
from scripts.knowledge_reader import KnowledgeReader

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def main(args):
    augmented_knowledge = KnowledgeReader("../data/aug/")
    entities = augmented_knowledge.get_entity_list(args.domain)

    new_knowledge = {args.domain: {}}

    for entity in tqdm(entities):
        entity_id = entity["id"]
        entity_name = entity["name"]
        entity_obj = augmented_knowledge.knowledge[args.domain][str(entity_id)]
        faqs = {
            doc_id: {
                "question": doc_obj["title"],
                "answer": doc_obj["body"]
            }
            for doc_id, doc_obj in entity_obj["docs"].items()
        }

        new_entity = {
           entity_id : {
                "name": entity_name,
                "faqs": faqs,
           }
        }

        prompt_text = f"Given this example: {new_entity}, can you generate three more reviews, not more than 2 sentences, as: traveler type: review?"
        prompt = [{
            "role": "system",
            "content": prompt_text,
        }]

        output = chatgpt(prompt)
        response = output["text"]
        entity_reviews = list(filter(bool, response.splitlines()))

        reviews = {}
        for i, review in enumerate(entity_reviews):
            split_review = review.split(":")
            if len(split_review) == 1:
                continue
            traveler_type = split_review[0]
            traveler_type = re.sub(r"^\d+\.\s", "", traveler_type)

            traveler_review = {}
            for j, sentence in enumerate(split_review[1].split(".")):
                sentence = sentence.strip().replace('"', "")
                if sentence:
                    if sentence[-1] != ".":
                        sentence = f"{sentence}."
                    traveler_review[str(j)] = sentence

            reviews[str(i)] = {
                "traveler_type": traveler_type,
                "sentences": traveler_review,
            }

        new_knowledge[args.domain][str(entity_id)] = {
            "name": entity_name,
            "reviews": reviews,
            "faqs": faqs,
        }

    with open(args.output_file, "w") as f:
        json.dump(new_knowledge, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', default="taxi", type=str,
                        help="Choose one of the following domains for which reviews are needed: train, taxi, attraction")
    parser.add_argument('--output_file', type=str,
                        help="Path to where the new knowledge base should be saved.")
    args = parser.parse_args()
    config = vars(args)
    print("Parameters:")
    for k, v in config.items():
        print(f"  {k:>21} : {v}")
    main(args)
