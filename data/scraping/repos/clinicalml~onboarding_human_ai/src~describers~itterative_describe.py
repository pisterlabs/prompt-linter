import logging
import os
import time

import numpy as np
import openai
import torch
from PIL import Image
from sklearn.metrics.pairwise import pairwise_distances
from transformers import CLIPModel, CLIPProcessor


def get_text_embeddings(text):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    # create random sample Image
    img = Image.new("RGB", (224, 224))
    images = [np.array(img)]
    inputs = processor(text=text, images=images, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeds = outputs.text_embeds.numpy()
    return embeds[0]


class IterativeRegionDescribe:
    def __init__(
        self,
        descriptions,
        embeddings,
        cluster_scores,
        cluster_labels,
        open_ai_key,
        get_text_embedding_fn,
        n_rounds=5,
        initial_positive_set_size=15,
        initial_negative_set_size=5,
        chat_correct=False,
    ):
        self.descriptions = descriptions
        self.embeddings = embeddings
        self.cluster_scores = cluster_scores
        self.cluster_labels = cluster_labels
        self.points_to_keep = 200
        self.chat_correct = chat_correct
        self.pre_instruction = (
            "I will provide you with a set of descriptions of points that belong to a region and a set of descriptions of point that do not belong to the region."
            + "Your task is to summarize the points inside the region in a concise and precise short sentence while making sure the summary contrast to points outside the region."
            + "Your one sentence summary should be able to allow a person to distinguish between points inside and outside the region while describing the region well."
            + "The summary should be no more than 20 words, it should be accurate, concise, distinguishing and precise."
            + "Example: \n"
            + "inside the region: \n two cows and two sheep grazing in a pasture. \n the sheep is standing near a tree. \n outside the region:  the cows are lying on the grass beside the water.\n"
            + "summary: The region consists of descriptions that have have sheep in them outside in nature, it could have cows but must have sheep. \n End of Example \n"
        )
        self.post_instruction = "summary:"
        self.initial_positive_set_size = initial_positive_set_size
        self.initial_negative_set_size = initial_negative_set_size
        self.n_rounds = n_rounds
        self.get_text_embeddings = get_text_embedding_fn
        openai.api_key = open_ai_key

    def get_completion(self, prompt, history=[]):
        while True:
            try:
                if len(history) == 0:
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ]
                else:
                    messages = history

                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo", messages=messages
                )
                logging.info("Called OPENAI API")
                return response["choices"][0]["message"]["content"]
            except:
                print("pausing openai api")
                time.sleep(0.3)
                continue

    def get_prompt(self, positives, negatives):
        prompt = self.pre_instruction + "\n"
        prompt += "inside the region: \n "
        counter = 1
        for p in positives:
            prompt += p[0] + ", \n "
            counter += 1
        if len(negatives) > 0:
            prompt += ". \n not in the region: \n"
            counter = 1
            for p in negatives:
                prompt += p[0] + ",\n"
                counter += 1
        prompt += self.post_instruction
        return prompt

    def describe_region(self, cluster_selected):
        inside_caption_set_idx = []
        outside_caption_set_idx = []

        scores_to_cluster_selected = self.cluster_scores[:, cluster_selected]
        # sort with the highest score first and keep indices
        sorted_indices_to_cluster = np.argsort(scores_to_cluster_selected)[::-1]
        # walk through sorted_indices, if the cluster is not the selected cluster, add it to the outside set
        for i in range(len(sorted_indices_to_cluster)):
            if self.cluster_labels[sorted_indices_to_cluster[i]] != cluster_selected:
                if len(outside_caption_set_idx) < self.points_to_keep:
                    outside_caption_set_idx.append(sorted_indices_to_cluster[i])
            else:
                if len(inside_caption_set_idx) < self.points_to_keep:
                    inside_caption_set_idx.append(sorted_indices_to_cluster[i])
            if (
                len(outside_caption_set_idx) >= self.points_to_keep
                and len(inside_caption_set_idx) >= self.points_to_keep
            ):
                break

        # INITIALIZATION
        # add the top 5 high scored captions to the positive set

        positives_selected = []
        negatives_selected = []
        descriptions_over_time = []

        for i in range(self.initial_positive_set_size):
            positives_selected.append(
                (
                    self.descriptions[inside_caption_set_idx[i]],
                    inside_caption_set_idx[i],
                )
            )

        for i in range(self.initial_negative_set_size):
            negatives_selected.append(
                (
                    self.descriptions[outside_caption_set_idx[i]],
                    outside_caption_set_idx[i],
                )
            )
        # get description
        prompt = self.get_prompt(positives_selected, negatives_selected)
        description = self.get_completion(prompt)
        # get description embedding
        logging.info(f"initial description: {description}")
        description_emb = self.get_text_embeddings(description)
        descriptions_over_time.append(description)
        if self.n_rounds == 0:
            return descriptions_over_time, positives_selected, negatives_selected

        history_prompts = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        history_prompts.append({"role": "user", "content": prompt})
        for i in range(self.n_rounds):
            history_prompts.append({"role": "assistant", "content": description})

            # in positive set find max distance image
            distances_inside = pairwise_distances(
                description_emb.reshape(1, -1), self.embeddings[inside_caption_set_idx]
            )
            sorted_indices = np.argsort(distances_inside)
            already_added = False
            index_to_add = 0
            while not already_added:
                index_to_add -= 1
                # check if inside_caption_set_idx[sorted_indices[0][index_to_add]] is in positives_selected
                # need to check in second tuple element
                if inside_caption_set_idx[sorted_indices[0][index_to_add]] not in [
                    x[1] for x in positives_selected
                ]:
                    positives_selected.append(
                        (
                            self.descriptions[
                                inside_caption_set_idx[sorted_indices[0][index_to_add]]
                            ],
                            inside_caption_set_idx[sorted_indices[0][index_to_add]],
                        )
                    )
                    already_added = True

            # in negative find best distance
            distances_outside = pairwise_distances(
                description_emb.reshape(1, -1), self.embeddings[outside_caption_set_idx]
            )
            sorted_indices = np.argsort(distances_outside)
            already_added = False
            index_to_add = -1
            while not already_added:
                index_to_add += 1
                if outside_caption_set_idx[sorted_indices[0][index_to_add]] not in [
                    x[1] for x in negatives_selected
                ]:
                    negatives_selected.append(
                        (
                            self.descriptions[
                                outside_caption_set_idx[sorted_indices[0][index_to_add]]
                            ],
                            outside_caption_set_idx[sorted_indices[0][index_to_add]],
                        )
                    )
                    already_added = True
            # print positives_selected added
            logging.info(f"positive added {positives_selected[-1][0]}")
            logging.info(f"negative added {negatives_selected[-1][0]}")
            new_prompt = (
                "The following points were found to not match the desription you just provided:"
                + description
                + ", so consider updating the description. \n inside the region:"
                + positives_selected[-1][0]
                + "\n not in the region: "
                + negatives_selected[-1][0]
                + "\n summary:"
            )
            history_prompts.append({"role": "user", "content": new_prompt})
            if self.chat_correct:
                description = self.get_completion(
                    self.get_prompt(positives_selected, negatives_selected),
                    history=history_prompts,
                )
            else:
                description = self.get_completion(
                    self.get_prompt(positives_selected, negatives_selected)
                )
            # get description embedding
            description_emb = self.get_text_embeddings(description)
            descriptions_over_time.append(description)
            # print description
            logging.info(f"new description: {description}")
        return descriptions_over_time, positives_selected, negatives_selected
