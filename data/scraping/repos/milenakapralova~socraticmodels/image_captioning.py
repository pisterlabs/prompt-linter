"""
This file contains the functionality related to the image captioning task.

"""
# Package loading
import sys
import os
import time
from typing import List, Union
import requests
import clip
import cv2
from PIL import Image
from dotenv import load_dotenv
from profanity_filter import ProfanityFilter
import matplotlib.pyplot as plt
import torch
import zipfile
import numpy as np
import openai
import pandas as pd
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer, Blip2Processor, Blip2ForConditionalGeneration, AutoProcessor,
    AutoModelForCausalLM, BlipProcessor, BlipForConditionalGeneration
)
sys.path.append('..')

# Local imports
from scripts.utils import print_time_dec, prepare_dir, set_all_seeds, get_device


class ImageCaptionerParent:
    """
    This is the parent class of the ImageCaptionerBaseline and ImageCaptionerImproved classes. It contains the
    functionality that is common to both child classes: the constructor of the class.
    """
    def __init__(self, random_seed=42, n_images=50, set_type='train'):
        """
        The constructor instantiates all the helper classes needed for the captioning. It sets the random seeds.
        Loads the vocabulary embeddings (these never change, so they are loaded from a cache). It loads a different set
        of images depending on the 'set_type' input. The embeddings of the images are derived. The cosine similarities
        between the text and image embeddings are determined. The remaining of the image captioning process is done
        by the children image captioning classes.

        Importantly, this method calculates all of the outputs that are independent of the hyperparameter choice for the
        image captioning process. This means this is only done once for the entire run during a hyperparameter tuning
        run.

        :param random_seed: The random seed that will be used in order to ensure deterministic results.
        :param n_images: The number of images that will be captioned.
        :param set_type: The data set type (train/valid/test).
        """

        """
        1. Set up
        """
        # Store set type
        self.set_type = set_type

        # Set the seeds
        set_all_seeds(random_seed)

        # ## Step 1: Downloading the MS COCO images and annotations
        self.coco_manager = CocoManager()

        # ### Set the device, instantiate managers and calculate the variables that are image independent.

        # Set the device to use
        device = get_device()

        # Instantiate the clip manager
        self.clip_manager = ClipManager(device)

        # Instantiate the image manager
        self.image_manager = ImageManager()

        # Instantiate the vocab manager
        self.vocab_manager = VocabManager()

        # Instantiate the language model manager
        self.lm_manager = LmManager()

        # Instantiate the GPT manager
        self.gpt_manager = GptManager()

        # Instantiate the prompt generator
        self.prompt_generator = LmPromptGenerator()

        # Set up the prompt generator map
        self.pg_map = {
            'original': self.prompt_generator.create_socratic_original_prompt,
            'creative': self.prompt_generator.create_improved_lm_creative,
            'gpt': self.prompt_generator.create_gpt_prompt_likely,
        }

        """
        2. Text embeddings
        """

        # Calculate the place features
        self.place_emb = CacheManager.get_place_emb(self.clip_manager, self.vocab_manager)

        # Calculate the object features
        self.object_emb = CacheManager.get_object_emb(self.clip_manager, self.vocab_manager)

        # Calculate the features of the number of people
        self.ppl_texts = None
        self.ppl_emb = None
        self.ppl_texts_bool = None
        self.ppl_emb_bool = None
        self.ppl_texts_mult = None
        self.ppl_emb_mult = None
        self.get_nb_of_people_emb()

        # Calculate the features for the image types
        self.img_types = ['photo', 'cartoon', 'sketch', 'painting']
        self.img_types_emb = self.clip_manager.get_text_emb([f'This is a {t}.' for t in self.img_types])

        # Create a dictionary that maps the objects to the cosine sim.
        self.object_embeddings = dict(zip(self.vocab_manager.object_list, self.object_emb))

        """
        3. Load images and compute image embedding
        """
        if self.set_type == 'demo':
            img_files = [
                self.image_manager.image_folder + d
                for d in self.image_manager.demo_names
            ]
        else:
            # Randomly select images from the COCO dataset
            img_files = self.coco_manager.get_random_image_paths(n_images=n_images, set_type=set_type)

        # Create dictionaries to store the images features
        self.img_dic = {}
        self.img_feat_dic = {}

        for img_file in img_files:
            # Load the image
            self.img_dic[img_file] = self.image_manager.load_image(img_file)
            # Generate the CLIP image embedding
            self.img_feat_dic[img_file] = self.clip_manager.get_img_emb(self.img_dic[img_file]).flatten()

        """
        4. Zero-shot VLM (CLIP): We zero-shot prompt CLIP to produce various inferences of an image, such as image type or 
        the number of people in the image.
        """

        # Classify image type
        # Create a dictionary to store the image types
        self.img_type_dic = {}
        for img_name, img_feat in self.img_feat_dic.items():
            sorted_img_types, img_type_scores = self.clip_manager.get_nn_text(
                self.img_types, self.img_types_emb, img_feat
            )
            self.img_type_dic[img_name] = sorted_img_types[0]

        # Classify number of people
        self.n_people_dic = None
        self.determine_nb_of_people()

        # Classify image place
        # Create a dictionary to store the location
        self.location_dic = {}
        for img_name, img_feat in self.img_feat_dic.items():
            sorted_places, places_scores = self.clip_manager.get_nn_text(
                self.vocab_manager.place_list, self.place_emb, img_feat
            )
            self.location_dic[img_name] = sorted_places

        # Classify image object
        # Create a dictionary to store the similarity of each object with the images
        self.object_score_map = {}
        self.sorted_obj_dic = {}
        for img_name, img_feat in self.img_feat_dic.items():
            sorted_obj_texts, obj_scores = self.clip_manager.get_nn_text(
                self.vocab_manager.object_list, self.object_emb, img_feat
            )
            self.object_score_map[img_name] = dict(zip(sorted_obj_texts, obj_scores))
            self.sorted_obj_dic[img_name] = sorted_obj_texts

    def get_nb_of_people_emb(self):
        """
        Gets the embeddings for the number of people.

        Method to be overriden in the child class.

        :return:
        """
        pass

    def determine_nb_of_people(self):
        """
        Determines the number of people in the image.

        Method to be overriden in the child class.

        :return:
        """
        pass

    def show_demo_image(self, img_name):
        """
        Creates a visualisation of the image using matplotlib.

        :param img_name: Input image to show
        :return:
        """
        # Show the image
        plt.imshow(self.img_dic[self.image_manager.image_folder + img_name])
        plt.show()


class ImageCaptionerBaseline(ImageCaptionerParent):
    @print_time_dec
    def main(
            self, n_captions=10, lm_temperature=0.9, lm_max_length=40, lm_do_sample=True,
            n_objects=10, n_places=3, caption_strategy='original'
    ):
        """
        The main method contains all the image captioning functionality that is dependent on the hyperparameters.

        It will generate a prompt given the input parameters, generate captions using the language models, then rank the
        generated captions by comparing their CLIP cosine similarity with the image and store the best one.

        :param n_captions: The number of captions to be generated by the model. Only the top one is stored.
        :param lm_temperature: The temperature of the language model.
        :param lm_max_length: The maximum length to be generated by the language model.
        :param lm_do_sample: Whether the language model should use sampling.
        :param n_objects: The number of objects to be included in the language model prompt.
        :param n_places: The number of places to be included in the language model prompt.
        :param caption_strategy: The captioning strategy to be used. Determines the prompt format give to the LM.
        :return:
        """

        # Set LM params
        model_params = {'temperature': lm_temperature, 'max_length': lm_max_length, 'do_sample': lm_do_sample}

        # Create dictionaries to store the outputs
        prompt_dic = {}
        sorted_caption_map = {}
        caption_score_map = {}

        # Loop through the image dictionary.
        for img_file in self.img_dic:
            # Generate a prompt
            prompt_dic[img_file] = self.pg_map[caption_strategy](
                self.img_type_dic[img_file], self.n_people_dic[img_file], self.location_dic[img_file][:n_places],
                self.sorted_obj_dic[img_file][:n_objects]
            )

            # Generate the caption using the language model
            caption_texts = self.generate_lm_response(n_captions * [prompt_dic[img_file]], model_params)

            # Zero-shot VLM: rank captions.
            caption_emb = self.clip_manager.get_text_emb(caption_texts)
            sorted_captions, caption_scores = self.clip_manager.get_nn_text(
                caption_texts, caption_emb, self.img_feat_dic[img_file]
            )
            # Store the caption outputs and score in memory.
            sorted_caption_map[img_file] = sorted_captions
            caption_score_map[img_file] = dict(zip(sorted_captions, caption_scores))

        """
        6. Outputs.
        """

        # Store the captions
        data_list = []
        for img_file in self.img_dic:
            generated_caption = sorted_caption_map[img_file][0]
            data_list.append({
                'image_name': img_file.split('/')[-1],
                'generated_caption': generated_caption,
                'cosine_similarity': caption_score_map[img_file][generated_caption]
            })

        file_path = self.get_output_file_name(lm_temperature, n_objects, n_places, caption_strategy)
        prepare_dir(file_path)
        self.generated_caption_df = pd.DataFrame(data_list)
        self.generated_caption_df.to_csv(file_path, index=False)

    def generate_lm_response(self, prompt_list, model_params):
        """
        Generates a response from the language model for a given prompt list and model parameters.

        :param prompt_list: The list of prompts to be passed to the language model.
        :param model_params: The language model parameters.
        :return: The response from the language model
        """
        return self.lm_manager.generate_response(prompt_list, model_params)

    def get_nb_of_people_emb(self):
        """
        This method is used to generate the embeddings related to the number of people. These will be compared to
        the CLIP embeddings of the images.

        :return:
        """
        # Classify number of people
        self.ppl_texts_bool = ['no people', 'people']
        self.ppl_emb_bool = self.clip_manager.get_text_emb([
            f'There are {p} in this photo.' for p in self.ppl_texts_bool
        ])
        self.ppl_texts_mult = ['is one person', 'are two people', 'are three people', 'are several people', 'are many people']
        self.ppl_emb_mult = self.clip_manager.get_text_emb([f'There {p} in this photo.' for p in self.ppl_texts_mult])

    def determine_nb_of_people(self):
        """
        This method evaluates the 'number of people' text embeddings against the image embeddings in the img_feat_dic
        dictionary. This determines the number of people in each images according to CLIP.

        :return:
        """
        # Create a dictionary to store the number of people
        self.n_people_dic = {}
        for img_name, img_feat in self.img_feat_dic.items():
            sorted_ppl_texts, ppl_scores = self.clip_manager.get_nn_text(
                self.ppl_texts_bool, self.ppl_emb_bool, img_feat
            )
            ppl_result = sorted_ppl_texts[0]
            if ppl_result == 'people':
                sorted_ppl_texts, ppl_scores = self.clip_manager.get_nn_text(
                    self.ppl_texts_mult, self.ppl_emb_mult, img_feat
                )
                ppl_result = sorted_ppl_texts[0]
            else:
                ppl_result = f'are {ppl_result}'

            self.n_people_dic[img_name] = ppl_result

    def random_parameter_search(
            self, n_iterations=100, n_captions=10, lm_max_length=40, lm_do_sample=True, lm_temp_min=0.5, lm_temp_max=1,
            n_objects_min=5, n_objects_max=15, n_places_min=1, n_places_max=6, caption_strategies=None
    ):
        """
        Runs a random parameter search.

        :param n_iterations: Number of random search iterations
        :param n_captions: Number of captions generated by the language model.
        :param lm_max_length: The max length the language model is allowed to generate.
        :param lm_do_sample: The argument that determines whether the language model is probabilistic or deterministic.
        :param lm_temp_min: The minimum temperature of the language model.
        :param lm_temp_max: The maximum temperature of the language model.
        :param n_objects_min: The minimum number of objects.
        :param n_objects_max: The maximum number of objects (the maximum will be n-1).
        :param n_places_min: The minimum number of places.
        :param n_places_max: The maximum number of places (the maximum will be n-1).
        :param caption_strategies: The caption strategies to test.
        :return:
        """
        if caption_strategies is None:
            caption_strategies = ['original', 'creative']
        for _ in range(n_iterations):
            template_params = {
                'n_captions': n_captions,
                'lm_temperature': np.round(np.random.uniform(lm_temp_min, lm_temp_max), 3),
                'lm_max_length': lm_max_length,
                'lm_do_sample': lm_do_sample,
                'n_objects': np.random.choice(range(n_objects_min, n_objects_max)),
                'n_places': np.random.choice(range(n_places_min, n_places_max)),
                'caption_strategy': np.random.choice(caption_strategies)
            }
            self.main(**template_params)

    def get_output_file_name(self, lm_temperature, n_objects, n_places, caption_strategy):
        """
        This method is used to create a file path name according to the input variables.

        This allows us to differentiate the runs under different parameters. This comes in particularly handy when
        performing the parameter search.

        :param lm_temperature: The temperature of the language model.
        :param n_objects: The number of objects that were included in the language model prompt.
        :param n_places: The number of places that were included in the language model prompt.
        :param caption_strategy: The captioning strategy. This decides on the prompt format.
        :return: A file name as a string.
        """
        extension = ''
        # The language model temperature
        extension += f'_temp_{lm_temperature}'.replace('.', '')
        # Number of objects
        extension += f'_nobj_{n_objects}'
        # Number of places
        extension += f'_npl_{n_places}'
        # Caption strategy
        extension += f'_strat_{caption_strategy}'
        # Train/test set
        extension += f'_{self.set_type}'
        return f'../data/outputs/captions/baseline_caption{extension}.csv'


class ImageCaptionerGpt(ImageCaptionerBaseline):
    def generate_lm_response(self, prompt_list, model_params):
        """
        Generates a response from the language model for a given prompt list and model parameters.

        :param prompt_list: The list of prompts to be passed to the language model.
        :param model_params: The language model parameters.
        :return: The response from the language model
        """
        return [
            self.gpt_manager.generate_response(
                prompt, temperature=model_params['temperature'], max_tokens=64, stop=None
            )
            for prompt in prompt_list
        ]

    def get_output_file_name(self, lm_temperature, n_objects, n_places, caption_strategy):
        """
        This method is used to create a file path name according to the input variables.

        This allows us to differentiate the runs under different parameters. This comes in particularly handy when
        performing the parameter search.

        :param lm_temperature: The temperature of the language model.
        :param n_objects: The number of objects that were included in the language model prompt.
        :param n_places: The number of places that were included in the language model prompt.
        :param caption_strategy: The captioning strategy. This decides on the prompt format.
        :return: A file name as a string.
        """
        extension = ''
        # The language model temperature
        extension += f'_temp_{lm_temperature}'.replace('.', '')
        # Number of objects
        extension += f'_nobj_{n_objects}'
        # Number of places
        extension += f'_npl_{n_places}'
        # Caption strategy
        extension += f'_strat_{caption_strategy}'
        # Train/test set
        extension += f'_{self.set_type}'
        return f'../data/outputs/captions/gpt_caption{extension}.csv'


class ImageCaptionerImproved(ImageCaptionerParent):
    @print_time_dec
    def main(
            self, n_captions=10, lm_temperature=0.9, lm_max_length=40, lm_do_sample=True,
            cos_sim_thres=0.7, n_objects=5, n_places=2, caption_strategy='original'
    ):
        """
        The main method contains all of the image captioning functionality that is dependent on the hyperparameters.

        It will generate a prompt given the input paramets, generate captions using the language models, then rank the
        generated captions by comparing their CLIP cosine similarity with the image and store the best one.

        :param n_captions: The number of captions to be generated by the model. Only the top one is stored.
        :param lm_temperature: The temperature of the language model.
        :param lm_max_length: The maximum length to be generated by the language model.
        :param lm_do_sample: Whether the language model should use sampling.
        :param cos_sim_thres: The cosine similarity threshold that will be use to filter out the object terms.
        :param n_objects: The number of objects to be included in the language model prompt.
        :param n_places: The number of places to be included in the language model prompt.
        :param caption_strategy: The captioning strategy to be used. Determines the prompt format give to the LM.
        :return:
        """

        """
        5. Finding both relevant and different objects using cosine similarity
        """
        best_matches = self.find_best_object_matches(cos_sim_thres)

        """
        6. Zero-shot LM (Flan-T5): We zero-shot prompt Flan-T5 to produce captions and use CLIP to rank the captions
        generated
        """

        # Set LM params
        model_params = {'temperature': lm_temperature, 'max_length': lm_max_length, 'do_sample': lm_do_sample}

        # Create dictionaries to store the outputs
        prompt_dic = {}
        sorted_caption_map = {}
        caption_score_map = {}

        for img_file in self.img_dic:
            prompt_dic[img_file] = self.pg_map[caption_strategy](
                self.img_type_dic[img_file], self.n_people_dic[img_file], self.location_dic[img_file][:n_places],
                object_list=best_matches[img_file][:n_objects]
            )

            # Generate the caption using the language model
            caption_texts = self.lm_manager.generate_response(n_captions * [prompt_dic[img_file]], model_params)

            # Zero-shot VLM: rank captions.
            caption_emb = self.clip_manager.get_text_emb(caption_texts)
            sorted_captions, caption_scores = self.clip_manager.get_nn_text(
                caption_texts, caption_emb, self.img_feat_dic[img_file]
            )
            sorted_caption_map[img_file] = sorted_captions
            caption_score_map[img_file] = dict(zip(sorted_captions, caption_scores))

        data_list = []
        for img_file in self.img_dic:
            generated_caption = sorted_caption_map[img_file][0]
            data_list.append({
                'image_name': img_file.split('/')[-1],
                'generated_caption': generated_caption,
                'cosine_similarity': caption_score_map[img_file][generated_caption],
                'set_type': self.set_type
            })
        file_path = self.get_output_file_name(lm_temperature, cos_sim_thres, n_objects, n_places, caption_strategy)
        prepare_dir(file_path)
        self.generated_caption_df = pd.DataFrame(data_list)
        if self.set_type != 'demo':
            self.generated_caption_df.to_csv(file_path, index=False)

    def get_output_file_name(self, lm_temperature, cos_sim_thres, n_objects, n_places, caption_strategy):
        """
        This method is used to create a file path name according to the input variables.

        This allows us to differentiate the runs under different parameters. This comes in particularly handy when
        performing the parameter search.

        :param lm_temperature: The temperature of the language model.
        :param cos_sim_thres: The cosine similarity threshold of the improved image captioning class.
        :param n_objects: The number of objects that were included in the language model prompt.
        :param n_places: The number of places that were included in the language model prompt.
        :param caption_strategy: The captioning strategy. This decides on the prompt format.
        :return: A file name as a string.
        """
        extension = ''
        # The language model temperature
        extension += f'_temp_{lm_temperature}'.replace('.', '')
        # The cosine thresold.
        extension += f'_costhres_{cos_sim_thres}'.replace('.', '')
        # Number of objects
        extension += f'_nobj_{n_objects}'
        # Number of places
        extension += f'_npl_{n_places}'
        # Caption strategy
        extension += f'_strat_{caption_strategy}'
        # Train/test set
        extension += f'_{self.set_type}'
        return f'../data/outputs/captions/improved_caption{extension}.csv'

    def find_best_object_matches(self, cos_sim_thres):
        """
        This method is integral to the ImageCaptionerImproved. It filters the objects to only returned
        terms that do not have too high of cosine similarity with each other. It is controled by the cos_sim_thres
        parameter.

        :param cos_sim_thres:
        :return:
        """
        # Create a dictionary to store the best object matches
        best_matches = {}

        for img_name, sorted_obj_texts in self.sorted_obj_dic.items():

            # Create a list that contains the objects ordered by cosine sim.
            embeddings_sorted = [self.object_embeddings[w] for w in sorted_obj_texts]

            # Create a list to store the best matches
            best_matches[img_name] = [sorted_obj_texts[0]]

            # Create an array to store the embeddings of the best matches
            unique_embeddings = embeddings_sorted[0].reshape(-1, 1)

            # Loop through the 100 best objects by cosine similarity
            for i in range(1, 100):
                # Obtain the maximum cosine similarity when comparing object i to the embeddings of the current best matches
                max_cos_sim = (unique_embeddings.T @ embeddings_sorted[i]).max()
                # If object i is different enough to the current best matches, add it to the best matches
                if max_cos_sim < cos_sim_thres:
                    unique_embeddings = np.concatenate([unique_embeddings, embeddings_sorted[i].reshape(-1, 1)], 1)
                    best_matches[img_name].append(sorted_obj_texts[i])
        return best_matches

    def get_nb_of_people_emb(self):
        """
        Determines the number of people in the image.

        :return:
        """
        self.ppl_texts = [
            'are no people', 'is one person', 'are two people', 'are three people', 'are several people',
            'are many people'
        ]
        self.ppl_emb = self.clip_manager.get_text_emb([f'There {p} in this photo.' for p in self.ppl_texts])

    def random_parameter_search(
            self, n_iterations=100, n_captions=10, lm_max_length=40, lm_do_sample=True, lm_temp_min=0.5, lm_temp_max=1,
            cos_sim_thres_min=0.6, cos_sim_thres_max=1, n_objects_min=5, n_objects_max=15, n_places_min=1,
            n_places_max=6, caption_strategies=None
    ):
        """
        Runs a random parameter search.

        :param n_iterations: Number of random search iterations
        :param n_captions: Number of captions generated by the language model.
        :param lm_max_length: The max length the language model is allowed to generate.
        :param lm_do_sample: The argument that determines whether the language model is probabilistic or deterministic.
        :param lm_temp_min: The minimum temperature of the language model.
        :param lm_temp_max: The maximum temperature of the language model.
        :param cos_sim_thres_min: The cosine similarity threshold minimum (specific to the improved pipeline).
        :param cos_sim_thres_max: The cosine similarity threshold maximum (specific to the improved pipeline).
        :param n_objects_min: The minimum number of objects.
        :param n_objects_max: The maximum number of objects (the maximum will be n-1).
        :param n_places_min: The minimum number of places.
        :param n_places_max: The maximum number of places (the maximum will be n-1).
        :param caption_strategies: The caption strategies to test.
        :return:
        """
        if caption_strategies is None:
            caption_strategies = ['original', 'creative']
        for _ in range(n_iterations):
            template_params = {
                'n_captions': n_captions,
                'lm_temperature': np.round(np.random.uniform(lm_temp_min, lm_temp_max), 3),
                'lm_max_length': lm_max_length,
                'lm_do_sample': lm_do_sample,
                'cos_sim_thres': np.round(np.random.uniform(cos_sim_thres_min, cos_sim_thres_max), 3),
                'n_objects': np.random.choice(range(n_objects_min, n_objects_max)),
                'n_places': np.random.choice(range(n_places_min, n_places_max)),
                'caption_strategy': np.random.choice(caption_strategies)
            }
            self.main(**template_params)


    def determine_nb_of_people(self):
        """
        Determines the number of people in the image.

        :return:
        """
        self.n_people_dic = {}
        for img_name, img_feat in self.img_feat_dic.items():
            sorted_ppl_texts, ppl_scores = self.clip_manager.get_nn_text(self.ppl_texts, self.ppl_emb, img_feat)
            self.n_people_dic[img_name] = sorted_ppl_texts[0]


class ImageCaptionerImprovedExtended(ImageCaptionerImproved):
    """
    This class extends ImageCaptionerImproved. It simply has a more extensive find_best_object_matches method.
    """
    def find_best_object_matches(self, cos_sim_thres):
        """
        This method is integral to the ImageCaptionerImproved. It filters the objects to only returned
        terms that do not have too high of cosine similarity with each other. It is controled by the cos_sim_thres
        parameter.

        :param cos_sim_thres:
        :return:
        """
        # Create a dictionary to store the best object matches
        best_matches = {}
        terms_to_include = {}

        for img_name, sorted_obj_texts in self.sorted_obj_dic.items():

            # Create a list that contains the objects ordered by cosine sim.
            embeddings_sorted = [self.object_embeddings[w] for w in sorted_obj_texts]

            # Create a list to store the best matches
            best_matches[img_name] = [sorted_obj_texts[0]]

            # Create an array to store the embeddings of the best matches
            unique_embeddings = embeddings_sorted[0].reshape(-1, 1)

            # Loop through the 100 best objects by cosine similarity
            for i in range(1, 100):
                # Obtain the maximum cosine similarity when comparing object i to the embeddings of the current best matches
                max_cos_sim = (unique_embeddings.T @ embeddings_sorted[i]).max()
                # If object i is different enough to the current best matches, add it to the best matches
                if max_cos_sim < cos_sim_thres:
                    unique_embeddings = np.concatenate([unique_embeddings, embeddings_sorted[i].reshape(-1, 1)], 1)
                    best_matches[img_name].append(sorted_obj_texts[i])

            # Looping through the best matches, consider each terms separately by splitting the commas and spaces.
            data_list = []
            for terms in best_matches[img_name]:
                for term_split in terms.split(', '):
                    score = self.clip_manager.get_image_caption_score(term_split, self.img_feat_dic[img_name])
                    data_list.append({
                        'term': term_split, 'score': score, 'context': terms
                    })
                    term_split_split = term_split.split(' ')
                    if len(term_split_split) > 1:
                        for term_split2 in term_split_split:
                            score = self.clip_manager.get_image_caption_score(term_split2, self.img_feat_dic[img_name])
                            data_list.append({
                                'term': term_split2, 'score': score, 'context': terms
                            })

            # Create a dataframe with the terms and scores and only keep the top term per context.
            term_df = pd.DataFrame(data_list).sort_values('score', ascending=False).drop_duplicates('context').reset_index(drop=True)

            # Prepare loop to find if additional terms can improve cosine similarity
            best_terms_sorted = term_df['term'].tolist()
            best_term = best_terms_sorted[0]
            terms_to_check = list(set(best_terms_sorted[1:]))
            best_cos_sim = term_df['score'].iloc[0]
            terms_to_include[img_name] = [best_term]

            # Perform a loop to find if additional terms can improve the cosine similarity
            n_iteration = 5
            for iteration in range(n_iteration):
                data_list = []
                for term_to_test in terms_to_check:
                    new_term = f"{best_term} {term_to_test}"
                    score = self.clip_manager.get_image_caption_score(new_term, self.img_feat_dic[img_name])
                    data_list.append({
                        'term': new_term, 'candidate': term_to_test, 'score': score
                    })
                combined_df = pd.DataFrame(data_list).sort_values('score', ascending=False)
                if combined_df['score'].iloc[0] > best_cos_sim:
                    best_cos_sim = combined_df['score'].iloc[0]
                    terms_to_include[img_name].append(combined_df['candidate'].iloc[0])
                    terms_to_check = combined_df['candidate'].tolist()[1:]
                    best_term += f" {combined_df['candidate'].iloc[0]}"
                else:
                    break

        return terms_to_include


class CocoManager:
    def __init__(self):
        """
        The CocoManager handles all the functionality relating to the COCO dataset.

        It downloads the data if needed and manages the zip files. It also ensures that there is no data leakage between
        the train/valid/test sets.
        """
        self.image_dir = '../data/coco/val2017/'
        self.dataset_to_download = {
            '../data/coco/val2017': 'http://images.cocodataset.org/zips/val2017.zip',
            '../data/coco/annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
        }
        self.download_data()

    def download_unzip_delete(self, folder, url):
        """
        Checks if the COCO data is there, otherwise it downloads and unzips the data.

        :param folder:
        :param url:
        :return:
        """
        if not os.path.exists(folder):
            prepare_dir(folder)
            response = requests.get(url)
            parent = '/'.join(folder.split('/')[:-1])
            with open(parent + '/zip.zip', "wb") as f:
                f.write(response.content)
            with zipfile.ZipFile(parent + '/zip.zip', "r") as zip_ref:
                zip_ref.extractall(parent)
            os.remove(parent + '/zip.zip')

    def download_data(self):
        """
        Downloads the images and annotations of the COCO dataset of interest if the file does not exist.
        """
        for folder, url in self.dataset_to_download.items():
            self.download_unzip_delete(folder, url)

    def get_random_image_paths(self, n_images, set_type):
        """
        This method randomly and deterministically determines the images to use for the different set types.

        The process is deterministic because a numpy random seed is set. The images are then randomly selected in
        sequence. The images are not replaced in the selection process, such that the train, valid and test sets will
        all have different images.

        :param n_images:
        :param set_type:
        :return:
        """
        img_list = os.listdir(self.image_dir)
        img_list.sort()
        # Train set
        train_set = np.random.choice(img_list, size=n_images).tolist()
        remaining_images = list(set(img_list) - set(train_set))
        remaining_images.sort()

        # Valid set
        valid_set = np.random.choice(remaining_images, size=n_images).tolist()
        remaining_images = list(set(remaining_images) - set(valid_set))
        remaining_images.sort()

        # Test set
        test_set = np.random.choice(remaining_images, size=n_images).tolist()

        # Return the image path list
        if set_type == 'train':
            return [f'{self.image_dir}{c}' for c in train_set]
        elif set_type == 'valid':
            return [f'{self.image_dir}{c}' for c in valid_set]
        elif set_type == 'test':
            return [f'{self.image_dir}{c}' for c in test_set]
        elif set_type == 'demo':
            return None
        else:
            raise ValueError(f'set_type {test_set} not supported.')


class ImageManager:
    def __init__(self):
        """
        The ImageManager class manages the functionality related to the demo images.

        It downloads the demo images if needed and has the functionality to load the images.
        """
        self.image_folder = '../data/images/example_images/'
        self.images_to_download = {
            'demo_img.png': 'https://github.com/rmokady/CLIP_prefix_caption/raw/main/Images/COCO_val2014_000000165547.jpg',
            'monkey_with_gun.jpg': 'https://drive.google.com/uc?export=download&id=1iG0TJTZ0yRJEC8dA-WwS7X-GhuX8sfy8',
            'astronaut_with_beer.jpg': 'https://drive.google.com/uc?export=download&id=1p5RwifMFtl1CLlUXaIR_y60_laDTbNMi',
            'fruit_bowl.jpg': 'https://drive.google.com/uc?export=download&id=1gRYMoTfCwuV4tNy14Qf2Q_hebx05GNd9',
            'cute_bear.jpg': 'https://drive.google.com/uc?export=download&id=1WvgweWH_vSODLv2EOoXqGaHDcUKPDHbh',
            'wedding_image.jpg': 'https://drive.google.com/uc?export=download&id=1Apn_e2sBXOV-Nx7KSAsEsWLpAJ1kQw1g'
        }
        self.demo_names = list(self.images_to_download)
        self.download_data()

    def download_data(self):
        """
        Downloads the images of self.images_to_download if the file does not exist.

        :return:
        """
        # Download images
        for img_path, img_url in self.images_to_download.items():
            if not os.path.exists(self.image_folder + img_path):
                self.download_image_from_url(img_path, img_url)

    def download_image_from_url(self, img_path: str, img_url: str):
        """
        Downloads an image from an url.

        :param img_path: Output path.
        :param img_url: Download url.
        :return:
        """
        file_path = self.image_folder + img_path
        prepare_dir(file_path)
        with open(self.image_folder + img_path, 'wb') as f:
            f.write(requests.get(img_url).content)

    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """
        Loads an image in RGB from an image path.

        :param image_path:
        :return:
        """
        return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    def show_image(self, image):
        """
        Creates a visualisation of the image using matplotlib.

        :param image: Input image to show.
        :return:
        """
        # Show the image
        plt.imshow(image)
        plt.show()


class VocabManager:
    def __init__(self):
        """
        The VocabManager class handles the functionality related to the vocabularies. It handles the preprocessing of
        the vocabulary, including the filtering using the ProfanityFilter.

        'categories_places365.txt' was the vocabulary used in the original Socratic Models paper.
        'dictionary_and_semantic_hierarchy.txt' was tested on, but it did not provide an improvement, so it was not
        used in the proposed image captioning solution.
        """
        self.vocab_folder = '../data/vocabulary/'
        self.cache_folder = '../data/cache/'
        self.files_to_download = {
            'categories_places365.txt': "https://raw.githubusercontent.com/zhoubolei/places_devkit/master/categories_pl"
                                        "aces365.txt",
            'dictionary_and_semantic_hierarchy.txt': "https://raw.githubusercontent.com/Tencent/tencent-ml-images/maste"
                                                     "r/data/dictionary_and_semantic_hierarchy.txt"
        }
        self.download_data()
        self.place_list = self.load_places()
        self.object_list = self.load_objects(remove_profanity=False)

    def download_data(self):
        """
        Download the vocabularies.

        :return:
        """
        # Download the vocabularies
        for file_name, url in self.files_to_download.items():
            file_path = self.vocab_folder + file_name
            if not os.path.exists(file_path):
                self.download_vocab_from_url(file_path, url)

    @staticmethod
    def download_vocab_from_url(file_path, url):
        """
        Downloads a file for a given url and stores it in the file_path.

        :param file_path: Output file
        :param url: Download url
        :return:
        """
        prepare_dir(file_path)
        response = requests.get(url)
        with open(file_path, "wb") as f:
            f.write(response.content)

    @print_time_dec
    def load_places(self) -> List[str]:
        """
        Load the places.

        This function comes from the original Socratic Models repository. A cache was added to speed up execution.

        :return:
        """
        file_path = self.vocab_folder + 'categories_places365.txt'
        cache_path = self.cache_folder + 'place_texts.txt'
        # Ensure the cache folder exists
        prepare_dir(cache_path)
        if not os.path.exists(cache_path):
            # Load the raw places file
            place_categories = np.loadtxt(file_path, dtype=str)
            place_texts = []
            for place in place_categories[:, 0]:
                place = place.split('/')[2:]
                if len(place) > 1:
                    place = place[1] + ' ' + place[0]
                else:
                    place = place[0]
                place = place.replace('_', ' ')
                place_texts.append(place)
            # Cache the file for the next run
            with open(cache_path, 'w') as f:
                for place in place_texts:
                    f.write(f"{place}\n")
        else:
            # Read the cache file
            with open(cache_path) as f:
                place_texts = f.read().splitlines()
        place_texts.sort()
        return place_texts

    @print_time_dec
    def load_objects(self, remove_profanity: bool = False) -> List[str]:
        """
        Load the objects.

        This function comes from the original Socratic Models repository. A cache was added to speed up execution.

        :return:
        """
        file_path = self.vocab_folder + 'dictionary_and_semantic_hierarchy.txt'
        cache_path = self.cache_folder + 'object_texts.txt'
        # Ensure the cache folder exists
        prepare_dir(cache_path)
        if not os.path.exists(cache_path):
            # Load the raw object file
            with open(file_path) as fid:
                object_categories = fid.readlines()
            object_texts = []
            pf = ProfanityFilter()
            for object_text in object_categories[1:]:
                object_text = object_text.strip()
                object_text = object_text.split('\t')[3]
                if remove_profanity:
                    safe_list = ''
                    for variant in object_text.split(','):
                        text = variant.strip()
                        if pf.is_clean(text):
                            safe_list += f'{text}, '

                    safe_list = safe_list[:-2]
                    if len(safe_list) > 0:
                        object_texts.append(safe_list)
                else:
                    object_texts.append(object_text)
            # Cache the file for the next run
            with open(cache_path, 'w') as f:
                for obj in object_texts:
                    f.write(f"{obj}\n")
        else:
            # Read the cache file
            with open(cache_path) as f:
                object_texts = f.read().splitlines()
        object_texts = [o for o in list(set(object_texts)) if o not in self.place_list]
        object_texts.sort()
        return object_texts


class ClipManager:
    def __init__(self, device: str, version: str = "ViT-L/14"):
        """
        The ClipManager handles all the methods relating to the CLIP model.

        :param device: The device to use ('cuda', 'mps', 'cpu').
        :param version: The CLIP model version.
        """
        self.device = device
        self.feat_dim_map = {
            'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'RN50x16': 768, 'RN50x64': 1024, 'ViT-B/32': 512,
            'ViT-B/16': 512,'ViT-L/14': 768
        }
        self.version = version
        self.feat_dim = self.feat_dim_map[version]
        self.model, self.preprocess = clip.load(version)
        self.model.to(self.device)
        self.model.eval()

    def get_text_emb(self, in_text: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Creates a numpy array of text features with the columns containing the features and the rows containing the
        representations for each of the strings in the input in_text list.

        :param in_text: List of prompts
        :param batch_size: The batch size
        :return: Array with n_features columns and len(in_text) rows
        """
        text_tokens = clip.tokenize(in_text).to(self.device)
        text_id = 0
        text_emb = np.zeros((len(in_text), self.feat_dim), dtype=np.float32)
        while text_id < len(text_tokens):  # Batched inference.
            batch_size = min(len(in_text) - text_id, batch_size)
            text_batch = text_tokens[text_id:text_id + batch_size]
            with torch.no_grad():
                batch_emb = self.model.encode_text(text_batch).float()
            batch_emb /= batch_emb.norm(dim=-1, keepdim=True)
            batch_emb = np.float32(batch_emb.cpu())
            text_emb[text_id:text_id + batch_size, :] = batch_emb
            text_id += batch_size
        return text_emb

    def get_img_emb(self, img):
        """
        For the given imput image, this method returns a CLIP embedding.

        :param img: Input image.
        :return: CLIP embedding.
        """
        img_pil = Image.fromarray(np.uint8(img))
        img_in = self.preprocess(img_pil)[None, ...]
        with torch.no_grad():
            img_emb = self.model.encode_image(img_in.to(self.device)).float()
        img_emb /= img_emb.norm(dim=-1, keepdim=True)
        img_emb = np.float32(img_emb.cpu())
        return img_emb

    @staticmethod
    def get_nn_text(raw_texts, text_emb, img_emb):
        """
        This method ranks the cosine similarities of the input text embeddings with the input image embedding.

        :param raw_texts: The input texts that will be ranked.
        :param text_emb: The respective embeddings corresponding to each raw text.
        :param img_emb: The image embedding the text will be compared to.
        :return: A tuple containing the ordered texts and their cosine similarity score with the image.
        """
        scores = text_emb @ img_emb.T
        scores = scores.squeeze()
        high_to_low_ids = np.argsort(scores).squeeze()[::-1]
        high_to_low_texts = [raw_texts[i] for i in high_to_low_ids]
        high_to_low_scores = np.sort(scores).squeeze()[::-1]
        return high_to_low_texts, high_to_low_scores

    def get_image_caption_score(self, caption, img_emb):
        """
        For a given input caption and image embedding, determines the cosine similarity.

        :param caption: Input caption to be assessed.
        :param img_emb: Input image embedding the input caption is being tested against.
        :return:
        """
        text_emb = self.get_text_emb([caption])
        return float(text_emb @ img_emb.T)

    def get_img_info(self, img, place_feats, obj_feats, vocab_manager, obj_topk=10):
        """
        This method retrieves all of the information for a given image using CLIP's cosine similarity metric.

        :param img: The input image.
        :param place_feats: The place features.
        :param obj_feats: The objects features.
        :param vocab_manager: The vocabulary manager.
        :param obj_topk: The number of images to include.
        :return: A tuple containing all the different information.
        """
        # get image features
        img_feats = self.get_img_emb(img)
        # classify image type
        img_types = ['photo', 'cartoon', 'sketch', 'painting']
        img_types_feats = self.get_text_emb([f'This is a {t}.' for t in img_types])
        sorted_img_types, img_type_scores = self.get_nn_text(img_types, img_types_feats, img_feats)
        img_type = sorted_img_types[0]
        print(f'This is a {img_type}.')

        # classify number of people
        ppl_texts = [
            'are no people', 'is one person', 'are two people', 'are three people', 'are several people', 'are many people'
        ]
        ppl_feats = self.get_text_emb([f'There {p} in this photo.' for p in ppl_texts])
        sorted_ppl_texts, ppl_scores = self.get_nn_text(ppl_texts, ppl_feats, img_feats)
        n_people = sorted_ppl_texts[0]
        print(f'There {n_people} in this photo.')

        # classify places
        sorted_places, places_scores = self.get_nn_text(vocab_manager.place_list, place_feats, img_feats)
        location = sorted_places[0]
        print(f'It was taken in {location}.')

        # classify objects
        sorted_obj_texts, obj_scores = self.get_nn_text(vocab_manager.object_list, obj_feats, img_feats)
        object_list = ''
        for i in range(obj_topk):
            object_list += f'{sorted_obj_texts[i]}, '
        object_list = object_list[:-2]
        print(f'Top 10 objects in the image: \n{sorted_obj_texts[:10]}')

        return img_type, n_people, location, sorted_obj_texts, object_list, obj_scores

    def rank_gen_outputs(self, img, output_texts, k=5):
        """
        This method is used to print out ranked top k captions for a given image using CLIP's cosine similarity with
        the image.

        :param img: The input image.
        :param output_texts: The output text to be ranked.
        :param k: The number of texts to be printed out.
        :return:
        """
        img_emb = self.get_img_emb(img)
        output_feats = self.get_text_emb(output_texts)
        sorted_outputs, output_scores = self.get_nn_text(output_texts, output_feats, img_emb)
        output_score_map = dict(zip(sorted_outputs, output_scores))
        for i, output in enumerate(sorted_outputs[:k]):
            print(f'{i + 1}. {output} ({output_score_map[output]:.2f})')


class CacheManager:
    """
    The CacheManager is used to store and load caches.

    This is to speed up development, as it can takes significant time to generate the embeddings. As the embedding
    generation is deterministic, it is unnecessary to generate them each time a new script is run.

    """
    @staticmethod
    def get_place_emb(clip_manager, vocab_manager):
        """
        This method retrieves the embeddings related to the places. If the embedding cache has not yet been created, it
        generates the embeddings using the CLIP manager, then caches them and returns them.

        :param clip_manager: The CLIP helper manager instance.
        :param vocab_manager: The vocabulary helper manager instance.
        :return: The place embeddings.
        """
        place_emb_path = '../data/cache/place_emb.npy'
        if not os.path.exists(place_emb_path):
            # Ensure the directory exists
            prepare_dir(place_emb_path)
            # Calculate the place features
            place_emb = clip_manager.get_text_emb([f'Photo of a {p}.' for p in vocab_manager.place_list])
            np.save(place_emb_path, place_emb)
        else:
            # Load cache
            place_emb = np.load(place_emb_path)
        return place_emb

    @staticmethod
    def get_object_emb(clip_manager, vocab_manager):
        """
        This method retrieves the embeddings related to the objects. If the embedding cache has not yet been created, it
        generates the embeddings using the CLIP manager, then caches them and returns them.

        :param clip_manager: The CLIP helper manager instance.
        :param vocab_manager: The vocabulary helper manager instance.
        :return: The place embeddings.
        """
        object_emb_path = '../data/cache/object_emb.npy'
        if not os.path.exists(object_emb_path):
            # Ensure the directory exists
            prepare_dir(object_emb_path)
            # Calculate the place features
            object_emb = clip_manager.get_text_emb([f'Photo of a {p}.' for p in vocab_manager.object_list])
            np.save(object_emb_path, object_emb)
        else:
            # Load cache
            object_emb = np.load(object_emb_path)
        return object_emb


class LmManager:
    def __init__(self, version="google/flan-t5-xl", use_api=False):
        """
        The LmManager class handles the functionality of the language model.

        It offers the flexibility of using either the API or a locally cached and run model.

        :param version: The language model version to use.
        :param use_api: Where to use the API or a locally run model.
        """
        self.model = None
        self.tokenizer = None
        self.api_url = None
        self.headers = None
        self.use_api = use_api
        if use_api:
            load_dotenv()
            if 'HUGGINGFACE_API' in os.environ:
                hf_api = os.environ['HUGGINGFACE_API']
            else:
                raise ValueError(
                    "You need to store your huggingface api key in your environment under "
                    "'HUGGINGFACE_API' if you want to use the API. Otherwise, set 'use_api' to False."
                )
            self.api_url = f"https://api-inference.huggingface.co/models/{version}"
            self.headers = {"Authorization": f"Bearer {hf_api}"}
        else:
            # Instantiate the model
            self.model = AutoModelForSeq2SeqLM.from_pretrained(version)
            # Instantiate the tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(version)

    @print_time_dec
    def generate_response(
            self, prompt: Union[List[str], str], model_params: Union[dict, None] = None
    ) -> Union[List[str], str]:
        """
        This method generates a response from the language model. It has the flexibility to receive either a single
        prompt as a string or multiple prompts as a list of strings.

        :param prompt: Either a single prompt as a string or multiple prompts as a list of prompts.
        :param model_params: The language model parameters.
        :return: The response from the language model.
        """
        if self.use_api:
            if isinstance(prompt, str):
                return self.generate_response_api(prompt, model_params)
            else:
                return [self.generate_response_api(p, model_params) for p in prompt]
        else:
            return self.generate_response_local(prompt, model_params)

    def generate_response_local(
            self, prompt: Union[List[str], str], model_params: Union[dict, None] = None
    ) -> Union[List[str], str]:
        """
        Generates a response using a local model. Accepts a single prompt or a list of prompts.

        :param prompt: Prompt(s) as list or str
        :param model_params: Model parameters
        :return: str if 1 else list
        """
        if model_params is None:
            model_params = {}
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, **model_params)
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        if len(decoded) == 1:
            return decoded[0]
        return decoded

    def generate_response_api(
            self, prompt: Union[List[str], str], model_params: Union[dict, None] = None
    ) -> Union[List[str], str]:
        """
        Generate a response through the API. Accepts a single prompt or a list of prompts.

        :param prompt: Prompt(s) as list or str
        :param model_params: Model parameters
        :return: str if 1 else list
        """
        if model_params is None:
            model_params = {}
        outputs = self.query({
            "inputs": prompt,
            "parameters": model_params,
            "options": {"use_cache": False, "wait_for_model": True}
        })
        decoded = [output['generated_text'] for output in outputs][0]
        return decoded

    def query(self, payload):
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        return response.json()


class GptManager:
    def __init__(self, version="text-davinci-002"):
        """
           The GPT manager handles the functionality related to the GPT API. By default, 'text-davinci-002' is used, the
           same as the original Socratic Models paper.

           :param version: The engine version to query.
       """
        self.version = version

        if 'OPENAI_API_KEY' in os.environ:
            openai.api_key = os.environ['OPENAI_API_KEY']

    def generate_response(
            self, prompt, max_tokens=64, temperature=0, stop=None, n=1
    ):
        """
        Makes an API call to generate a response from an open AI model.

        :param prompt: The prompt passed to the model.
        :param max_tokens: The maximum token desired in the response.
        :param temperature: The temperature of the language model to use.
        :param stop: Whether to perform early stopping or not.
        :param n: How many completions to generate for each prompt.
        :return:
        """
        response = openai.Completion.create(
            engine=self.version, prompt=prompt, max_tokens=max_tokens, temperature=temperature, stop=stop, n=n
        )
        return response["choices"][0]["text"].strip()

    def get_response_gpt(self, prompt, model='gpt-3.5-turbo', temperature=1., max_tokens=100, n=1, **kwargs):
        """
        Get response by prompting GPT-3

        :param model: prompt to GPT-3
        :param temperature: GPT-3 model
        :param max_tokens: temperature for sampling
        :param n: How many completions to generate for each prompt.
        :param kwargs: maximum number of tokens to generate
        :return: generated response from GPT-3
        """
        try:
            response = openai.ChatCompletion.create(
                model=model, temperature=temperature, max_tokens=max_tokens, n=n, messages=[
                    {"role": "user", "content": prompt}
                ], **kwargs
            )
        except openai.error.RateLimitError:
            # sleep if API rate limit exceeded
            print('API rate limit exceeded,sleeping for 120s...')
            time.sleep(120)
            response = openai.ChatCompletion.create(
                model=model, temperature=temperature, max_tokens=max_tokens, n=n, messages=[
                    {"role": "user", "content": prompt}
                ], **kwargs
            )

        output = response['choices'][0]['message']['content']
        return output


class BlipManager:
    def __init__(self, device, version="Salesforce/blip-image-captioning-base"):
        """
        This helper class makes it easy to generate responses from BLIP.

        :param device: The device to use.
        :param version: The BLIP model version to use.
        """
        self.processor = BlipProcessor.from_pretrained(version)
        self.model = BlipForConditionalGeneration.from_pretrained(version, torch_dtype=torch.float16)
        self.device = device

    def generate_response(self, image, prompt=None, model_params=None):
        """
        Generate a response passing an image and an optional prompt.

        :param image: Input image.
        :param prompt: The prompt to pass to BLIP.
        :param model_params:
        :return:
        """
        if model_params is None:
            model_params = {}
        if prompt is None:
            inputs = self.processor(image, return_tensors="pt").to(self.device, torch.float16)
        else:
            inputs = self.processor(image, prompt, return_tensors="pt").to(self.device, torch.float16)
        self.model.to(self.device)
        out = self.model.generate(**inputs, **model_params)
        return self.processor.decode(out[0], skip_special_tokens=True).strip()


class Blip2Manager:
    def __init__(self, device, version="Salesforce/blip2-opt-2.7b"):
        """
        This helper class makes it easy to generate responses from BLIP2.

        :param device: The device to use.
        :param version: The BLIP2 model version to use.
        """
        self.processor = Blip2Processor.from_pretrained(version)
        self.model = Blip2ForConditionalGeneration.from_pretrained(version, torch_dtype=torch.float16)
        self.device = device

    def generate_response(self, image, prompt=None, model_params=None):
        """
        Generate a response passing an image and an optional prompt.

        :param image: Input image.
        :param prompt: The prompt to pass to BLIP.
        :param model_params:
        :return:
        """
        if model_params is None:
            model_params = {}
        if prompt is None:
            inputs = self.processor(image, return_tensors="pt").to(self.device, torch.float16)
        else:
            inputs = self.processor(image, prompt, return_tensors="pt").to(self.device, torch.float16)
        self.model.to(self.device)
        out = self.model.generate(**inputs, **model_params)
        return self.processor.decode(out[0], skip_special_tokens=True).strip()


class GitVisionManager:
    def __init__(self, device, version="microsoft/git-base-coco"):
        """
        This helper class makes it easy to generate responses from GIT.

        :param device: The device to use.
        :param version: The GIT model version to use.
        """
        self.processor = AutoProcessor.from_pretrained(version)
        self.model = AutoModelForCausalLM.from_pretrained(version)
        self.device = device

    def generate_response(self, image, max_length=50):
        """
        Returns a caption for the given image.

        :param image: The image to caption.
        :param max_length: Max length of the caption.
        :return:
        """
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values=pixel_values, max_length=max_length)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


class LmPromptGenerator:
    """
    The LmPromptGenerator makes it easy to try out new prompt designs. It contains all the prompts that were tested.
    """
    @staticmethod
    def create_baseline_lm_prompt(img_type, ppl_result, sorted_places, object_list_str):
        """
        This method is a replica of the prompt from the original Socratic Models paper.

        :param img_type: The image type inferred by CLIP. A string.
        :param ppl_result: How many people were detected in the image. Inferred by CLIP. A string.
        :param sorted_places: The most likely places. Inferred by CLIP. A string.
        :param object_list_str: A comma separated set of objects. A string.
        :return:
        """
        return f'''I am an intelligent image captioning bot.
        This image is a {img_type}. There {ppl_result}.
        I think this photo was taken at a {sorted_places[0]}, {sorted_places[1]}, or {sorted_places[2]}.
        I think there might be a {object_list_str} in this {img_type}.
        A creative short caption I can generate to describe this image is:'''

    def create_socratic_original_prompt(self, img_type, ppl_result, sorted_places, object_list):
        """
        This method generates the same prompt from the original Socratic Models paper. However, it makes it easier to
        change the number of places/objects passed and still create a well formatted prompt.

        :param img_type: The image type inferred by CLIP. A string.
        :param ppl_result: How many people were detected in the image. Inferred by CLIP. A string.
        :param sorted_places: The most likely places. Inferred by CLIP. A list of strings.
        :param object_list_str: A list of the most likely objects inferred by CLIP.
        :return: A prompt string to be passed to the language model of choice.
        """
        places_string = self.get_places_string(sorted_places)
        return f'''I am an intelligent image captioning bot.
        This image is a {img_type}. There {ppl_result}.
        I think this photo was taken at a {places_string}.
        I think there might be a {', '.join(object_list)} in this {img_type}.
        A creative short caption I can generate to describe this image is:'''

    def create_baseline_lm_prompt_likely(self, img_type, ppl_result, sorted_places, object_list):
        """
        This method generates a prompt alternative that was reported to work well in some contexts according to the
        original Socratic Models paper.

        :param img_type: The image type inferred by CLIP. A string.
        :param ppl_result: How many people were detected in the image. Inferred by CLIP. A string.
        :param sorted_places: The most likely places. Inferred by CLIP. A list of strings.
        :param object_list_str: A list of the most likely objects inferred by CLIP.
        :return: A prompt string to be passed to the language model of choice.
        """
        places_string = self.get_places_string(sorted_places)
        return f'''I am an intelligent image captioning bot.
        This image is a {img_type}. There {ppl_result}.
        I think this photo was taken at a {places_string}.
        I think there might be a {', '.join(object_list)} in this {img_type}.
        A short, likely caption I can generate to describe this image is:'''

    def create_gpt_prompt_likely(self, img_type, ppl_result, sorted_places, object_list_str):
        """
        This method generates a prompt alternative that was reported to work well in some contexts according to the
        original Socratic Models paper. This is the prompt that was used in the image captioning notebook of the
        original paper.

        :param img_type: The image type inferred by CLIP. A string.
        :param ppl_result: How many people were detected in the image. Inferred by CLIP. A string.
        :param sorted_places: The most likely places. Inferred by CLIP. A string.
        :param object_list_str: A list of the most likely objects inferred by CLIP.
        :return: A prompt string to be passed to the language model of choice.
        """
        places_string = self.get_places_string(sorted_places)
        return f'''I am an intelligent image captioning bot.
        This image is a {img_type}. There {ppl_result}.
        I think this photo was taken at a {places_string}.
        I think there might be a {object_list_str} in this {img_type}.
        A short, likely caption I can generate to describe this image is:'''

    @staticmethod
    def create_improved_lm_prompt(img_type, ppl_result, terms_to_include):
        """
        One of the prompts that was experimented on.
        This was a prompt that performed well on some demo images. However, it did not offer the best performance during
        the COCO dataset benchmark. It does not include the location information.

        :param img_type: The image type inferred by CLIP. A string.
        :param ppl_result: How many people were detected in the image. Inferred by CLIP. A string.
        :param terms_to_include: The list of objects/terms to use.
        :return: A prompt string to be passed to the language model of choice.
        """
        return f'''Create a creative beautiful caption from this context:
        "This image is a {img_type}. There {ppl_result}.
        The context is: {', '.join(terms_to_include)}.
        A creative short caption I can generate to describe this image is:'''

    def create_improved_lm_creative(self, img_type, ppl_result, sorted_places, object_list):
        """
        One of the prompts that was experimented on.
        This was a prompt that performed well on some demo images. However, it did not offer the best performance during
        the COCO dataset benchmark. It does not include the location information.

        :param img_type: The image type inferred by CLIP. A string.
        :param ppl_result: How many people were detected in the image. Inferred by CLIP. A string.
        :param terms_to_include: The list of objects/terms to use.
        :return: A prompt string to be passed to the language model of choice.
        """
        places_string = self.get_places_string(sorted_places)
        return f'''I am a poetic writer that creates image captions.
        This image is a {img_type}. There {ppl_result}.
        This photo may have been taken at a {places_string}.
        There might be a {', '.join(object_list)} in this {img_type}.
        A creative short caption I can generate to describe this image is:'''

    def create_cot_prompt(self, sample, sorted_places, sorted_obj_texts, obj_topk=10):
        prompt = (
            f"This image was taken in a {sorted_places[0]}. It contains a {', '.join(sorted_obj_texts[:obj_topk])}.\n"
            f"Question: {sample['question']}\nChoices: {sample['choices']}\nHint: {sample['hint']}\n"
            f"Answer: Let's think step by step..."
        )
        return prompt

    def create_vqa_prompt(self, sample, sorted_places, sorted_objs, obj_topk=10):
        prompt = (
            f"This image was taken in a {sorted_places[0]}. It contains a {', '.join(sorted_objs[:obj_topk])}. "
            f"Using this information, answer the following question: {sample['question']}\nHint: {sample['hint']}\n"
            f"Select the index of the correct choice: "
            f"{[f'{i} {choice}' for i, choice in enumerate(sample['choices'])]}."
            f"Your answer should be a single integer (no text) and you must choose exactly one of the options.\nAnswer:"
        )
        return prompt

    @staticmethod
    def get_places_string(place_list):
        """
        This method takes a list of places and returns a string.

        :param place_list: The list of places to combine in a string.
        :return:
        """
        if len(place_list) == 1:
            return place_list[0]
        elif len(place_list) == 2:
            return f'{place_list[0]} or {place_list[1]}'
        else:
            return f'{", ".join(place_list[:-1])} or {place_list[-1]}'
