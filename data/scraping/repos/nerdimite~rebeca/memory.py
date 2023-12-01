import os
import glob
import json
import pickle
from tqdm.auto import tqdm

import cv2
import numpy as np
import torch
import faiss

from openai_vpt.lib.policy import MinecraftPolicy
from openai_vpt.lib.tree_util import tree_map

from model import VPTEncoder
from action_utils import ActionProcessor


def generate_situation_weights(time_array=np.linspace(0, 5, 20), A=1, tau=0.2, C=0):
    """Generate a weight array for situations based on an exponential decay function"""
    weights = A * np.exp(-time_array / tau) + C # exp decay
    weights = np.array(weights / np.sum(weights))
    return weights[::-1]


class SituationLoader:
    """Data loader for loading expert demonstrations and creating situation embeddings"""

    def __init__(self, vpt_model: VPTEncoder, data_dir="data/MakeWaterfall/"):
        self.vpt = vpt_model
        self.action_processor = ActionProcessor()
        self.load_expert_data(data_dir)

    def load_expert_data(self, data_dir):
        """Load expert demonstrations from data_dir"""

        unique_ids = glob.glob(os.path.join(data_dir, "*.mp4"))
        unique_ids = list(set([os.path.basename(x).split(".")[0] for x in unique_ids]))
        unique_ids.sort()

        self.demonstration_tuples = []
        for unique_id in unique_ids:
            video_path = os.path.abspath(os.path.join(data_dir, unique_id + ".mp4"))
            json_path = os.path.abspath(os.path.join(data_dir, unique_id + ".jsonl"))
            self.demonstration_tuples.append((unique_id, video_path, json_path))

    def load_demonstrations(self, num_demos=None):
        """Load expert demonstrations from demonstration tuples"""

        if num_demos is not None:
            _demonstration_tuples = self.demonstration_tuples[:num_demos]
        else:
            _demonstration_tuples = self.demonstration_tuples

        demonstrations = []
        for unique_id, video_path, json_path in tqdm(
            _demonstration_tuples, desc="Loading expert demonstrations"
        ):
            video = self._load_video(video_path)
            jsonl = self._load_jsonl(json_path)
            demonstrations.append(
                {"demo_id": unique_id, "video": video, "jsonl": jsonl}
            )
        return demonstrations

    def encode_demonstrations(self, demonstrations):
        encoded_demos = []
        for demo in tqdm(demonstrations, desc="Encoding expert demonstrations"):
            encoded_demo = self.vpt.encode_trajectory(demo["video"])
            encoded_demos.append(
                {"demo_id": demo["demo_id"], "encoded_demo": encoded_demo, "actions": demo["jsonl"]}
            )
        return encoded_demos

    def load_encode_save_demos(self, num_demos=None, save_dir="data/MakeWaterfallEncoded/"):
        '''Load, encode and save expert demonstrations to disk'''
        
        # Select the number of demonstrations to load
        _demonstration_tuples = self.demonstration_tuples[:num_demos]

        # Create a directory to save the encoded demonstrations
        os.makedirs(save_dir, exist_ok=True)
        
        for unique_id, video_path, json_path in tqdm(
            _demonstration_tuples, desc="Loading expert demonstrations"
        ):
            video = self._load_video(video_path)
            jsonl = self._load_jsonl(json_path)
            
            # Encode the demonstration
            encoded_demo = self.vpt.encode_trajectory(video, tolist=True)

            encoded_demo_json = {"demo_id": unique_id, "encoded_demo": encoded_demo, "actions": jsonl}

            # Save the encoded demonstration to disk
            with open(os.path.join(save_dir, unique_id + ".pkl"), "wb") as f:
                pickle.dump(encoded_demo_json, f)

    
    def load_encoded_demos_to_situations(self, save_dir="data/MakeWaterfallEncoded/", num=None, window_size=128, stride=2):
        '''Load encoded demonstrations from disk and create situations'''
        
        situation_weights = generate_situation_weights()

        situations = []
        for pkl_path in tqdm(glob.glob(os.path.join(save_dir, "*.pkl"))[:num], desc="Loading encoded demonstrations"):
            with open(pkl_path, "rb") as f:
                demo = pickle.load(f)
                for i in range(
                    window_size, len(demo["encoded_demo"]) - window_size, stride
                ):  
                    # Get the situation embeddings and actions
                    situation_embeds = np.array(demo['encoded_demo'][i-(window_size-1):i+1])
                    situation_actions = self.action_processor.json_to_action_vector(demo["actions"][i-(window_size-1):i+1])
                    next_action = self.action_processor.json_to_action_vector([demo["actions"][i]])
                    
                    # Take a weighted average of the situation embeddings
                    situation = np.average(situation_embeds, axis=0, weights=situation_weights)
                    
                    situations.append(
                        {
                            "demo_id": demo["demo_id"],
                            "sit_frame_idx": i, # Frame index of the situation in the video
                            "situation": situation,
                            "situation_actions": situation_actions,
                            "next_action": next_action
                        }
                    )

        return situations

    def _load_video(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (128, 128), interpolation=cv2.INTER_LINEAR)
            frames.append(frame)
        cap.release()
        return frames

    def _load_jsonl(self, jsonl_path):
        with open(jsonl_path) as f:
            return [json.loads(line) for line in f]

    def save_situations(self, situations, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(situations, f)

    def load_situations(self, save_path):
        with open(save_path, "rb") as f:
            return pickle.load(f)


class Memory:
    """FAISS based Memory class for indexing and retrieving situations"""

    def create_index(self, situations):
        self.index = faiss.IndexFlatL2(1024)
        self.index.add(self._create_situation_array(situations))

        # Store the situations without the situation embeddings
        self.situations_meta = [{k: v for k, v in s.items() if k != "situation"} for s in situations]

    def search(self, query, k=4):
        distances, nearest_indices = self.index.search(query.reshape(1, 1024), k)
        result = []
        for i, idx in enumerate(nearest_indices[0]):
            result.append(
                {   
                    "idx": int(idx),
                    "distance": distances[0][i],
                    **self.situations_meta[idx],
                    'embedding': self.index.reconstruct(int(idx))
                }
            )
        return result

    def save_index(self, save_dir, filename="memory"):
        os.makedirs(save_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(save_dir, filename + ".index"))

        with open(os.path.join(save_dir, filename + ".json"), "w") as f:
            json.dump(
                {
                    "index_file": filename + ".index",
                    "situations_meta": self.situations_meta,
                },
                f,
            )

    def load_index(self, json_path):
        with open(json_path, "r") as f:
            situations_meta = json.load(f)

        self.situations_meta = situations_meta["situations_meta"]
        self.index = faiss.read_index(
            os.path.join(os.path.dirname(json_path), situations_meta["index_file"])
        )

    def _create_situation_array(self, situations):
        """Create numpy array of situation latents from situations dictionary"""
        situation_latents = np.array([x["situation"] for x in situations])
        return situation_latents
