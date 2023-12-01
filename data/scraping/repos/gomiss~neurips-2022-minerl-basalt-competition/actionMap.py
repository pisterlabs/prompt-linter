import minerl.herobraine.hero.mc as mc
import numpy as np

from collections import OrderedDict
import itertools
from typing import Dict, List

import numpy as np
from gym3.types import DictType, Discrete, TensorType
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from openai_vptV3.lib.actions import Buttons
from openai_vptV3.lib.action_mapping import CameraHierarchicalMapping
from openai_vptV3.lib.actions import ActionTransformer

ACTION_TRANSFORMER_KWARGS = dict(
    camera_binsize=2,
    camera_maxval=10,
    camera_mu=10,
    camera_quantization_scheme="mu_law",
)



class CameraHierarchicalMappingV2(CameraHierarchicalMapping):
    BUTTONS_GROUPS = OrderedDict(
        hotbar=["none"] + [f"hotbar.{i}" for i in range(1, 10)],
        fore_back_left_right_use_drop_attack_hotbar = ["none", "drop", "attack", "back", "left", "right", "forward", "jump","use"]
    )
    BUTTONS_GROUPS["camera"] = ["none", "camera"]
    BUTTONS_COMBINATIONS = list(itertools.product(*BUTTONS_GROUPS.values())) + ["inventory"]
    BUTTONS_COMBINATION_TO_IDX = {comb: i for i, comb in enumerate(BUTTONS_COMBINATIONS)}
    BUTTONS_IDX_TO_COMBINATION = {i: comb for i, comb in enumerate(BUTTONS_COMBINATIONS)}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class CameraHierarchicalMappingMoving(CameraHierarchicalMapping):
    BUTTONS_GROUPS = OrderedDict(
        fore_back=["none", "forward", "back"],
        left_right=["none", "left", "right"],
        jump=["none", "jump"],
    )
    BUTTONS_GROUPS["camera"] = ["none", "camera"]
    BUTTONS_COMBINATIONS = list(itertools.product(*BUTTONS_GROUPS.values()))
    BUTTONS_COMBINATION_TO_IDX = {comb: i for i, comb in enumerate(BUTTONS_COMBINATIONS)}
    BUTTONS_IDX_TO_COMBINATION = {i: comb for i, comb in enumerate(BUTTONS_COMBINATIONS)}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

if __name__=="__main__":
    
    env_action = {
        'ESC': 0, 
        'back': 0, 
        'drop': 0, 
        'forward': 0, 
        'hotbar.1': 0, 'hotbar.2': 0, 'hotbar.3': 0, 'hotbar.4': 0, 'hotbar.5': 0, 'hotbar.6': 0, 'hotbar.7': 0, 'hotbar.8': 0, 'hotbar.9': 0, 
        'inventory': 0, 
        'jump': 1, 
        'left': 0, 
        'right': 0, 
        'sneak': 0, 
        'sprint': 0, 
        'swapHands': 0, 
        'camera': np.array([-0.15,  0.  ]), 
        'attack': 0, 
        'use': 0, 
        'pickItem': 0
    }

    action_transformer = ActionTransformer(**ACTION_TRANSFORMER_KWARGS)

    policy_action = action_transformer.env2policy(env_action)
    
    print(f"policy_action:{policy_action}")



    if policy_action["camera"].ndim == 1:
        policy_action = {k: v[None] for k, v in policy_action.items()}
    action_mapper = CameraHierarchicalMappingV2(n_camera_bins=11)
    agent_action = action_mapper.from_factored(policy_action)
    print(f"agent_action: {agent_action}")
    print(action_mapper.BUTTONS_COMBINATION_TO_IDX)
    # action_transformer.action