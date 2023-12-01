# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import openai
import dotenv
import os
import json
from stage import Stage
from config import config
from llm import llm
from loguru import logger
from pipelinecontext import PipelineContext
from dacite import from_dict
from utilities import dataclass_to_dict

class AnalyzerStage(Stage):

    def __repr__(self) -> str:
        return 'AnalyzerStage'

    def __str__(self) -> str:
        return self.__repr__()

    def _checkpoint(self, *args, context: PipelineContext, **kwargs) -> bool:
        file_path = self._checkpoint_path(context)
        if os.path.isfile(file_path):
            return True
        with open(file_path, 'w') as f:
            f.write(json.dumps(dataclass_to_dict(context)))
        return True

    def _load_checkpoint(self, context: PipelineContext) -> PipelineContext:
        checkpoint_path = self._checkpoint_path(context)
        if not os.path.isfile(checkpoint_path):
            return context

        logger.debug(f'Checkpoint found for stage: {self.__class__}')
        with open(checkpoint_path, 'r') as f:
            loaded_context = from_dict(data=json.loads(f.read()), data_class=PipelineContext)

        scenes_to_be_proccessed = []
        for scene in loaded_context.segmentation_analysis.scenes:
            potential_scene_path = self._get_response_output_path(loaded_context, sceneIndex=scene.index)
            if not os.path.isfile(potential_scene_path):
                scenes_to_be_proccessed.append(scene)
        loaded_context.segmentation_analysis.scenes = scenes_to_be_proccessed
        return loaded_context

    def _get_response_output_path(self, context, sceneIndex): 
        filename = "1_analysis_stage.json"
        scene_dir = os.path.join(config.server_root, config.stories_dir, context.id, str(sceneIndex))
        file = os.path.join(scene_dir, filename)
        return file
    
    def _save_response_output(self, response_payload, context, sceneIndex):
       
        file = self._get_response_output_path(context, sceneIndex)
        # Convert the string representation of JSON to a Python dictionary
        json_data = json.loads(response_payload)
        
        with open(file, "w") as f:
            json.dump(json_data, f, indent=4)

    def _process(self, context):
        for scene in context.segmentation_analysis.scenes:
            analyis = llm.analize_scene(scene)
            try:
                self._save_response_output(analyis, context, scene.index)
                logger.debug("Successfully produced analysis for scene " + str(scene.index))
                self._checkpoint(context=context, scene=scene)
            except:
                print("Encountered an error processing story " + str(scene.index) + " skipping this story")
        return context